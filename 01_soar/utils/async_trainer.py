"""异步训练器：MCTS 与 NN 训练流水线并行

设计思路：
- MCTS loop 在主线程/进程运行，不断产生 (program, policy_target) 样本
- NN training 在后台线程运行，持续从 replay buffer 采样并更新网络
- 通过 queue/flag 进行轻量级同步，避免死锁和竞争

优势：
- GPU 在 MCTS 阶段不再完全闲置
- MCTS 不必等待 NN 训练完成
- 整体 wall-clock 时间减少

Trade-offs：
- 需要注意模型参数的读写同步（使用锁或定期 snapshot）
- Replay buffer 的线程安全
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, Callable
import torch


class AsyncNNTrainer:
    """异步神经网络训练器（后台线程版本）"""
    
    def __init__(self,
                 train_fn: Callable[[], Optional[Dict[str, float]]],
                 update_interval: float = 0.1,
                 max_queue_size: int = 100,
                 max_steps_per_iter: Optional[int] = None):
        """
        Args:
            train_fn: 训练函数，每次调用执行一步训练，返回 metrics 或 None
            update_interval: 训练步之间的最小间隔（秒），防止过度抢占
            max_queue_size: 任务队列最大长度
        """
        self.train_fn = train_fn
        self.update_interval = update_interval
        self.max_queue_size = max_queue_size
        self.max_steps_per_iter = max_steps_per_iter
        
        # 线程控制
        self.worker_thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.pause_flag.set()  # 默认不暂停
        
        # 任务队列（可选，用于接收外部触发）
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        
            # 统计
        self.total_steps = 0
        self.total_time = 0.0
        self.last_metrics = None
        self.steps_in_current_iter = 0
        
        # 线程安全的 metrics 缓存
        self._metrics_lock = threading.Lock()
        # 单步执行锁：用于 sync pause 等待当前训练完成
        self._step_lock = threading.Lock()
        
    def start(self):
        """启动后台训练线程"""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return  # 已经在运行
        
        self.stop_flag.clear()
        self.pause_flag.set()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def reset_iter(self):
        """重置当前迭代的训练步数统计（由主线程控制节奏）。"""
        self.steps_in_current_iter = 0
        
    def stop(self, wait: bool = True):
        """停止后台训练线程
        
        Args:
            wait: 是否等待线程完全退出
        """
        self.stop_flag.set()
        if wait and self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
            
    def pause(self):
        """暂停训练（保持线程但不执行）"""
        self.pause_flag.clear()
        
    def pause_and_wait(self, poll_interval: float = 0.02):
        """暂停并阻塞，直到当前训练步完成。"""
        self.pause()
        # 等待 worker 完成正在进行的 train_fn
        while self._step_lock.locked() and not self.stop_flag.is_set():
            time.sleep(poll_interval)

    def resume(self):
        """恢复训练"""
        self.pause_flag.set()
        
    def _worker_loop(self):
        """后台训练循环"""
        while not self.stop_flag.is_set():
            # 若设置了步数上限且当前轮已耗尽，则等待主线程重置
            if self.max_steps_per_iter is not None and \
               self.steps_in_current_iter >= self.max_steps_per_iter:
                time.sleep(self.update_interval)
                continue

            # 检查暂停标志
            if not self.pause_flag.is_set():
                time.sleep(0.1)
                continue
            
            try:
                start_time = time.time()
                
                # 执行一步训练
                with self._step_lock:
                    metrics = self.train_fn()
                
                elapsed = time.time() - start_time
                self.total_time += elapsed
                self.total_steps += 1
                self.steps_in_current_iter += 1
                
                # 更新 metrics
                if metrics is not None:
                    with self._metrics_lock:
                        self.last_metrics = metrics
                
                # 控制训练频率，避免过度抢占 CPU/GPU
                if elapsed < self.update_interval:
                    time.sleep(self.update_interval - elapsed)
                    
            except Exception as e:
                # 训练出错，记录并继续（避免线程崩溃）
                print(f"  ⚠️  异步训练出错: {e}")
                time.sleep(1.0)
                
    def get_metrics(self) -> Optional[Dict[str, float]]:
        """获取最新的训练 metrics（线程安全）"""
        with self._metrics_lock:
            return self.last_metrics.copy() if self.last_metrics else None
            
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'total_steps': self.total_steps,
            'total_time': self.total_time,
            'avg_time_per_step': self.total_time / max(1, self.total_steps),
            'steps_in_current_iter': self.steps_in_current_iter,
            'max_steps_per_iter': self.max_steps_per_iter,
            'is_running': self.worker_thread is not None and self.worker_thread.is_alive(),
            'is_paused': not self.pause_flag.is_set()
        }


class SyncTrainer:
    """同步训练器（保持向后兼容的包装器）"""
    
    def __init__(self, train_fn: Callable[[], Optional[Dict[str, float]]]):
        self.train_fn = train_fn
        self.total_steps = 0
        
    def start(self):
        """空操作（同步模式不需要启动）"""
        pass
    
    def stop(self, wait: bool = True):
        """空操作（同步模式不需要停止）"""
        pass
    
    def pause(self):
        """空操作"""
        pass
    
    def resume(self):
        """空操作"""
        pass
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """执行一步训练（同步调用）"""
        metrics = self.train_fn()
        if metrics is not None:
            self.total_steps += 1
        return metrics
    
    def get_metrics(self) -> Optional[Dict[str, float]]:
        """同步模式下返回 None（metrics 由外部管理）"""
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'is_running': False,
            'is_paused': False,
            'mode': 'synchronous'
        }


def create_trainer(train_fn: Callable[[], Optional[Dict[str, float]]],
                   async_mode: bool = False,
                   **kwargs) -> Any:
    """创建训练器（工厂函数）
    
    Args:
        train_fn: 训练函数
        async_mode: True=异步模式，False=同步模式
        **kwargs: 传递给 AsyncNNTrainer 的额外参数
        
    Returns:
        AsyncNNTrainer 或 SyncTrainer 实例
    """
    if async_mode:
        return AsyncNNTrainer(train_fn, **kwargs)
    else:
        return SyncTrainer(train_fn)
