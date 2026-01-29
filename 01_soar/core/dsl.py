"""DSL 节点定义 + 可选算子级性能剖析 (OP_PROFILE=1 启用)"""
import numpy as np, math, abc, os, time
from collections import deque

SAFE_VALUE_MIN = -6.0
SAFE_VALUE_MAX = 6.0
MAX_DELAY_STEPS = 3
MAX_DIFF_STEPS = 3
MIN_EMA_ALPHA = 0.05
MAX_EMA_ALPHA = 0.8
MAX_RATE_LIMIT = 1.0
TERMINAL_VALUE_MIN = -3.0
TERMINAL_VALUE_MAX = 3.0
MAX_SMOOTH_SCALE = 2.0

def _clamp_value(v: float) -> float:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
    return float(min(max(v, SAFE_VALUE_MIN), SAFE_VALUE_MAX))

def _safe_get_state(state_dict: dict, key: str) -> float:
    val = state_dict.get(key, 0.0)
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(SAFE_VALUE_MIN, min(SAFE_VALUE_MAX, float(val)))
    return 0.0

class ProgramNode(abc.ABC):
    def evaluate(self, state_dict: dict) -> float: ...
    def __str__(self): return 'ProgramNode'

class TerminalNode(ProgramNode):
    def __init__(self,value): self.value=value
    def evaluate(self,state_dict):
        if isinstance(self.value,str): return _safe_get_state(state_dict,self.value)
        if isinstance(self.value,(int,float)):
            val=float(self.value)
            val=max(TERMINAL_VALUE_MIN, min(TERMINAL_VALUE_MAX, val))
            return val
        return 0.0
    def __str__(self):
        if isinstance(self.value,float): return f"{self.value:.2f}"
        return str(self.value)

class ConstantNode(ProgramNode):
    """显式常量节点，用于可调参数（BO/GNN 优化目标）"""
    def __init__(self, value: float, name: str = None, min_val: float = None, max_val: float = None):
        self.value = float(value)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
    def evaluate(self, state_dict):
        return float(self.value)
    def __str__(self):
        if self.name:
            return f"Const({self.name}={self.value:.3f})"
        return f"{self.value:.3f}"

class UnaryOpNode(ProgramNode):
    def __init__(self, op, child, params=None):
        self.op = op
        self.child = child
        self.params = params or {}
        if ':' in op and not params:
            parts = op.split(':')
            self.op = parts[0]
            self._legacy_params = parts[1:]
        else:
            self._legacy_params = None

    def get_param(self, name, default_value, min_val=None, max_val=None):
        if name in self.params:
            param = self.params[name]
            if isinstance(param, ConstantNode):
                return param.value
            elif isinstance(param, (int, float)):
                return float(param)
        if self._legacy_params:
            param_map = {
                'ema': {'alpha': 0},
                'delay': {'k': 0},
                'diff': {'k': 0},
                'clamp': {'lo': 0, 'hi': 1},
                'deadzone': {'eps': 0},
                'rate': {'r': 0},
                'rate_limit': {'r': 0},
                'smooth': {'s': 0},
                'smoothstep': {'s': 0}
            }
            if self.op in param_map and name in param_map[self.op]:
                idx = param_map[self.op][name]
                if idx < len(self._legacy_params):
                    return float(self._legacy_params[idx])
        return default_value

    def evaluate(self,sd):
        v = self.child.evaluate(sd)
        _profile_enabled = os.getenv('OP_PROFILE','0') in ('1','true','True')
        _t0 = time.perf_counter() if _profile_enabled else None
        op = self.op

        def clamp_result(val):
            return _clamp_value(val)

        if op=='abs':
            res = clamp_result(abs(v))
            if _profile_enabled: _op_profile_record('abs', _t0)
            return res
        if op=='sign':
            res = clamp_result(float(np.sign(v)))
            if _profile_enabled: _op_profile_record('sign', _t0)
            return res
        if op=='sin':
            res = clamp_result(math.sin(v))
            if _profile_enabled: _op_profile_record('sin', _t0)
            return res
        if op=='cos':
            res = clamp_result(math.cos(v))
            if _profile_enabled: _op_profile_record('cos', _t0)
            return res
        if op=='tan':
            try: val=math.tan(v)
            except Exception: val=0.0
            res = clamp_result(val)
            if _profile_enabled: _op_profile_record('tan', _t0)
            return res
        if op=='log1p':
            try: res = clamp_result(math.log1p(abs(v)))
            except Exception: res = 0.0
            if _profile_enabled: _op_profile_record('log1p', _t0)
            return res
        if op=='sqrt':
            try: res = clamp_result(math.sqrt(abs(v)))
            except Exception: res = 0.0
            if _profile_enabled: _op_profile_record('sqrt', _t0)
            return res

        prefix = self.op

        if prefix=='ema':
            alpha = self.get_param('alpha', 0.2, MIN_EMA_ALPHA, MAX_EMA_ALPHA)
            alpha = min(max(alpha, MIN_EMA_ALPHA), MAX_EMA_ALPHA)
            if not hasattr(self, '_ema_prev'): self._ema_prev = 0.0
            y = (1.0 - alpha) * self._ema_prev + alpha * v
            self._ema_prev = clamp_result(y)
            if _profile_enabled: _op_profile_record('ema', _t0)
            return self._ema_prev

        if prefix=='delay':
            k = int(self.get_param('k', 1, 1, MAX_DELAY_STEPS))
            k = max(1, min(MAX_DELAY_STEPS, k))
            if not hasattr(self, '_buf'): self._buf = deque(maxlen=k)
            if isinstance(self._buf, deque) and self._buf.maxlen != k:
                self._buf = deque(list(self._buf), maxlen=k)
            out = self._buf[0] if len(self._buf) == k else 0.0
            self._buf.appendleft(v)
            res = clamp_result(float(out))
            if _profile_enabled: _op_profile_record('delay', _t0)
            return res

        if prefix=='diff':
            k = int(self.get_param('k', 1, 1, MAX_DIFF_STEPS))
            k = max(1, min(MAX_DIFF_STEPS, k))
            if not hasattr(self, '_buf_d'): self._buf_d = deque(maxlen=k)
            if isinstance(self._buf_d, deque) and self._buf_d.maxlen != k:
                self._buf_d = deque(list(self._buf_d), maxlen=k)
            prev = self._buf_d[0] if len(self._buf_d) == k else v
            self._buf_d.appendleft(v)
            res = clamp_result(float(v - prev))
            if _profile_enabled: _op_profile_record('diff', _t0)
            return res

        if prefix=='clamp':
            lo = self.get_param('lo', -5.0, SAFE_VALUE_MIN, SAFE_VALUE_MAX)
            hi = self.get_param('hi', 5.0, SAFE_VALUE_MIN, SAFE_VALUE_MAX)
            lo = max(SAFE_VALUE_MIN, lo); hi = min(SAFE_VALUE_MAX, hi)
            if lo>hi: lo,hi = hi,lo
            res = float(min(max(v, lo), hi))
            if _profile_enabled: _op_profile_record('clamp', _t0)
            return res

        if prefix=='deadzone':
            eps = self.get_param('eps', 0.01, 0.0, 1.0)
            eps = min(max(0.0, eps), 1.0)
            if abs(v) <= eps: res = 0.0
            else: res = clamp_result(float(v - math.copysign(eps, v)))
            if _profile_enabled: _op_profile_record('deadzone', _t0)
            return res

        if prefix=='rate' or prefix=='rate_limit':
            r = self.get_param('r', 1.0, 0.01, MAX_RATE_LIMIT)
            r = min(max(0.01, r), MAX_RATE_LIMIT)
            if not hasattr(self, '_rate_prev'): self._rate_prev = 0.0
            y_prev = self._rate_prev
            lo = y_prev - r; hi = y_prev + r
            y = min(max(v, lo), hi)
            self._rate_prev = clamp_result(y)
            if _profile_enabled: _op_profile_record('rate', _t0)
            return self._rate_prev

        if prefix=='smooth' or prefix=='smoothstep':
            s = self.get_param('s', 1.0, 1e-3, MAX_SMOOTH_SCALE)
            s = min(max(1e-3, s), MAX_SMOOTH_SCALE)
            res = clamp_result(float(s * math.tanh(v / s)))
            if _profile_enabled: _op_profile_record('smooth', _t0)
            return res

        raise ValueError('未知的一元操作:'+self.op)
    def __str__(self): return f"{self.op}({self.child})"

class BinaryOpNode(ProgramNode):
    def __init__(self,op,left,right): self.op=op; self.left=left; self.right=right
    def evaluate(self,sd):
        l=self.left.evaluate(sd); r=self.right.evaluate(sd)
        if self.op=='+': return _clamp_value(l+r)
        if self.op=='-': return _clamp_value(l-r)
        if self.op=='/':
            try: val = l/ (r if abs(r) > 1e-6 else (1e-6 if r>=0 else -1e-6))
            except Exception: val = 0.0
            return _clamp_value(val)
        if self.op=='>': return 1.0 if l>r else 0.0
        if self.op=='<': return 1.0 if l<r else 0.0
        if self.op=='max': return _clamp_value(max(l,r))
        if self.op=='min': return _clamp_value(min(l,r))
        if self.op=='*': return _clamp_value(l*r)
        if self.op=='==': return 1.0 if l==r else 0.0
        if self.op=='!=': return 1.0 if l!=r else 0.0
        raise ValueError('未知的二元操作:'+self.op)
    def __str__(self): return f"({self.left} {self.op} {self.right})"

class IfNode(ProgramNode):
    def __init__(self,condition,then_branch,else_branch): self.condition=condition; self.then_branch=then_branch; self.else_branch=else_branch
    def evaluate(self,sd):
        val = self.then_branch.evaluate(sd) if self.condition.evaluate(sd)>0 else self.else_branch.evaluate(sd)
        return _clamp_value(val)
    def __str__(self): return f"if {self.condition} then ({self.then_branch}) else ({self.else_branch})"


def reset_node_state(node) -> None:
    """递归重置 AST 节点的内部状态（ema, delay, diff, rate 等时间算子的缓冲区）。
    
    在每次评估程序前调用此函数，确保评估结果是确定性的、可复现的。
    支持 AST 对象和 JSON dict 两种格式。
    """
    if node is None:
        return
    
    # 如果是 JSON dict，跳过（dict 没有运行时状态）
    if isinstance(node, dict):
        # JSON dict 格式没有运行时状态缓冲区，递归处理子节点
        node_type = node.get('type', '')
        if node_type == 'Unary':
            reset_node_state(node.get('child'))
        elif node_type == 'Binary':
            reset_node_state(node.get('left'))
            reset_node_state(node.get('right'))
        elif node_type == 'If':
            reset_node_state(node.get('condition'))
            reset_node_state(node.get('then'))
            reset_node_state(node.get('else'))
        return
    
    # 重置 UnaryOpNode 的状态缓冲区
    if isinstance(node, UnaryOpNode):
        # ema 状态
        if hasattr(node, '_ema_prev'):
            node._ema_prev = 0.0
        # delay 状态
        if hasattr(node, '_buf'):
            node._buf.clear()
        # diff 状态
        if hasattr(node, '_buf_d'):
            node._buf_d.clear()
        # rate/rate_limit 状态
        if hasattr(node, '_rate_prev'):
            node._rate_prev = 0.0
        # 递归处理子节点
        reset_node_state(node.child)
    
    # BinaryOpNode: 递归处理左右子节点
    elif isinstance(node, BinaryOpNode):
        reset_node_state(node.left)
        reset_node_state(node.right)
    
    # IfNode: 递归处理条件和两个分支
    elif isinstance(node, IfNode):
        reset_node_state(node.condition)
        reset_node_state(node.then_branch)
        reset_node_state(node.else_branch)


def reset_program_state(program) -> None:
    """重置整个程序（规则列表）中所有 AST 节点的状态。
    
    Args:
        program: 程序规则列表，每个规则包含 'condition' 和 'action'
                 支持 AST 对象和 JSON dict 两种格式
    """
    if not program:
        return
    
    for rule in program:
        # 重置条件节点
        cond = rule.get('condition')
        if cond is not None:
            reset_node_state(cond)
        
        # 重置动作节点
        for action in rule.get('action', []) or []:
            # AST 对象
            if hasattr(action, 'left'):
                reset_node_state(action.left)
            if hasattr(action, 'right'):
                reset_node_state(action.right)
            # JSON dict
            if isinstance(action, dict):
                reset_node_state(action.get('left'))
                reset_node_state(action.get('right'))
                reset_node_state(action.get('child'))


# --- 简易算子性能剖析支持 ---
_OP_PROFILE_STORE = {}

def _op_profile_record(name: str, t0: float):
    dt = time.perf_counter() - t0
    rec = _OP_PROFILE_STORE.get(name)
    if rec is None:
        _OP_PROFILE_STORE[name] = {'time': dt, 'count': 1}
    else:
        rec['time'] += dt
        rec['count'] += 1

def get_op_profile(reset: bool = False):
    out = {k: {'avg_us': (v['time']/v['count'])*1e6, 'count': v['count'], 'total_ms': v['time']*1e3} for k,v in _OP_PROFILE_STORE.items()}
    if reset:
        _OP_PROFILE_STORE.clear()
    return out
