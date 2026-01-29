#!/usr/bin/env python3
"""单元测试：验证 DSL V2 参数化重构的正确性"""
#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import unittest

# 使用相对导入
import importlib
dsl_module = importlib.import_module('01_soar.core.dsl')
ser_module = importlib.import_module('01_soar.core.serialization')
bo_module = importlib.import_module('01_soar.utils.bayesian_tuner')

TerminalNode = dsl_module.TerminalNode
ConstantNode = dsl_module.ConstantNode
UnaryOpNode = dsl_module.UnaryOpNode
BinaryOpNode = dsl_module.BinaryOpNode
IfNode = dsl_module.IfNode

serialize_ast = ser_module.serialize_ast
deserialize_ast = ser_module.deserialize_ast
serialize_program = ser_module.serialize_program
deserialize_program = ser_module.deserialize_program

extract_tunable_params = bo_module.extract_tunable_params
inject_tuned_params = bo_module.inject_tuned_params


class TestConstantNode(unittest.TestCase):
    """测试 ConstantNode 基本功能"""
    
    def test_creation(self):
        node = ConstantNode(0.5, name='alpha', min_val=0.0, max_val=1.0)
        self.assertEqual(node.value, 0.5)
        self.assertEqual(node.name, 'alpha')
        self.assertEqual(node.min_val, 0.0)
        self.assertEqual(node.max_val, 1.0)
    
    def test_evaluate(self):
        node = ConstantNode(2.5)
        state = {'x': 1.0}
        self.assertEqual(node.evaluate(state), 2.5)
    
    def test_string_repr(self):
        node1 = ConstantNode(0.123, name='kp')
        self.assertIn('kp', str(node1))
        self.assertIn('0.123', str(node1))
        
        node2 = ConstantNode(0.456)
        self.assertIn('0.456', str(node2))


class TestUnaryOpNodeParams(unittest.TestCase):
    """测试 UnaryOpNode 新参数系统"""
    
    def test_new_format_with_params(self):
        """测试新格式：显式 params 字典"""
        child = TerminalNode('x')
        params = {'alpha': ConstantNode(0.3)}
        node = UnaryOpNode('ema', child, params)
        
        self.assertEqual(node.op, 'ema')
        self.assertEqual(node.get_param('alpha', 0.2), 0.3)
    
    def test_legacy_format_compatibility(self):
        """测试旧格式兼容性：op='ema:0.3'"""
        child = TerminalNode('x')
        node = UnaryOpNode('ema:0.3', child)
        
        self.assertEqual(node.op, 'ema')
        self.assertEqual(node.get_param('alpha', 0.2), 0.3)
    
    def test_evaluate_with_params(self):
        """测试带参数的求值"""
        child = TerminalNode(1.0)
        params = {'alpha': ConstantNode(0.5)}
        node = UnaryOpNode('ema', child, params)
        
        state = {}
        # EMA: y = (1-α)*prev + α*x, prev初始为0
        result1 = node.evaluate(state)
        self.assertAlmostEqual(result1, 0.5 * 1.0, places=5)
        
        result2 = node.evaluate(state)
        expected = (1 - 0.5) * result1 + 0.5 * 1.0
        self.assertAlmostEqual(result2, expected, places=5)
    
    def test_multiple_params(self):
        """测试多参数算子（clamp）"""
        child = TerminalNode(10.0)
        params = {
            'lo': ConstantNode(-2.0),
            'hi': ConstantNode(2.0)
        }
        node = UnaryOpNode('clamp', child, params)
        
        state = {}
        result = node.evaluate(state)
        self.assertEqual(result, 2.0)  # 10.0 被裁剪到 2.0


class TestSerialization(unittest.TestCase):
    """测试序列化/反序列化"""
    
    def test_constant_node_serialization(self):
        """测试 ConstantNode 序列化"""
        node = ConstantNode(1.5, name='kp', min_val=0.5, max_val=3.0)
        serialized = serialize_ast(node)
        
        self.assertEqual(serialized['type'], 'Constant')
        self.assertEqual(serialized['value'], 1.5)
        self.assertEqual(serialized['name'], 'kp')
        self.assertEqual(serialized['min_val'], 0.5)
        self.assertEqual(serialized['max_val'], 3.0)
        
        deserialized = deserialize_ast(serialized)
        self.assertIsInstance(deserialized, ConstantNode)
        self.assertEqual(deserialized.value, 1.5)
        self.assertEqual(deserialized.name, 'kp')
    
    def test_unary_with_params_serialization(self):
        """测试带参数的 UnaryOpNode 序列化"""
        child = TerminalNode('x')
        params = {'alpha': ConstantNode(0.25, name='ema_alpha')}
        node = UnaryOpNode('ema', child, params)
        
        serialized = serialize_ast(node)
        self.assertEqual(serialized['type'], 'Unary')
        self.assertEqual(serialized['op'], 'ema')
        self.assertIn('params', serialized)
        self.assertIn('alpha', serialized['params'])
        
        deserialized = deserialize_ast(serialized)
        self.assertIsInstance(deserialized, UnaryOpNode)
        self.assertEqual(deserialized.op, 'ema')
        self.assertIn('alpha', deserialized.params)
        self.assertIsInstance(deserialized.params['alpha'], ConstantNode)
        self.assertEqual(deserialized.params['alpha'].value, 0.25)
    
    def test_backward_compatibility(self):
        """测试旧格式加载（无 params 字段）"""
        old_format = {
            'type': 'Unary',
            'op': 'deadzone:0.05',
            'child': {'type': 'Terminal', 'value': 'pos_err'}
        }
        
        node = deserialize_ast(old_format)
        self.assertIsInstance(node, UnaryOpNode)
        # 旧格式应该被解析
        self.assertEqual(node.get_param('eps', 0.01), 0.05)


class TestBayesianTunerIntegration(unittest.TestCase):
    """测试 BO 参数提取/注入"""
    
    def test_extract_constant_nodes(self):
        """测试提取 ConstantNode"""
        program = [{
            'condition': TerminalNode(1.0),
            'action': [
                UnaryOpNode(
                    'ema',
                    TerminalNode('x'),
                    params={'alpha': ConstantNode(0.3)}
                )
            ]
        }]
        
        params = extract_tunable_params(program)
        # 应该找到 alpha 参数
        param_dict = {path: val for path, val in params}
        alpha_key = [k for k in param_dict if 'param_alpha' in k][0]
        self.assertAlmostEqual(param_dict[alpha_key], 0.3, places=5)
    
    def test_inject_tuned_values(self):
        """测试注入优化后的参数"""
        program = [{
            'condition': TerminalNode(1.0),
            'action': [
                UnaryOpNode(
                    'deadzone',
                    TerminalNode('err'),
                    params={'eps': ConstantNode(0.05)}
                )
            ]
        }]
        
        # 提取路径
        params = extract_tunable_params(program)
        eps_path = [path for path, _ in params if 'param_eps' in path][0]
        
        # 注入新值
        tuned = {eps_path: 0.1}
        inject_tuned_params(program, tuned)
        
        # 验证
        new_val = program[0]['action'][0].params['eps'].value
        self.assertAlmostEqual(new_val, 0.1, places=5)


class TestExecutionCompatibility(unittest.TestCase):
    """测试执行器兼容性"""
    
    def test_mixed_format_execution(self):
        """测试新旧格式混合执行"""
        # 新格式
        new_node = UnaryOpNode(
            'ema',
            TerminalNode(5.0),
            params={'alpha': ConstantNode(0.4)}
        )
        
        # 旧格式
        old_node = UnaryOpNode('ema:0.4', TerminalNode(5.0))
        
        state = {}
        result_new = new_node.evaluate(state)
        result_old = old_node.evaluate(state)
        
        # 应该产生相同结果
        self.assertAlmostEqual(result_new, result_old, places=5)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestConstantNode))
    suite.addTests(loader.loadTestsFromTestCase(TestUnaryOpNodeParams))
    suite.addTests(loader.loadTestsFromTestCase(TestSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianTunerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestExecutionCompatibility))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
