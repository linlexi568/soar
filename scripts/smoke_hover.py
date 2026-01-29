# Minimal smoke test for MathProgramController + Isaac Gym
# Run with the project venv python: /home/linlexi/桌面/soar/.venv/bin/python scripts/smoke_hover.py

import sys, os, importlib.util, traceback, types
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Prefer local isaacgym python bindings if present
isaac_py = os.path.join(root, 'isaacgym', 'python')
if os.path.isdir(isaac_py):
    sys.path.insert(0, isaac_py)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'utilities'))

print('PYTHONPATH prepared; attempting to load modules...')

try:
    # Create a synthetic package 'soar' so relative imports (from .dsl) work
    # when loading modules via file paths from '01_soar'.
    if 'soar' not in sys.modules:
        pkg = types.ModuleType('soar')
        pkg.__path__ = [os.path.join(root, '01_soar')]
        sys.modules['soar'] = pkg

    # Load DSL and controller modules
    spec = importlib.util.spec_from_file_location('soar.dsl', os.path.join(root, '01_soar', 'core', 'dsl.py'))
    dsl = importlib.util.module_from_spec(spec)
    sys.modules['soar.dsl'] = dsl  # Register in sys.modules so relative imports work
    spec.loader.exec_module(dsl)  # type: ignore

    spec2 = importlib.util.spec_from_file_location('soar.program_executor', os.path.join(root, '01_soar', 'core', 'program_executor.py'))
    pe = importlib.util.module_from_spec(spec2)
    sys.modules['soar.program_executor'] = pe  # Register in sys.modules
    spec2.loader.exec_module(pe)  # type: ignore

    spec3 = importlib.util.spec_from_file_location('utilities.isaac_tester', os.path.join(root, 'utilities', 'isaac_tester.py'))
    isaac_tester = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(isaac_tester)  # type: ignore

    TerminalNode = dsl.TerminalNode
    BinaryOpNode = dsl.BinaryOpNode
    MathProgramController = pe.MathProgramController
    SimulationTester = isaac_tester.SimulationTester

    print('Modules loaded.')

    # Build a trivial hover program: always set u_fz = mg + k * pos_err_z
    m = 0.027
    g = 9.81
    mg = m * g
    k = 0.20
    cond = TerminalNode(1.0)  # always true
    expr = BinaryOpNode('+', TerminalNode(mg), BinaryOpNode('*', TerminalNode(k), TerminalNode('pos_err_z')))
    prog = [{'condition': cond, 'action': [BinaryOpNode('set', TerminalNode('u_fz'), expr)]}]

    ctrl = MathProgramController(program=prog, suppress_init_print=False)

    # reward weights
    weights = {
        'position_rmse': 1.0,
        'settling_time': 0.5,
        'control_effort': 0.1,
        'smoothness_jerk': 0.1,
        'gain_stability': 0.1,
        'saturation': 0.5,
        'peak_error': 0.5,
        'high_freq': 0.1
    }
    scenarios = []

    sim = SimulationTester(controller=ctrl, test_scenarios=scenarios, weights=weights, duration_sec=4, gui=False, in_memory=True, quiet=False)
    print('Starting simulation (4s)...')
    score = sim.run()
    print('FINAL SCORE:', score)

except Exception as e:
    print('SMOKE TEST FAILED:')
    traceback.print_exc()
    sys.exit(2)
