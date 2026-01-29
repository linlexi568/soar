
import sys
import os
import numpy as np
import torch

# Add path to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '01_soar'))

from mcts_training.mcts import MCTS_Agent as MCTS
from core.program_executor import MathProgramController
from core.dsl import TerminalNode, ConstantNode, BinaryOpNode, UnaryOpNode
from utils.batch_evaluation import BatchEvaluator

def debug_pipeline():
    print("=== Debugging Pipeline ===")
    
    # 1. Generate a random program
    print("\n1. Generating random program...")
    mcts = MCTS(
        evaluation_function=lambda p: 0.0,
        dsl_variables=['STATE_ERR_P', 'STATE_ERR_D', 'STATE_ERR_I', 'STATE_FF'],
        dsl_constants=[1.0],
        dsl_operators=['+', '-', '*', 'set'],
        max_depth=2
    )
    
    # Force generation of u_generic
    program = mcts._generate_random_segmented_program()
    print(f"Generated Program (Raw):")
    print(mcts.program_to_str(program))
    
    # 2. Instantiate u_generic
    print("\n2. Instantiating u_generic...")
    # We need an instance of BatchEvaluator to use _instantiate_generic_program, 
    # or we can just copy the method. Let's try to use the class method if possible, 
    # but it's an instance method. I'll create a dummy instance or just copy the logic 
    # if instantiation is too heavy.
    
    # BatchEvaluator init is heavy (isaacgym). Let's copy the logic for lightweight test.
    instantiated_program = instantiate_generic_program_standalone(program)
    
    print(f"Instantiated Program:")
    # Helper to print instantiated program
    for i, rule in enumerate(instantiated_program):
        acts = rule.get('action', [])
        act_strs = []
        for act in acts:
            if isinstance(act, BinaryOpNode) and act.op == 'set':
                act_strs.append(f"{act.left} = {act.right}")
        print(f"  Rule {i}: {', '.join(act_strs)}")

    # 3. Execute with MathProgramController
    print("\n3. Executing with MathProgramController...")
    controller = MathProgramController(instantiated_program, suppress_init_print=False)
    
    # Create a dummy state
    # We need to simulate inputs that would produce non-zero values for the mapped variables
    # Mappings:
    # STATE_ERR_P -> err_p_roll, err_p_pitch, err_p_yaw, pos_err_z
    # STATE_ERR_D -> ang_vel_x, ang_vel_y, ang_vel_z, vel_z
    
    # Let's set some errors
    cur_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([0.0, 0.0, 1.0]) # pos_err_z = 1.0
    
    cur_quat = np.array([0.0, 0.0, 0.0, 1.0]) # Identity
    target_rpy = np.array([0.1, 0.1, 0.1]) # Small error in roll/pitch/yaw
    
    cur_vel = np.array([0.0, 0.0, -0.5]) # vel_z = -0.5
    cur_ang_vel = np.array([0.1, 0.1, 0.1]) # ang_vel = 0.1
    
    control_timestep = 0.01
    
    rpm, pos_e, rpy_e = controller.computeControl(
        control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy
    )
    
    print(f"\nInputs:")
    print(f"  pos_err_z: {target_pos[2] - cur_pos[2]}")
    print(f"  rpy_error: {target_rpy}")
    
    print(f"\nOutputs:")
    print(f"  RPM: {rpm}")
    
    if np.all(rpm == 0):
        print("\n[WARNING] RPM is all zeros!")
    else:
        print("\n[SUCCESS] RPM is non-zero.")

def instantiate_generic_program_standalone(program):
    # Copied logic from BatchEvaluator._instantiate_generic_program
    # Simplified for standalone test
    
    has_generic = False
    for rule in program:
        actions = rule.get('action', [])
        if not actions: continue
        first_action = actions[0]
        if isinstance(first_action, BinaryOpNode) and first_action.op == 'set' and \
           isinstance(first_action.left, TerminalNode) and first_action.left.value == 'u_generic':
            has_generic = True
            break
            
    if not has_generic:
        return program
        
    roll_map = {'STATE_ERR_P': 'err_p_roll', 'STATE_ERR_D': 'ang_vel_x', 'STATE_ERR_I': 'err_i_roll', 'STATE_FF': '0.0'}
    pitch_map = {'STATE_ERR_P': 'err_p_pitch', 'STATE_ERR_D': 'ang_vel_y', 'STATE_ERR_I': 'err_i_pitch', 'STATE_FF': '0.0'}
    yaw_map = {'STATE_ERR_P': 'err_p_yaw', 'STATE_ERR_D': 'ang_vel_z', 'STATE_ERR_I': 'err_i_yaw', 'STATE_FF': '0.0'}
    thrust_map = {'STATE_ERR_P': 'pos_err_z', 'STATE_ERR_D': 'vel_z', 'STATE_ERR_I': 'err_i_z', 'STATE_FF': 'hover_thrust'}
    
    def _process_node(node, mapping, suffix):
        if isinstance(node, TerminalNode):
            if node.value in mapping:
                val = mapping[node.value]
                try:
                    return ConstantNode(value=float(val), name=None)
                except ValueError:
                    return TerminalNode(value=val)
            return node
        elif isinstance(node, ConstantNode):
            new_name = f"{node.name}_{suffix}" if node.name else None
            return ConstantNode(value=node.value, name=new_name, min_val=node.min_val, max_val=node.max_val)
        elif isinstance(node, BinaryOpNode):
            return BinaryOpNode(node.op, _process_node(node.left, mapping, suffix), _process_node(node.right, mapping, suffix))
        elif isinstance(node, UnaryOpNode):
            return UnaryOpNode(node.op, _process_node(node.child, mapping, suffix), node.params)
        return node

    new_program = []
    # We assume single rule for u_generic in this test case, but loop to be safe
    for rule in program:
        actions = rule.get('action', [])
        if not actions: continue
        
        # Find the u_generic expression
        generic_expr = None
        for act in actions:
            if isinstance(act, BinaryOpNode) and act.op == 'set' and act.left.value == 'u_generic':
                generic_generic_expr = act.right
                break
        
        if generic_generic_expr:
            # Create 4 rules or 1 rule with 4 actions? 
            # The original code creates 1 rule with 4 actions if the input was 1 rule.
            # Actually, let's look at the original code again.
            # It seems it replaces the u_generic action with 4 specific actions in the SAME rule list if possible,
            # or creates a new program structure.
            # The original code iterates over rules and builds a new program.
            
            # Re-implementing the loop structure from original code:
            new_actions = []
            
            # 1. Roll
            expr_roll = _process_node(generic_generic_expr, roll_map, 'roll')
            new_actions.append(BinaryOpNode('set', TerminalNode('u_tx'), expr_roll))
            
            # 2. Pitch
            expr_pitch = _process_node(generic_generic_expr, pitch_map, 'pitch')
            new_actions.append(BinaryOpNode('set', TerminalNode('u_ty'), expr_pitch))
            
            # 3. Yaw
            expr_yaw = _process_node(generic_generic_expr, yaw_map, 'yaw')
            new_actions.append(BinaryOpNode('set', TerminalNode('u_tz'), expr_yaw))
            
            # 4. Thrust
            expr_thrust = _process_node(generic_generic_expr, thrust_map, 'thrust')
            new_actions.append(BinaryOpNode('set', TerminalNode('u_fz'), expr_thrust))
            
            new_program.append({'condition': rule.get('condition'), 'action': new_actions})
            
    return new_program

if __name__ == "__main__":
    debug_pipeline()
