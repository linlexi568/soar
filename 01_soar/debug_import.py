from tune_circle_pid_bo import create_pid_programs
try:
    prog_tx, prog_ty, prog_tz, prog_fz = create_pid_programs(1.0, 0.5, 0.4, 1.0, 0.01, 0.2)
    print("Success!")
    print("prog_ty:", prog_ty)
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
