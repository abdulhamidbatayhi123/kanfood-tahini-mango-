"""Run one experiment module with a thread cap and a timestamped log."""
import os, sys, time, runpy, traceback
sys.path.insert(0, os.getcwd())
mod = sys.argv[1]
threads = sys.argv[2] if len(sys.argv) > 2 else "4"
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = threads
os.environ["TQDM_DISABLE"] = "1"
# joblib/loky otherwise sizes its worker pool from the machine's core count, once per library
# that asks for one. With several experiments running side by side that exhausted memory and
# left orphaned pools behind when a job was stopped.
os.environ["LOKY_MAX_CPU_COUNT"] = threads
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
try:                                   # otherwise a script that prints without flush=True shows
    sys.stdout.reconfigure(line_buffering=True)   # nothing in its log until the process ends
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
# The runner's own arguments must not reach the module: several use argparse and
# would reject them.
sys.argv = [mod] + sys.argv[3:]
t0 = time.time()
print(f"[bg] {mod} threads={threads} start", flush=True)
try:
    runpy.run_module(mod, run_name="__main__")
    rc = 0
except SystemExit as e:
    rc = int(e.code or 0)
except Exception:
    traceback.print_exc(); rc = 1
print(f"[bg] {mod} rc={rc} ({time.time()-t0:.0f}s)", flush=True)
sys.exit(rc)
