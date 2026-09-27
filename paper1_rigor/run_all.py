"""Run every rigor experiment in order, capturing each run's full output to results_rigor/logs/.

Output is captured and written after the fact (rather than piped straight to a file handle) so that
a crashed or killed child still leaves its traceback on disk -- an empty log file and a non-zero
return code is the one failure mode that tells you nothing.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results_rigor" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

ORDER = ["r00_verify", "r05_lod_ci", "r01_matched_mlp", "r02_ablation",
         "r03_stability", "r06_shap", "r07_robustness", "r08_lod_cv", "r04_generalisation"]


def run(mod):
    log = LOGS / f"{mod}.log"
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {mod}", flush=True)
    proc = subprocess.run([sys.executable, "-u", "-m", f"paper1_rigor.{mod}"], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = proc.stdout.decode("utf-8", errors="replace")
    # pykan writes a per-step progress bar even with TQDM_DISABLE on some versions; drop those
    # lines so the log stays readable and small.
    keep = [ln for ln in text.splitlines() if "it/s" not in ln and "it]" not in ln]
    log.write_text("\n".join(keep) + f"\n[exit {proc.returncode}]\n", encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {mod} rc={proc.returncode} "
          f"({time.time() - t0:.0f}s) -> {log.name}", flush=True)
    if proc.returncode != 0:
        print("  last lines:", flush=True)
        for ln in keep[-12:]:
            print("   | " + ln, flush=True)
    return proc.returncode


if __name__ == "__main__":
    only = sys.argv[1:] or ORDER
    for mod in only:
        rc = run(mod)
        if rc != 0 and mod == "r00_verify":
            print("r00_verify failed -- stopping (the pipeline must reproduce the published table)",
                  flush=True)
            sys.exit(1)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)
