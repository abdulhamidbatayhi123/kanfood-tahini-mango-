"""Run a list of experiment modules sequentially, capturing a log per module.

Named `_runqueue`, not `_queue`: a module called `_queue` in this directory shadows the CPython
builtin of that name, and every job that imports pandas from here then dies inside `queue.py`.

Used so several long experiments can be queued behind one another without oversubscribing the
machine: each queue is one process, and several queues run in parallel with a thread cap each.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results_rigor" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

threads = sys.argv[1]
mods = sys.argv[2:]
runner = str(Path(__file__).with_name("_bg.py"))
qlog = LOGS / f"_queue_{os.getpid()}.log"

with open(qlog, "w", encoding="utf-8") as q:
    for mod in mods:
        name = mod.split(".")[-1].split("_")[0]
        q.write(f"[{time.strftime('%H:%M:%S')}] START {mod}\n")
        q.flush()
        t0 = time.time()
        with open(LOGS / f"{name}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, runner, mod, threads],
                                 cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        q.write(f"[{time.strftime('%H:%M:%S')}] DONE  {mod} rc={rc} ({time.time() - t0:.0f}s)\n")
        q.flush()
    q.write("QUEUE DONE\n")
