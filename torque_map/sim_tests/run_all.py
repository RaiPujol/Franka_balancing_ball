"""Run every fake-robot test in parallel and print one PASS/FAIL line each. Usage: python3 run_all.py
(takes ~2 min; needs only numpy). TM_PKG=<dir> tests another copy of the torque_map package."""
import glob, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
tests = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
t0 = time.time()
procs = {t: subprocess.Popen([sys.executable, t], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
         for t in tests}
fails = 0
for t, p in procs.items():
    out = p.communicate()[0].strip().splitlines()
    line = next((l for l in reversed(out) if l.startswith(("PASS", "FAIL"))), "FAIL: no result\n" + "\n".join(out[-5:]))
    fails += not line.startswith("PASS")
    print(f"{os.path.basename(t):18s} {line}")
print(f"{len(tests) - fails}/{len(tests)} passed in {time.time() - t0:.0f} s")
sys.exit(1 if fails else 0)
