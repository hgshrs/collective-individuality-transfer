"""Download the OSF ZRHWT project files (public, no login) into data/raw/osf.
Skips very large ABM parameter-sweep outputs, prior-only brms fits and videos
(can be fetched later with --all). Code files are fetched first."""
import json, os, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/raw/osf")
lst = json.load(open(os.path.join(OUT, "_filelist.json")))
allf = "--all" in sys.argv

def skip(p):
    if allf: return False
    if "/parameter_sweep/" in p: return True
    if p.endswith("_prior.rds"): return True
    if p.startswith("/videos/"): return True
    if p.startswith("/figures/resource_discoveries/"): return True
    return False

def prio(p):
    if p.startswith("/code/") and "/fits/" not in p: return 0
    if p.startswith("/data/processed") or p.startswith("/data/raw_metadata") or p in ("/README.md", "/LICENSE", "/LICENSE-DATA"): return 1
    if p.startswith("/data/raw_traces"): return 2
    if p.startswith("/figures"): return 3
    return 4

todo = sorted([(p, s, u) for p, s, u in lst if not skip(p)], key=lambda t: prio(t[0]))
print(f"{len(todo)} files, {sum((s or 0) for _, s, _ in todo)/1e9:.2f} GB", flush=True)
log = open(os.path.join(ROOT, "logs/download.log"), "a")
for i, (p, s, u) in enumerate(todo):
    dst = os.path.join(OUT, p.lstrip("/"))
    if os.path.exists(dst) and (s is None or os.path.getsize(dst) == s):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = False
    for attempt in range(6):
        try:
            t0 = time.time()
            urllib.request.urlretrieve(u, dst + ".part")
            os.replace(dst + ".part", dst)
            msg = f"[{i+1}/{len(todo)}] {p} {os.path.getsize(dst)} B {time.time()-t0:.1f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            ok = True
            break
        except Exception as e:
            print("retry", p, e, flush=True)
            time.sleep(15 * (attempt + 1))
    if not ok:
        log.write(f"FAILED {p}\n"); log.flush()
    time.sleep(1.0)
print("DONE", flush=True); log.write("DONE\n")
