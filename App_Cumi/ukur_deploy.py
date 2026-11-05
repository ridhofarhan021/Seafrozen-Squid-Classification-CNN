"""Ukur kinerja layanan grading cumi (FastAPI) untuk revisi paper ICIMCIS 2026.

Jalankan di laptop server, setelah server (main.py revisi) dan ngrok menyala:

    python ukur_deploy.py --images "D:\\...\\dataset\\test" --url http://127.0.0.1:8000 --url https://xxxx.ngrok-free.app

Yang diukur:
  - spesifikasi laptop (CPU, RAM, OS), versi TensorFlow, dan perangkat inferensi (dari /health)
  - waktu proses di server per request (decode + resize + inferensi), dari field processing_ms
  - waktu bolak-balik (round trip) per request untuk tiap URL
  - throughput dengan 1, 5, dan 10 request bersamaan
  - memori (RSS) dan CPU proses server; butuh psutil (pip install psutil)

Hasil lengkap disimpan ke hasil_ukur_deploy.json, dan kalimat siap tempel untuk paper dicetak di akhir.
Hanya memakai pustaka standar Python (psutil opsional).
"""
import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

try:
    import psutil
except ImportError:
    psutil = None

EXTS = {".jpg", ".jpeg", ".png"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
HEADERS = {"ngrok-skip-browser-warning": "1", "User-Agent": "ukur-deploy/1.0"}
NAN = float("nan")


# ---------- utilitas ----------
def pct(values, q):
    v = sorted(values)
    if not v:
        return NAN
    k = (len(v) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def med(values):
    return statistics.median(values) if values else NAN


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def cpu_name():
    system = platform.system()
    name = ""
    if system == "Windows":
        name = run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if not name:
            name = run_cmd(["wmic", "cpu", "get", "name"]).replace("Name", "")
    elif system == "Darwin":
        name = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        except OSError:
            pass
    lines = [ln.strip() for ln in name.splitlines() if ln.strip()]
    name = lines[0] if lines else (platform.processor() or "unknown CPU")
    return " ".join(name.replace("(R)", "").replace("(TM)", "").replace("(tm)", "").split())


def ram_gb():
    if psutil is not None:
        return psutil.virtual_memory().total / 1024 ** 3
    if platform.system() == "Windows":
        out = run_cmd(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        try:
            return int(out) / 1024 ** 3
        except ValueError:
            return NAN
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, AttributeError, OSError):
        return NAN


def is_local(url):
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


def opener_for(url):
    if is_local(url):   # jangan lewat proxy untuk alamat lokal
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


# ---------- HTTP ----------
def get_json(opener, url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def make_body(path):
    data = path.read_bytes()
    ext = path.suffix.lower()
    ctype = "image/png" if ext == ".png" else "image/jpeg"
    boundary = "----cumi" + uuid.uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="kamera{ext}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n").encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def post_once(opener, url, body, ctype, timeout=120):
    req = urllib.request.Request(url, data=body, headers={**HEADERS, "Content-Type": ctype}, method="POST")
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as r:
            payload = r.read()
        rt = (time.perf_counter() - t0) * 1000
        d = json.loads(payload.decode("utf-8"))
        return {"ok": True, "rt_ms": rt, "proc_ms": d.get("processing_ms"), "pred": d.get("prediction")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "rt_ms": (time.perf_counter() - t0) * 1000, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "rt_ms": (time.perf_counter() - t0) * 1000, "error": type(e).__name__}


# ---------- proses server (psutil) ----------
def find_server(port, pid=None):
    if psutil is None:
        return None
    if pid:
        return psutil.Process(pid)
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                return psutil.Process(c.pid)
    except (psutil.Error, PermissionError):
        pass
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if "uvicorn" in cmd and ("main:app" in cmd or str(port) in cmd):
            return p
    return None


def proc_tree(proc):
    try:
        return [proc] + proc.children(recursive=True)   # termasuk proses anak (mis. uvicorn --reload)
    except psutil.Error:
        return [proc]


def rss_mb(proc):
    total = 0
    for p in proc_tree(proc):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / 1024 ** 2


class Sampler(threading.Thread):
    """Mencatat memori (RSS) dan CPU proses server setiap 0,2 detik selama pengukuran."""

    def __init__(self, proc, interval=0.2):
        super().__init__(daemon=True)
        self.proc, self.interval = proc, interval
        self.rss, self.cpu = [], []
        self._halt = threading.Event()

    def run(self):
        procs = proc_tree(self.proc)
        for p in procs:
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        while not self._halt.wait(self.interval):
            rss, cpu = 0, 0.0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
                except psutil.Error:
                    pass
            self.rss.append(rss / 1024 ** 2)
            self.cpu.append(cpu)

    def stop(self):
        self._halt.set()
        self.join()


# ---------- pengukuran ----------
def run_level(opener, url, bodies, n, conc, proc):
    jobs = [bodies[i % len(bodies)] for i in range(n)]
    sampler = Sampler(proc) if proc is not None else None
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda b: post_once(opener, url, b[0], b[1]), jobs))
    wall = time.perf_counter() - t0
    if sampler:
        sampler.stop()
    ok = [r for r in res if r["ok"]]
    rts = [r["rt_ms"] for r in ok]
    procs = [r["proc_ms"] for r in ok if isinstance(r.get("proc_ms"), (int, float))]
    return {
        "concurrency": conc, "requests": n, "ok": len(ok), "errors": dict(Counter(r["error"] for r in res if not r["ok"])),
        "wall_s": wall, "throughput_rps": len(ok) / wall if wall > 0 else NAN,
        "rt_median_ms": med(rts), "rt_p95_ms": pct(rts, 95),
        "proc_median_ms": med(procs), "proc_p95_ms": pct(procs, 95), "proc_n": len(procs),
        "rss_mean_mb": statistics.mean(sampler.rss) if sampler and sampler.rss else NAN,
        "rss_peak_mb": max(sampler.rss) if sampler and sampler.rss else NAN,
        "cpu_mean_pct": statistics.mean(sampler.cpu) if sampler and sampler.cpu else NAN,
        "predictions": dict(Counter(r.get("pred") for r in ok)),
    }


def join_and(items):
    items = [str(i) for i in items]
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def fmt(x, nd=0):
    return "n/a" if x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Ukur latency, throughput, dan memori server grading cumi.")
    ap.add_argument("--images", required=True, help="folder gambar uji (boleh berisi subfolder kelas), mis. folder test")
    ap.add_argument("--url", action="append", required=True,
                    help="alamat server, boleh diulang: lokal (http://127.0.0.1:8000) dan ngrok (https://....ngrok-free.app)")
    ap.add_argument("--n", type=int, default=60, help="jumlah request per tingkat concurrency (default 60)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 5, 10], help="tingkat concurrency (default 1 5 10)")
    ap.add_argument("--port", type=int, default=8000, help="port lokal server, untuk mencari prosesnya (default 8000)")
    ap.add_argument("--pid", type=int, default=None, help="PID proses server, kalau tidak terdeteksi otomatis")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="hasil_ukur_deploy.json")
    args = ap.parse_args()

    # 1. Gambar uji
    files = sorted(p for p in Path(args.images).rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"Tidak ada gambar .jpg/.jpeg/.png di {args.images}")
    random.Random(args.seed).shuffle(files)
    files = files[:max(args.n, 1)]
    bodies = [make_body(p) for p in files]
    size_kb = statistics.mean(p.stat().st_size for p in files) / 1024

    # 2. Spesifikasi laptop
    hw = {"cpu": cpu_name(), "logical_cores": os.cpu_count(),
          "physical_cores": psutil.cpu_count(logical=False) if psutil else None,
          "ram_gb": ram_gb(), "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}
    print("=== Laptop server ===")
    print(f"CPU   : {hw['cpu']} ({hw['physical_cores'] or '?'} core fisik, {hw['logical_cores']} thread)")
    print(f"RAM   : {fmt(hw['ram_gb'], 1)} GB")
    print(f"OS    : {hw['os']}, Python {hw['python']}")
    print(f"Gambar: {len(files)} file dari {args.images}, rata-rata {size_kb:.0f} KB")

    proc = find_server(args.port, args.pid)
    if psutil is None:
        print("psutil belum terpasang: memori dan CPU server tidak diukur (pip install psutil).")
    elif proc is None:
        print(f"Proses server di port {args.port} tidak ditemukan; memori tidak diukur. Coba --pid <PID>.")
    rss_idle = rss_mb(proc) if proc is not None else NAN
    if proc is not None:
        print(f"Proses server: PID {proc.pid}, memori saat diam {rss_idle:.0f} MB")

    results = {"hardware": hw, "images": {"folder": args.images, "n": len(files), "mean_kb": size_kb},
               "server_pid": proc.pid if proc is not None else None, "rss_idle_mb": rss_idle, "urls": {}}
    health = None

    # 3. Ukur tiap URL
    for base in args.url:
        base = base.rstrip("/")
        opener = opener_for(base)
        print(f"\n=== {base} ===")
        try:
            h = get_json(opener, base + "/health")
            health = health or h
            print(f"/health: model dimuat = {h.get('model_loaded')}, bobot = {h.get('weights_file')}, "
                  f"threshold = {h.get('threshold')}, TensorFlow {h.get('tensorflow')}, perangkat = {h.get('device')}")
            if not h.get("model_loaded"):
                raise SystemExit("Model belum dimuat di server (lihat log server). Pengukuran dihentikan.")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("PERINGATAN: /health tidak ada; server masih memakai main.py lama. "
                      "Waktu proses server tidak bisa diukur.")
            else:
                print(f"PERINGATAN: /health gagal (HTTP {e.code}).")
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"Server tidak bisa dihubungi di {base}: {e}")

        url = base + "/predict/"
        for b in bodies[:3]:   # pemanasan
            post_once(opener, url, b[0], b[1])
        levels = {}
        print(f"{'conc':>4} {'ok':>4} {'req/s':>7} {'RT med':>8} {'RT p95':>8} {'proses med':>11} "
              f"{'proses p95':>11} {'RAM puncak':>11} {'CPU':>6}")
        for c in args.levels:
            r = run_level(opener, url, bodies, args.n, c, proc if is_local(base) else None)
            levels[c] = r
            err = f"  error: {r['errors']}" if r["errors"] else ""
            print(f"{c:>4} {r['ok']:>4} {fmt(r['throughput_rps'], 2):>7} {fmt(r['rt_median_ms']):>8} "
                  f"{fmt(r['rt_p95_ms']):>8} {fmt(r['proc_median_ms']):>11} {fmt(r['proc_p95_ms']):>11} "
                  f"{fmt(r['rss_peak_mb']):>8} MB {fmt(r['cpu_mean_pct']):>5}%{err}")
        results["urls"][base] = {"local": is_local(base), "health": health, "levels": levels}

    results["health"] = health
    Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # 4. Ringkasan untuk paper
    urls = results["urls"]
    local = next((u for u, v in urls.items() if v["local"]), None)
    remote = next((u for u, v in urls.items() if not v["local"]), None)
    cap = urls[local or remote]["levels"]           # kapasitas server: diukur lewat alamat lokal
    lat = urls[remote or local]["levels"]           # round trip: lewat tunnel kalau ada
    first = min(args.levels)
    proc_ref = next((v["levels"][first] for v in urls.values() if v["levels"][first]["proc_n"]), cap[first])
    device = (health or {}).get("device", "CPU") or "CPU"
    dev_txt = "the CPU" if device.upper().startswith("CPU") else f"the GPU ({device[4:] or 'GPU'})"
    peak = max((v["rss_peak_mb"] for v in cap.values() if v["rss_peak_mb"] == v["rss_peak_mb"]), default=NAN)
    tps = [cap[c]["throughput_rps"] for c in args.levels]
    where = "through the tunnel" if remote else "from a client on the same machine"

    print("\n=== Ringkasan ===")
    print(f"Waktu proses server : median {fmt(proc_ref['proc_median_ms'])} ms, p95 {fmt(proc_ref['proc_p95_ms'])} ms")
    print(f"Round trip ({'ngrok' if remote else 'lokal'})  : median {fmt(lat[first]['rt_median_ms'])} ms, "
          f"p95 {fmt(lat[first]['rt_p95_ms'])} ms")
    print("Throughput          : " + ", ".join(f"{c} bersamaan = {fmt(cap[c]['throughput_rps'], 2)} req/s"
                                               for c in args.levels))
    print(f"Memori server       : {fmt(rss_idle)} MB saat diam, puncak {fmt(peak)} MB")
    flat = len(tps) > 1 and tps[-1] < 1.2 * tps[0]
    if flat:
        print("Catatan: throughput hampir tidak naik saat request bersamaan. Ini wajar untuk main.py ini, karena "
              "model.predict dipanggil langsung di handler async sehingga request diproses satu per satu.")

    sent = (f"On the serving laptop ({hw['cpu']}, {fmt(hw['ram_gb'])} GB RAM; inference on {dev_txt}), the median "
            f"server processing time per request was {fmt(proc_ref['proc_median_ms'])} ms (95th percentile "
            f"{fmt(proc_ref['proc_p95_ms'])} ms), the median round-trip time {where} was "
            f"{fmt(lat[first]['rt_median_ms'])} ms ({fmt(lat[first]['rt_p95_ms'])} ms), and the service handled "
            f"{join_and(fmt(t, 1) for t in tps)} requests per second with {join_and(args.levels)} concurrent requests, "
            f"using about {fmt(rss_idle)} MB of memory.")
    if flat:
        sent += " Requests were processed one at a time because inference blocks the request handler."
    print("\nKalimat untuk paper (Section IV-E), periksa dulu sebelum ditempel:\n")
    print(sent)
    print(f"\nHasil lengkap: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
