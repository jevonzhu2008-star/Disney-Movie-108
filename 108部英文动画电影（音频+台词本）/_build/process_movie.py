# -*- coding: utf-8 -*-
"""
Generic single-movie pipeline: transcribe → parse → align → build_html.
Usage:
  python process_movie.py <id>         # one movie by audio id from manifest
  python process_movie.py --all        # batch all
  python process_movie.py --range 1-10 # range of audio ids
  python process_movie.py --resume     # resume from manifest; skip done

Output in ROOT/movies/<id>_<title>/
"""
import os, sys, json, re, time, urllib.parse
from pathlib import Path

# --- CUDA DLL discovery ---
def _add_cuda_dlls():
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
            p = Path(sp) / sub.replace("/", os.sep)
            if p.is_dir():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
_add_cuda_dlls()

# 国内加速 HF 模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

HERE = Path(__file__).parent
ROOT = HERE.parent
MANIFEST = HERE / "manifest.json"
TEMPLATE = ROOT / "疯狂动物城_app" / "template.html"
MOVIES_DIR = ROOT / "movies"
MOVIES_DIR.mkdir(exist_ok=True)
ALIGN_SCRIPT = HERE / "align2.py"

_MODEL = None
def get_model(size="medium"):
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        print(f"[model] loading whisper {size} on cuda...", flush=True)
        _MODEL = WhisperModel(size, device="cuda", compute_type="float16")
    return _MODEL


def safe_slug(s):
    return re.sub(r"[\\/:*?\"<>|]", "_", s)[:60]


def transcribe(audio_path, out_path, model_size="medium"):
    if out_path.exists() and out_path.stat().st_size > 100:
        try:
            json.loads(out_path.read_text(encoding="utf-8"))
            print(f"  [skip transcribe] {out_path.name} exists", flush=True)
            return None
        except:
            pass
    model = get_model(model_size)
    print(f"  [transcribe] {Path(audio_path).name}", flush=True)
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path), language="en", beam_size=5,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
    )
    out = []
    last_print = time.time()
    for seg in segments:
        out.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()})
        if time.time() - last_print > 15:
            pct = seg.end / info.duration * 100
            print(f"    ... {seg.end:7.1f}s / {info.duration:.1f}s ({pct:5.1f}%) segs={len(out)}", flush=True)
            last_print = time.time()
    elapsed = time.time() - t0
    print(f"  [done] segments={len(out)} elapsed={elapsed:.0f}s", flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {"duration": round(info.duration, 1), "elapsed": round(elapsed, 1), "segments": len(out)}


def parse_script(script_path, out_path):
    out = []
    with open(script_path, encoding="utf-8-sig") as f:
        idx = 0
        for raw in f:
            raw = raw.rstrip("\r\n")
            if not raw.strip(): continue
            body = raw.strip(); idx += 1
            if idx == 1:
                out.append({"idx": idx, "zh": body, "en": ""}); continue
            parts = re.split(r" {2,}", body, maxsplit=1)
            if len(parts) == 2:
                zh, en = parts[0].strip(), parts[1].strip()
            else:
                asc = sum(1 for c in body if ord(c) < 128)
                zh, en = (body, "") if asc / max(1, len(body)) <= 0.7 else ("", body)
            out.append({"idx": idx, "zh": zh, "en": en})
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return {"lines": len(out)}


def build_html(timeline_path, audio_rel, title, out_path):
    tmpl = open(TEMPLATE, encoding="utf-8").read()
    timeline = json.loads(open(timeline_path, encoding="utf-8").read())
    audio_url = urllib.parse.quote(audio_rel, safe="/.")
    html = tmpl.replace("__AUDIO_SRC__", audio_url)
    html = html.replace("__TIMELINE__", json.dumps(timeline, ensure_ascii=False, separators=(",", ":")))
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html)
    open(out_path, "w", encoding="utf-8").write(html)
    return {"html_size": out_path.stat().st_size}


def process_one(entry, model_size="medium", force=False):
    aid = entry["id"]
    title = entry["title"]
    audio_path = Path(entry["audio"])
    script_path = Path(entry["script"])

    work_name = f"{aid:03d}_{safe_slug(title)}"
    work_dir = MOVIES_DIR / work_name
    work_dir.mkdir(exist_ok=True)

    seg_path = work_dir / "segments.json"
    lines_path = work_dir / "lines.json"
    timeline_path = work_dir / "timeline.json"
    slug = safe_slug(title)
    html_path = work_dir / f"{slug}.html"
    meta_path = work_dir / "meta.json"

    if not force and html_path.exists() and meta_path.exists():
        print(f"[skip] {title}", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"\n==== [{aid}] {title} ====", flush=True)
    meta = {"id": aid, "title": title}

    t0 = time.time()
    tinfo = transcribe(str(audio_path), seg_path, model_size)
    if tinfo: meta.update(tinfo)

    pinfo = parse_script(str(script_path), lines_path)
    meta.update(pinfo)

    # align by calling align2.py
    print(f"  [align] ...", flush=True)
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ALIGN_SCRIPT), str(seg_path), str(lines_path), str(timeline_path)],
        capture_output=True, text=True, timeout=600)
    for line in r.stdout.split("\n"): print(f"    {line.strip()}" if line.strip() else "", flush=True)

    # read align stats from output
    align_info = {}
    for line in r.stdout.split("\n"):
        for k in ["segments", "lines", "phase1_anchors", "phase2_refined", "anchored", "anchor_rate"]:
            if f"  {k}:" in line:
                v = line.split(":")[-1].strip()
                try: align_info[k] = float(v)
                except: align_info[k] = v
    meta.update(align_info)

    # relative audio path for HTML
    audio_rel = os.path.relpath(str(audio_path), str(work_dir))
    hinfo = build_html(timeline_path, audio_rel, title, html_path)
    meta.update(hinfo)

    meta["total_elapsed"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"  [done] total={meta['total_elapsed']:.1f}s anchors={meta.get('anchor_rate',0)*100:.1f}%", flush=True)
    return meta


def main():
    manifest = json.loads(open(MANIFEST, encoding="utf-8").read())
    args = sys.argv[1:]

    if not args:
        print("Usage: python process_movie.py <id> | --all | --range 1-10 | --resume", flush=True)
        return

    if "--all" in args:
        entries = [e for e in manifest if e["audio"] and e["script"] and e["match"] != "no_match"]
    elif "--range" in args:
        i = args.index("--range")
        r = args[i+1].split("-")
        lo, hi = int(r[0]), int(r[1])
        entries = [e for e in manifest if e["id"] and lo <= e["id"] <= hi and e["audio"] and e["script"]]
    elif "--resume" in args:
        entries = [e for e in manifest if e["audio"] and e["script"]]
        # filter by done
        entries = [e for e in entries if not (MOVIES_DIR / f"{e['id']:03d}_{safe_slug(e['title'])}" / "meta.json").exists()]
    else:
        ids = [int(a) for a in args if a.isdigit()]
        entries = [e for e in manifest if e["id"] in ids]

    print(f"\nwill process {len(entries)} movies\n", flush=True)
    results = []
    for i, entry in enumerate(entries):
        try:
            m = process_one(entry, model_size="medium")
            results.append(m)
            good = m.get("anchor_rate", 0) * 100 if m else 0
            print(f"[{i+1}/{len(entries)}] {entry['title']}: anchors={good:.1f}%", flush=True)
        except Exception as ex:
            import traceback
            print(f"[FAIL] {entry['title']}: {ex}", flush=True)
            traceback.print_exc()

    print(f"\n=== Done {len(results)}/{len(entries)} ===")
    for r in results:
        print(f"  id={r['id']:3} {r['title'][:20]:20} "
              f"anchor_rate={r.get('anchor_rate',0)*100:.0f}% "
              f"elapsed={r.get('total_elapsed',0):.0f}s",
              flush=True)


if __name__ == "__main__":
    main()