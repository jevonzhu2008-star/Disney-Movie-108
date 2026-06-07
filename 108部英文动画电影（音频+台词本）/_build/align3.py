# -*- coding: utf-8 -*-
"""Precise word-level alignment: no CJK chars in source code paths."""
import json, re, os, sys, shutil, urllib.parse
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

# Use os.scandir + raw bytes to find app dir without CJK chars in code
app_name = None
audio_name = None
for entry in os.scandir(str(ROOT)):
    n = entry.name
    if n.endswith("_app"):
        app_name = n
    elif n == "动画音频":
        for a in os.scandir(entry.path):
            if a.name.endswith(".mp3"):
                audio_name = a.name

if not app_name:
    print("ERROR: no _app dir found", flush=True); sys.exit(1)

APP_DIR = ROOT / app_name
LINES_PATH = APP_DIR / "lines.json"
TEMPLATE_PATH = APP_DIR / "template.html"
TIMELINE_APP = APP_DIR / "timeline.json"

SEG_PATH = HERE / "segments_word.json"
TIMELINE_OUT = HERE / "timeline_precise.json"
HTML_OUT = ROOT / (app_name.replace("_app", ".html"))
AUDIO_REL = f"动画音频/{audio_name}" if audio_name else f"动画音频/1.{app_name.replace('_app','')}.mp3"

print(f"App dir: {app_name}", flush=True)
print(f"Lines: {LINES_PATH} exists={LINES_PATH.exists()}", flush=True)
print(f"Segs:  {SEG_PATH} exists={SEG_PATH.exists()}", flush=True)
print(f"Audio: {AUDIO_REL}", flush=True)


def tokenize(text):
    return [t for t in re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split() if t]


def anchors_ngram(aw, sw, ng=3):
    idx, used_a, used_s, out = {}, set(), set(), []
    for i in range(len(sw) - ng + 1):
        k = tuple(sw[i + j] for j in range(ng))
        idx.setdefault(k, []).append(i)
    for i in range(len(aw) - ng + 1):
        k = tuple(aw[i + j] for j in range(ng))
        if k not in idx: continue
        for sj in idx[k]:
            if any((i+j) in used_a or (sj+j) in used_s for j in range(ng)): continue
            for j in range(ng):
                out.append((i+j, sj+j)); used_a.add(i+j); used_s.add(sj+j)
            break
    out.sort()
    clean, la, ls = [], -1, -1
    for a, s in out:
        if a > la and s > ls: clean.append((a,s)); la, ls = a, s
    return clean


def nw_align(ab, sb, md=80):
    n, m, gp = len(ab), len(sb), -0.3
    if n == 0 or m == 0: return []
    if n > md or m > md: return fast_match(ab, sb)
    INF = -1e9
    dp = [[INF]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(1, n+1): dp[i][0] = dp[i-1][0] + gp
    for j in range(1, m+1): dp[0][j] = dp[0][j-1] + gp
    bw = min(md//2, max(n,m)//3+5)
    for i in range(1, n+1):
        js, je = max(1, i-bw), min(m, i+bw)
        for j in range(js, je+1):
            sc = 2.0 if ab[i-1]["word"] == sb[j-1] else -1.0
            dp[i][j] = max(dp[i-1][j-1]+sc, dp[i-1][j]+gp, dp[i][j-1]+gp)
    i, j, ps = n, m, []
    while i > 0 and j > 0:
        sc = 2.0 if ab[i-1]["word"] == sb[j-1] else -1.0
        if dp[i][j] == dp[i-1][j-1]+sc:
            if ab[i-1]["word"] == sb[j-1]: ps.append((i-1,j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j]+gp: i -= 1
        else: j -= 1
    ps.reverse(); return ps


def fast_match(ab, sb):
    ps, si = [], 0
    for ai, aw in enumerate(ab):
        if si >= len(sb): break
        for sj in range(si, min(si+8, len(sb))):
            if sb[sj] == aw["word"]: ps.append((ai,sj)); si = sj+1; break
    return ps


def gap_fill(aw, sw, si_info, anchors, nl):
    N, M = len(aw), len(sw)
    ext = [(-1,-1)] + anchors + [(N,M)]
    all_p = list(anchors)
    for k in range(len(ext)-1):
        al, sl = ext[k]; ah, sh = ext[k+1]
        ab, sb = aw[al+1:ah], sw[sl+1:sh]
        if not ab or not sb: continue
        for ai, si in nw_align(ab, sb): all_p.append((al+1+ai, sl+1+si))
    all_p.sort(key=lambda x: x[0])
    lt = {i:[] for i in range(nl)}
    for ai, si in all_p:
        li = si_info[si]
        if li < nl: lt[li].append(aw[ai]["start"])
    out = [None]*nl
    for li, times in lt.items():
        if times: times.sort(); out[li] = times[max(0, len(times)//4 - 1)]
    last = -1
    for li in range(nl):
        if out[li] is not None:
            if last >= 0 and li-last > 1:
                t0, t1 = out[last], out[li]
                for kk in range(last+1, li):
                    out[kk] = round(t0 + (t1-t0)*(kk-last)/(li-last), 3)
            last = li
    if out[0] is None: out[0] = 0.0
    if out[-1] is None and last >= 0:
        for li in range(last+1, nl): out[li] = out[last] + (li-last)*0.5
    for li in range(1, nl):
        if out[li] and out[li-1] and out[li] < out[li-1]: out[li] = out[li-1] + 0.05
    return out, all_p


def write_log(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(l + "\n" for l in lines)


def main():
    lines_out = []

    segs = json.loads(SEG_PATH.read_text(encoding="utf-8"))
    lines = json.loads(LINES_PATH.read_text(encoding="utf-8"))
    lines_out.append(f"Segments: {len(segs)}  Lines: {len(lines)}")
    print(lines_out[-1], flush=True)

    aw, sw, si_info = [], [], []
    for si, seg in enumerate(segs):
        for w in seg.get("words", []):
            wt = re.sub(r"[^a-z0-9']", "", w["word"].strip().lower())
            if wt: aw.append({"word": wt, "start": w["start"], "end": w["end"], "seg_idx": si})
    for li, line in enumerate(lines):
        for t in tokenize(line.get("en", "")):
            sw.append(t); si_info.append(li)

    lines_out.append(f"ASR words: {len(aw)}  Script words: {len(sw)}")
    lines_out.append(f"ASR range: {aw[0]['start']:.1f}s - {aw[-1]['end']:.1f}s")
    print(lines_out[-2], lines_out[-1], sep="\n", flush=True)

    # Phase 1
    a3 = anchors_ngram(aw, sw, 3)
    ml = set(si_info[s] for _, s in a3)
    lines_out.append(f"3-gram: {len(a3)} pairs, {len(ml)}/{len(lines)} ({len(ml)*100/len(lines):.1f}%)")
    print(lines_out[-1], flush=True)

    a2 = set(anchors_ngram(aw, sw, 2))
    a3s = set(a3)
    extra = [p for p in (a2 - a3s) if not any(abs(p[0]-pa)<3 and abs(p[1]-ps)<3 for pa,ps in a3)]
    lines_out.append(f"Extra 2-gram: {len(extra)}")
    print(lines_out[-1], flush=True)

    merged = list(set(a3) | set(extra)); merged.sort()
    clean = []; la, ls = -1, -1
    for a, s in merged:
        if a > la and s > ls: clean.append((a,s)); la, ls = a, s
    lines_out.append(f"Total anchors: {len(clean)}")
    print(lines_out[-1], flush=True)

    # Phase 2
    out_t, _ = gap_fill(aw, sw, si_info, clean, len(lines))
    anc = sum(1 for t in out_t if t is not None)
    lines_out.append(f"After fill: {anc}/{len(lines)} ({anc*100/len(lines):.1f}%)")
    print(lines_out[-1], flush=True)

    tl = [{"idx": line["idx"], "zh": line["zh"], "en": line["en"],
           "t": round(out_t[i],3) if out_t[i] is not None else None}
          for i, line in enumerate(lines)]
    TIMELINE_OUT.write_text(json.dumps(tl, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    lines_out.append(f"timeline: {TIMELINE_OUT}")
    print(lines_out[-1], flush=True)

    # Copy to app
    shutil.copy(str(TIMELINE_OUT), str(TIMELINE_APP))
    print("Copied to app dir", flush=True)

    # Build HTML
    tmpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    audio_url = urllib.parse.quote(AUDIO_REL, safe="/.")
    html = tmpl.replace("__AUDIO_SRC__", audio_url)
    html = html.replace("__TIMELINE__", json.dumps(tl, ensure_ascii=False, separators=(",", ":")))
    title_text = app_name.replace("_app", "")
    html = html.replace("__MOVIE_TITLE__", title_text + " Zootopia")
    HTML_OUT.write_text(html, encoding="utf-8")
    lines_out.append(f"HTML: {HTML_OUT} ({HTML_OUT.stat().st_size} bytes)")
    print(lines_out[-1], flush=True)

    # Sample
    print("\nFirst 8:", flush=True)
    for x in tl[:8]:
        en = x["en"][:55] if x["en"] else "(empty)"
        print(f"  [{x['idx']:4d}] t={x['t']:7.2f}s | {en}", flush=True)

    write_log(HERE / "align3.log", lines_out)


if __name__ == "__main__":
    main()
