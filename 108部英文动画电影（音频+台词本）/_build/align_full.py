# -*- coding: utf-8 -*-
"""Full pipeline: load word-level ASR, align to scripts, build timeline and HTML.
Writes all progress to a UTF-8 log file to avoid console encoding issues.
"""
import json, re, os, sys, time, urllib.parse, shutil
from pathlib import Path

LOG = Path(__file__).with_name("align_full.log")
ROOT = Path("D:/OneDrive - IPowerBI/视频/外刊课/108部英文动画电影（音频+台词本）")
HERE = ROOT / "_build"

SEG_PATH = HERE / "segments_word.json"
LINES_PATH = ROOT / "疯狂动物城_app" / "lines.json"
TIMELINE_OUT = HERE / "timeline_precise.json"
TEMPLATE_PATH = ROOT / "疯狂动物城_app" / "template.html"
HTML_OUT = ROOT / "疯狂动物城.html"
AUDIO_REL = "动画音频/1.疯狂动物城.mp3"


def log(msg: str):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return [t for t in text.split() if t]


def load_data():
    segs = json.loads(SEG_PATH.read_text(encoding="utf-8"))
    lines = json.loads(LINES_PATH.read_text(encoding="utf-8"))
    return segs, lines


def build_word_seq(segs, lines):
    aw = []
    for si, seg in enumerate(segs):
        for w in seg.get("words", []):
            wt = re.sub(r"[^a-z0-9']", "", w["word"].strip().lower())
            if wt:
                aw.append({"word": wt, "start": w["start"], "end": w["end"], "seg_idx": si})
    sw, si = [], []
    for li, line in enumerate(lines):
        for t in tokenize(line.get("en", "")):
            sw.append(t)
            si.append(li)
    return aw, sw, si


def find_anchors(aw, sw, ng=3):
    idx = {}
    for i in range(len(sw) - ng + 1):
        k = tuple(sw[i + j] for j in range(ng))
        idx.setdefault(k, []).append(i)
    used_a, used_s = set(), set()
    anchors = []
    for i in range(len(aw) - ng + 1):
        k = tuple(aw[i + j]["word"] for j in range(ng))
        if k not in idx:
            continue
        for sj in idx[k]:
            if any((i + j) in used_a or (sj + j) in used_s for j in range(ng)):
                continue
            for j in range(ng):
                anchors.append((i + j, sj + j))
                used_a.add(i + j)
                used_s.add(sj + j)
            break
    anchors.sort()
    clean, la, ls = [], -1, -1
    for a, s in anchors:
        if a > la and s > ls:
            clean.append((a, s))
            la, ls = a, s
    return clean


def local_nw(ab, sb, md=80):
    n, m, gp = len(ab), len(sb), -0.3
    if n == 0 or m == 0:
        return []
    if n > md or m > md:
        return greedy(ab, sb)
    INF = -1e9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gp
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gp
    bw = min(md // 2, max(n, m) // 3 + 5)
    for i in range(1, n + 1):
        js, je = max(1, i - bw), min(m, i + bw)
        for j in range(js, je + 1):
            sc = 2.0 if ab[i-1]["word"] == sb[j-1] else -1.0
            dp[i][j] = max(dp[i-1][j-1] + sc, dp[i-1][j] + gp, dp[i][j-1] + gp)
    i, j, ps = n, m, []
    while i > 0 and j > 0:
        sc = 2.0 if ab[i-1]["word"] == sb[j-1] else -1.0
        if dp[i][j] == dp[i-1][j-1] + sc:
            if ab[i-1]["word"] == sb[j-1]:
                ps.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + gp:
            i -= 1
        else:
            j -= 1
    ps.reverse()
    return ps


def greedy(ab, sb):
    ps, si = [], 0
    for ai, aw in enumerate(ab):
        if si >= len(sb):
            break
        for sj in range(si, min(si + 8, len(sb))):
            if sb[sj] == aw["word"]:
                ps.append((ai, sj))
                si = sj + 1
                break
    return ps


def fill_gaps(aw, sw, si, anchors, nl):
    N, M = len(aw), len(sw)
    ext = [(-1, -1)] + anchors + [(N, M)]
    all_p = list(anchors)
    for k in range(len(ext) - 1):
        al, sl = ext[k]
        ah, sh = ext[k+1]
        ab, sb = aw[al+1:ah], sw[sl+1:sh]
        if not ab or not sb:
            continue
        for ai, s2 in local_nw(ab, sb):
            all_p.append((al + 1 + ai, sl + 1 + s2))
    all_p.sort(key=lambda x: x[0])
    lt = {i: [] for i in range(nl)}
    for ai, s2 in all_p:
        li = si[s2]
        if li < nl:
            lt[li].append(aw[ai]["start"])
    out = [None] * nl
    for li, times in lt.items():
        if times:
            times.sort()
            out[li] = times[max(0, len(times)//4 - 1)]
    last = -1
    for li in range(nl):
        if out[li] is not None:
            if last >= 0 and li - last > 1:
                t0, t1 = out[last], out[li]
                for kk in range(last + 1, li):
                    out[kk] = round(t0 + (t1 - t0) * (kk - last) / (li - last), 3)
            last = li
    if out[0] is None:
        out[0] = 0.0
    if out[-1] is None and last >= 0:
        for li in range(last + 1, nl):
            out[li] = out[last] + (li - last) * 0.5
    for li in range(1, nl):
        if out[li] and out[li-1] and out[li] < out[li-1]:
            out[li] = out[li-1] + 0.05
    return out, all_p


def main():
    log("=== Zootopia Precise Alignment ===")
    log(f"Starting at {time.strftime('%H:%M:%S')}")

    segs, lines = load_data()
    log(f"Segments: {len(segs)}  Lines: {len(lines)}")

    aw, sw, si = build_word_seq(segs, lines)
    log(f"ASR words: {len(aw)}  Script words: {len(sw)}")
    log(f"ASR range: {aw[0]['start']:.1f}s - {aw[-1]['end']:.1f}s")

    # Phase 1: 3-gram
    a3 = find_anchors(aw, sw, 3)
    ml = set(si[s] for _, s in a3)
    log(f"3-gram: {len(a3)} pairs, {len(ml)}/{len(lines)} lines ({len(ml)*100/len(lines):.1f}%)")

    # Phase 1b: 2-gram extra
    a2 = set(find_anchors(aw, sw, 2))
    a3s = set(a3)
    extra = []
    for p in a2 - a3s:
        a, s = p
        if any(abs(a - pa) < 3 and abs(s - ps) < 3 for pa, ps in a3):
            continue
        extra.append(p)
    log(f"Extra 2-gram: {len(extra)}")

    merged = list(set(a3) | set(extra))
    merged.sort()
    clean, la, ls = [], -1, -1
    for a, s in merged:
        if a > la and s > ls:
            clean.append((a, s))
            la, ls = a, s
    log(f"Total anchors: {len(clean)}")

    # Phase 2: fill
    out_t, all_p = fill_gaps(aw, sw, si, clean, len(lines))
    anc = sum(1 for t in out_t if t is not None)
    log(f"After fill: {anc}/{len(lines)} ({anc*100/len(lines):.1f}%)")

    # Build timeline
    tl = []
    for i, line in enumerate(lines):
        tl.append({
            "idx": line["idx"], "zh": line["zh"], "en": line["en"],
            "t": round(out_t[i], 3) if out_t[i] is not None else None,
        })
    TIMELINE_OUT.write_text(
        json.dumps(tl, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8"
    )
    log(f"timeline saved: {TIMELINE_OUT}")
    log(f"Lines: {len(tl)}  Range: {tl[0]['t']}s - {tl[-1]['t']}s")

    log("--- First 8 ---")
    for x in tl[:8]:
        en = x["en"][:55] if x["en"] else "(empty)"
        log(f"  [{x['idx']:4d}] t={x['t']:7.2f}s | {en}")

    # Copy to app dir
    shutil.copy(TIMELINE_OUT, ROOT / "疯狂动物城_app" / "timeline.json")
    log("timeline copied to app dir")

    # Build HTML
    tmpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    audio_url = urllib.parse.quote(AUDIO_REL, safe="/.")
    html = tmpl.replace("__AUDIO_SRC__", audio_url)
    html = html.replace("__TIMELINE__", json.dumps(tl, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__MOVIE_TITLE__", "疯狂动物城 Zootopia")
    HTML_OUT.write_text(html, encoding="utf-8")
    log(f"HTML saved: {HTML_OUT}  ({HTML_OUT.stat().st_size} bytes)")

    log("=== DONE ===")


if __name__ == "__main__":
    main()
