# -*- coding: utf-8 -*-
"""
精确对齐：word-level timestamps → 台词时间线
使用硬编码路径避免 PowerShell 编码问题
"""
import json, re, sys
from pathlib import Path

ROOT = Path("D:/OneDrive - IPowerBI/视频/外刊课/108部英文动画电影（音频+台词本）")
HERE = ROOT / "_build"

SEG_PATH = HERE / "segments_word.json"
LINES_PATH = ROOT / "疯狂动物城_app" / "lines.json"
OUT_PATH = HERE / "timeline_precise.json"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [t for t in text.split() if t]


def build_word_seq(segments, lines):
    asr_words = []
    for si, seg in enumerate(segments):
        for w in seg.get("words", []):
            wt = w["word"].strip().lower()
            wt = re.sub(r"[^a-z0-9']", "", wt)
            if wt:
                asr_words.append({
                    "word": wt, "start": w["start"],
                    "end": w["end"], "seg_idx": si,
                })
    script_words = []
    script_info = []
    for li, line in enumerate(lines):
        en = line.get("en", "").strip()
        if not en:
            continue
        for t in tokenize(en):
            script_words.append(t)
            script_info.append(li)
    return asr_words, script_words, script_info


def find_anchors(asr_words, script_words, ngram=3):
    script_ngrams = {}
    for i in range(len(script_words) - ngram + 1):
        ng = tuple(script_words[i + k] for k in range(ngram))
        script_ngrams.setdefault(ng, []).append(i)
    used_script, used_asr = set(), set()
    anchors = []
    for i in range(len(asr_words) - ngram + 1):
        ng = tuple(asr_words[i + k]["word"] for k in range(ngram))
        if ng not in script_ngrams:
            continue
        for sj in script_ngrams[ng]:
            if any((i+k) in used_asr or (sj+k) in used_script for k in range(ngram)):
                continue
            for k in range(ngram):
                anchors.append((i + k, sj + k))
                used_asr.add(i + k)
                used_script.add(sj + k)
            break
    anchors.sort()
    clean, last_a, last_s = [], -1, -1
    for a, s in anchors:
        if a > last_a and s > last_s:
            clean.append((a, s))
            last_a, last_s = a, s
    return clean


def local_nw(asr_block, scr_block, max_dim=80):
    n, m = len(asr_block), len(scr_block)
    if n == 0 or m == 0:
        return []
    if n > max_dim or m > max_dim:
        return greedy_match(asr_block, scr_block)
    INF_NEG, gap = -1e9, -0.3
    dp = [[INF_NEG] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap
    band = min(max_dim // 2, max(n, m) // 3 + 5)
    for i in range(1, n + 1):
        j_start, j_end = max(1, i - band), min(m, i + band)
        for j in range(j_start, j_end + 1):
            match = 2.0 if asr_block[i-1]["word"] == scr_block[j-1] else -1.0
            dp[i][j] = max(dp[i-1][j-1] + match, dp[i-1][j] + gap, dp[i][j-1] + gap)
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        match = 2.0 if asr_block[i-1]["word"] == scr_block[j-1] else -1.0
        if dp[i][j] == dp[i-1][j-1] + match:
            if asr_block[i-1]["word"] == scr_block[j-1]:
                pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + gap:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def greedy_match(asr_block, scr_block):
    pairs, si = [], 0
    for ai, aw in enumerate(asr_block):
        if si >= len(scr_block):
            break
        for sj in range(si, min(si + 8, len(scr_block))):
            if scr_block[sj] == aw["word"]:
                pairs.append((ai, sj))
                si = sj + 1
                break
    return pairs


def fill_gaps(asr_words, script_words, script_info, anchors, num_lines):
    N, M = len(asr_words), len(script_words)
    ext = [(-1,-1)] + anchors + [(N, M)]
    all_pairs = list(anchors)
    for k in range(len(ext) - 1):
        a_lo, s_lo = ext[k]
        a_hi, s_hi = ext[k+1]
        ab = asr_words[a_lo+1:a_hi]
        sb = script_words[s_lo+1:s_hi]
        if not ab or not sb:
            continue
        for ai, si in local_nw(ab, sb):
            all_pairs.append((a_lo+1+ai, s_lo+1+si))
    all_pairs.sort(key=lambda x: (x[0], x[1]))
    lt = {li: [] for li in range(num_lines)}
    for ai, si in all_pairs:
        li = script_info[si]
        if li < num_lines:
            lt[li].append(asr_words[ai]["start"])
    out = [None] * num_lines
    for li, times in lt.items():
        if times:
            times.sort()
            out[li] = times[max(0, len(times)//4 - 1)]
    last_ok = -1
    for li in range(num_lines):
        if out[li] is not None:
            if last_ok >= 0 and li - last_ok > 1:
                t0, t1 = out[last_ok], out[li]
                for kk in range(last_ok+1, li):
                    out[kk] = round(t0 + (t1 - t0) * (kk - last_ok) / (li - last_ok), 3)
            last_ok = li
    if out[0] is None:
        out[0] = 0.0
    if out[-1] is None and last_ok >= 0:
        for li in range(last_ok+1, num_lines):
            out[li] = out[last_ok] + (li - last_ok) * 0.5
    for li in range(1, num_lines):
        if out[li] and out[li-1] and out[li] < out[li-1]:
            out[li] = out[li-1] + 0.05
    return out, all_pairs


def main():
    print(f"Loading...", flush=True)
    segs = json.loads(SEG_PATH.read_text(encoding="utf-8"))
    lines = json.loads(LINES_PATH.read_text(encoding="utf-8"))
    print(f"  Segments: {len(segs)}  Lines: {len(lines)}", flush=True)

    asr_words, script_words, script_info = build_word_seq(segs, lines)
    na, ns = len(asr_words), len(script_words)
    print(f"  ASR words: {na}  Script words: {ns}", flush=True)
    print(f"  ASR range: {asr_words[0]['start']:.1f}s - {asr_words[-1]['end']:.1f}s", flush=True)

    # Phase 1: 3-gram
    a3 = find_anchors(asr_words, script_words, 3)
    ml = set()
    for _, s in a3:
        ml.add(script_info[s])
    print(f"  Phase1 (3-gram): {len(a3)} pairs, {len(ml)}/{len(lines)} lines ({len(ml)*100/len(lines):.1f}%)", flush=True)

    # Phase 1b: extra 2-gram
    a2 = set(find_anchors(asr_words, script_words, 2))
    a3s = set(a3)
    extra = []
    for p in a2 - a3s:
        a, s = p
        if any(abs(a-pa) < 3 and abs(s-ps) < 3 for pa, ps in a3):
            continue
        extra.append(p)
    print(f"  Extra 2-gram: {len(extra)}", flush=True)

    merged = list(set(a3) | set(extra))
    merged.sort()
    clean, la, ls = [], -1, -1
    for a, s in merged:
        if a > la and s > ls:
            clean.append((a, s))
            la, ls = a, s
    print(f"  Total anchors: {len(clean)}", flush=True)

    # Phase 2: gap fill
    out_times, all_pairs = fill_gaps(asr_words, script_words, script_info, clean, len(lines))
    anchored = sum(1 for t in out_times if t is not None)
    print(f"  After fill: {anchored}/{len(lines)} ({anchored*100/len(lines):.1f}%)", flush=True)

    # Build output
    timeline = []
    for i, line in enumerate(lines):
        timeline.append({
            "idx": line["idx"], "zh": line["zh"], "en": line["en"],
            "t": round(out_times[i], 3) if out_times[i] is not None else None,
        })

    OUT_PATH.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n  Written: {OUT_PATH}", flush=True)
    print(f"  Lines: {len(timeline)}  Range: {timeline[0]['t']}s - {timeline[-1]['t']}s", flush=True)

    print("\n  --- First 8 ---", flush=True)
    for x in timeline[:8]:
        print(f"  [{x['idx']:4d}] t={x['t']:7.2f}s | {x['en'][:55]}", flush=True)


if __name__ == "__main__":
    main()
