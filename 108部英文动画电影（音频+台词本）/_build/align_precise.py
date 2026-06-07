# -*- coding: utf-8 -*-
"""
精确对齐：Whisper word-level timestamps → 台词行
使用全局序列对齐（DP）将 ASR 单词序列匹配到剧本单词序列，
再把单词的时间戳聚合到台词行级别。
"""
import json, re, sys
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """将文本拆分为小写单词列表（不含标点）"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [t for t in text.split() if len(t) >= 1]


def build_script_word_seq(lines: list) -> list[dict]:
    """将整个剧本展开为单词序列，每个单词记录所属的行索引"""
    words = []
    for li, line in enumerate(lines):
        en = line.get("en", "").strip()
        if not en:
            words.append({"li": li, "word": "", "is_empty": True})
            continue
        toks = tokenize(en)
        for w in toks:
            words.append({"li": li, "word": w, "is_empty": False})
    return words


def build_asr_word_seq(segments: list) -> list[dict]:
    """将 ASR segments 展开为单词序列，每个单词有其时间戳"""
    words = []
    for seg in segments:
        text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        # 如果有 word-level timestamps，直接使用
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_text = w.get("word", "").strip().lower()
                word_text = re.sub(r"[^a-z0-9']", "", word_text)
                if word_text:
                    words.append({
                        "word": word_text,
                        "start": w.get("start", seg_start),
                        "end": w.get("end", seg_end),
                    })
        else:
            # 没有 word-level timestamps 则平均分配时间
            toks = tokenize(text)
            if not toks:
                continue
            dur = seg_end - seg_start
            for i, tok in enumerate(toks):
                t_start = seg_start + dur * i / len(toks)
                t_end = seg_start + dur * (i + 1) / len(toks)
                words.append({"word": tok, "start": t_start, "end": t_end})
    return words


def align_sequences(asr_words: list[dict], script_words: list[dict],
                    gap_open=-1.0, gap_extend=-0.5, match_bonus=2.0, mismatch_penalty=-1.0):
    """
    全局 Needleman-Wunsch 序列对齐。
    asr_words: [{"word": str, "start": float, "end": float}, ...]
    script_words: [{"li": int, "word": str, ...}, ...]

    返回 aligned_pairs: [(asr_idx, script_idx), ...] （双方索引）
    """
    N = len(asr_words)
    M = len(script_words)

    if N == 0 or M == 0:
        return []

    # 使用带状 DP 优化（限制偏移范围）
    MAX_SHIFT = max(N, M)  # 保守起见，全范围
    band = min(MAX_SHIFT, max(N, M) // 3 + 10)

    # 如果序列太长，分块处理
    MAX_SEQ = 8000
    if max(N, M) > MAX_SEQ:
        return align_sequences_chunked(asr_words, script_words, gap_open, gap_extend, match_bonus, mismatch_penalty)

    # DP 数组（带状）
    INF_NEG = -1e9
    # 实际使用全矩阵（简化），对 疯狂动物城 ~14000 ASR词 vs ~12000 剧本词 约 1.68 亿格
    # 太大了。需要用分块策略。
    # 改用 Hirschberg 或分块对齐

    return align_sequences_chunked(asr_words, script_words, gap_open, gap_extend, match_bonus, mismatch_penalty)


def align_sequences_chunked(asr_words, script_words, gap_open=-0.5, gap_extend=-0.2,
                            match_bonus=2.0, mismatch_penalty=-1.0):
    """
    分块 Needleman-Wunsch 对齐。
    先用粗粒度锚点分块，再逐块细对齐。
    """
    # 创建快速查找表：单词→在剧本中的位置列表
    word_positions: dict[str, list[int]] = {}
    for si, sw in enumerate(script_words):
        w = sw["word"]
        if w:
            word_positions.setdefault(w, []).append(si)

    # === 阶段 1: 用精确匹配建立锚点 ===
    anchors = []  # [(asr_idx, script_idx)]

    # 限制：只取每个剧本位置最多被一个 ASR 词匹配
    used_script = set()
    used_asr = set()

    # 对每个 ASR 词，找它在剧本中的位置
    # 要求连续 2-3 个词同时匹配才建立锚点（更可靠）
    min_ngram = 2

    # 构建 ngram 索引
    asr_ngrams = {}
    for i in range(len(asr_words) - min_ngram + 1):
        ngram = tuple(asr_words[i + k]["word"] for k in range(min_ngram))
        asr_ngrams.setdefault(ngram, []).append(i)

    # 寻找匹配的 ngram
    for i in range(len(script_words) - min_ngram + 1):
        ngram = tuple(script_words[i + k]["word"] for k in range(min_ngram))
        if ngram in asr_ngrams:
            for ai in asr_ngrams[ngram]:
                if any(idx in used_asr for idx in range(ai, ai + min_ngram)):
                    continue
                if any(idx in used_script for idx in range(i, i + min_ngram)):
                    continue
                for k in range(min_ngram):
                    anchors.append((ai + k, i + k))
                    used_asr.add(ai + k)
                    used_script.add(i + k)
                break  # 每个剧本 ngram 只匹配一个 ASR ngram

    anchors.sort(key=lambda x: (x[0], x[1]))

    # 去掉破坏单调性的锚点
    clean = []
    last_a, last_s = -1, -1
    for a, s in anchors:
        if a > last_a and s > last_s:
            clean.append((a, s))
            last_a, last_s = a, s
    anchors = clean

    # === 阶段 2: 在锚点间做局部 DP ===
    extended = [(-1, -1)] + anchors + [(len(asr_words), len(script_words))]
    aligned_pairs = list(anchors)

    for k in range(len(extended) - 1):
        a_lo, s_lo = extended[k]
        a_hi, s_hi = extended[k + 1]
        asr_block = asr_words[a_lo + 1:a_hi]
        scr_block = script_words[s_lo + 1:s_hi]

        if not asr_block or not scr_block:
            continue

        # 限制块大小
        if len(asr_block) > 200 or len(scr_block) > 200:
            # 块太大，用简单贪婪匹配
            pairs = greedy_align(asr_block, scr_block)
        else:
            pairs = nw_align(asr_block, scr_block, gap_open, gap_extend, match_bonus, mismatch_penalty)

        for ai, si in pairs:
            aligned_pairs.append((a_lo + 1 + ai, s_lo + 1 + si))

    aligned_pairs.sort(key=lambda x: (x[0], x[1]))
    return aligned_pairs


def nw_align(asr_block, scr_block, gap_open=-0.5, gap_extend=-0.2,
             match_bonus=2.0, mismatch_penalty=-1.0):
    """标准的 Needleman-Wunsch 对齐"""
    N = len(asr_block)
    M = len(scr_block)

    # 压缩：对长块用带状 DP
    band = min(N + M, 60)

    # 用线性空间版本降低内存
    INF_NEG = -1e9
    dp = [[INF_NEG] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(1, N + 1):
        dp[i][0] = dp[i - 1][0] + gap_extend if i > 1 else gap_open
    for j in range(1, M + 1):
        dp[0][j] = dp[0][j - 1] + gap_extend if j > 1 else gap_open

    for i in range(1, N + 1):
        a_word = asr_block[i - 1]["word"]
        # 带状约束
        j_start = max(1, i - band)
        j_end = min(M, i + band)
        for j in range(j_start, j_end + 1):
            s_word = scr_block[j - 1]["word"]
            match = match_bonus if a_word == s_word else mismatch_penalty
            diag = dp[i - 1][j - 1] + match if dp[i - 1][j - 1] > INF_NEG / 2 else INF_NEG
            up = dp[i - 1][j] + (gap_extend if dp[i - 1][j] != dp[i - 2][j] if i > 1 else True else gap_open)
            left = dp[i][j - 1] + (gap_extend if dp[i][j - 1] != dp[i][j - 2] if j > 1 else True else gap_open)
            dp[i][j] = max(diag, up, left)

    # 回溯
    pairs = []
    i, j = N, M
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            a_word = asr_block[i - 1]["word"]
            s_word = scr_block[j - 1]["word"]
            match = match_bonus if a_word == s_word else mismatch_penalty
            if dp[i][j] == dp[i - 1][j - 1] + match:
                if a_word == s_word:
                    pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + (gap_extend if dp[i - 1][j] != dp[i - 2][j] if i > 1 else True else gap_open):
                i -= 1
            else:
                j -= 1

    pairs.reverse()
    return pairs


def greedy_align(asr_block, scr_block):
    """贪婪且单调的对齐——块太大时的后备方案"""
    pairs = []
    si = 0
    for ai, a_word in enumerate(asr_block):
        a_w = a_word["word"]
        if si >= len(scr_block):
            break
        # 找最近的匹配
        best_dist = 999
        best_si = -1
        for sj in range(si, min(si + 5, len(scr_block))):
            if scr_block[sj]["word"] == a_w:
                d = sj - si
                if d < best_dist:
                    best_dist, best_si = d, sj
        if best_si >= 0:
            pairs.append((ai, best_si))
            si = best_si + 1
        else:
            # 不匹配则跳过 ASR 词
            pass
    return pairs


def timestamps_for_lines(aligned_pairs, asr_words, script_words, num_lines):
    """
    根据对齐结果计算每行的时间戳。
    对于每行，找到匹配到的 ASR 单词，取其 start 的中位数作为行时间。
    """
    line_starts = []
    for li in range(num_lines):
        starts = []
        for ai, si in aligned_pairs:
            if script_words[si]["li"] == li and not script_words[si].get("is_empty", False):
                starts.append(asr_words[ai]["start"])
        if starts:
            line_starts.append(sorted(starts)[len(starts) // 4])  # 取前 1/4 分位数（偏保守）
        else:
            line_starts.append(None)

    # 插值填补 None
    last_li = -1
    for li in range(num_lines):
        if line_starts[li] is not None:
            if li - last_li > 1:
                t0 = line_starts[last_li] if last_li >= 0 else 0.0
                t1 = line_starts[li]
                gap = li - last_li
                for kk in range(last_li + 1, li):
                    line_starts[kk] = round(t0 + (t1 - t0) * (kk - last_li) / gap, 3)
            last_li = li

    # 如果第一行没有时间
    if line_starts[0] is None:
        line_starts[0] = 0.0

    # 处理末尾
    if line_starts[-1] is None:
        for li in range(num_lines - 1, -1, -1):
            if line_starts[li] is not None:
                break
        if li < num_lines - 1:
            for kk in range(li + 1, num_lines):
                line_starts[kk] = line_starts[li] + (kk - li) * 0.5  # 每行 0.5s
        else:
            line_starts[-1] = line_starts[li] + 1.0

    # 单调性保证
    for li in range(1, num_lines):
        if line_starts[li] is not None and line_starts[li - 1] is not None:
            if line_starts[li] < line_starts[li - 1]:
                line_starts[li] = line_starts[li - 1]

    return line_starts


def build_timeline_from_alignment(asr_words, script_words, lines, aligned_pairs):
    """从对齐结果构建完整的 timeline"""
    M = len(lines)
    line_times = timestamps_for_lines(aligned_pairs, asr_words, script_words, M)

    out = []
    for i in range(M):
        out.append({
            "idx": lines[i]["idx"],
            "zh": lines[i]["zh"],
            "en": lines[i]["en"],
            "t": round(line_times[i], 3) if line_times[i] is not None else None,
        })
    return out


def align_precise(segments_path, lines_path, out_path, verbose=True):
    """主入口：精确对齐 ASR segments → 台词行"""
    SEG = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    LINES = json.loads(Path(lines_path).read_text(encoding="utf-8"))

    # 构建单词序列
    asr_words = build_asr_word_seq(SEG)
    script_words = build_script_word_seq(LINES)

    if verbose:
        print(f"  ASR words: {len(asr_words)}", flush=True)
        print(f"  Script words: {len(script_words)}", flush=True)
        print(f"  Lines: {len(LINES)}", flush=True)

    # 序列对齐
    aligned_pairs = align_sequences(asr_words, script_words)

    if verbose:
        matched = len(aligned_pairs)
        script_nonempty = sum(1 for sw in script_words if not sw.get("is_empty", False))
        print(f"  Matched pairs: {matched} ({matched * 100 / max(1, script_nonempty):.1f}% of script words)", flush=True)

    # 构建 timeline
    timeline = build_timeline_from_alignment(asr_words, script_words, LINES, aligned_pairs)

    Path(out_path).write_text(
        json.dumps(timeline, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8"
    )

    anchored = sum(1 for x in timeline if x["t"] is not None)
    return {
        "segments": len(SEG),
        "lines": len(LINES),
        "asr_words": len(asr_words),
        "script_words": len(script_words),
        "matched_pairs": len(aligned_pairs),
        "anchored": anchored,
        "anchor_rate": anchored / len(LINES) if LINES else 0,
    }


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        seg_path = sys.argv[1]
        lines_path = sys.argv[2]
        out_path = sys.argv[3]
    else:
        HERE = Path(__file__).parent
        ROOT = HERE.parent
        seg_path = ROOT / "疯狂动物城_app" / "segments.json"
        lines_path = ROOT / "疯狂动物城_app" / "lines.json"
        out_path = HERE / "timeline_precise.json"

    info = align_precise(seg_path, lines_path, out_path)
    print(f"\n=== Results ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"Output → {out_path}")
