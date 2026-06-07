# -*- coding: utf-8 -*-
"""
流程：重新转写（带 word-level timestamps）+ 精确对齐

用法：
  python align_zootopia.py
"""
import os, sys, json, re, time, urllib.parse, ctypes
from pathlib import Path

# --- CUDA DLL discovery (必须先于任何 import) ---
def _add_cuda_dlls():
    import site
    dll_dirs = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
            p = Path(sp) / sub.replace("/", os.sep)
            if p.is_dir():
                dll_dirs.append(str(p))
                os.add_dll_directory(str(p))
                # 也加入 PATH — ctypes 会从这里找
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    # 显式预加载关键 DLL（避免运行时找不到）
    for d in dll_dirs:
        for fname in os.listdir(d):
            if fname.lower().endswith('.dll') and ('cublas' in fname.lower() or 'cudnn' in fname.lower()):
                try:
                    path = os.path.join(d, fname)
                    ctypes.CDLL(path)
                except Exception:
                    pass
    return dll_dirs

_cuda_dirs = _add_cuda_dlls()
print(f"[CUDA DLLs] loaded from {_cuda_dirs}", flush=True)

# 不要设置 HF_ENDPOINT — 使用本地缓存加载
os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

HERE = Path(__file__).parent
ROOT = HERE.parent

AUDIO_PATH = ROOT / "动画音频" / "1.疯狂动物城.mp3"
SCRIPT_PATH = ROOT / "台词本" / "台词本TXT格式" / "004.疯狂动物城.txt"
# 使用已有 lines.json (1690行, 和旧的timeline对齐)
LINES_IN = ROOT / "疯狂动物城_app" / "lines.json"
SEGMENTS_OUT = HERE / "segments_word.json"
LINES_OUT = HERE / "lines.json"
TIMELINE_OUT = HERE / "timeline_precise.json"


def transcribe_word_level(audio_path, out_path):
    """
    使用 whisper 带 word-level timestamps 进行转写
    同时关闭 VAD 以避免漏开头
    """
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="cuda", compute_type="float16",
                         local_files_only=True)
    print(f"  [transcribe] {audio_path.name}", flush=True)

    t0 = time.time()
    # 使用更激进的 VAD 设置以捕获更多语音
    segments, info = model.transcribe(
        str(audio_path), language="en", beam_size=5,
        word_timestamps=True,
        vad_filter=False,  # 关闭 VAD 以保留开头舞台剧对话
    )
    out = []
    for seg in segments:
        words_data = []
        if seg.words:
            for w in seg.words:
                words_data.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 3) if w.probability else None,
                })
        out.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words_data,
        })
    elapsed = time.time() - t0
    print(f"  [done] segments={len(out)} words={sum(len(s['words']) for s in out)} elapsed={elapsed:.0f}s", flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # 验证输出
    if out:
        print(f"    First seg: start={out[0]['start']:.1f}s end={out[0]['end']:.1f}s | {out[0]['text'][:60]}", flush=True)
        print(f"    First seg words: {len(out[0].get('words', []))}", flush=True)
        if out[0].get('words'):
            print(f"    First word: {out[0]['words'][0]}", flush=True)

    return out


def parse_script(script_path):
    lines = []
    with open(script_path, encoding="utf-8-sig") as f:
        idx = 0
        for raw in f:
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                continue
            body = raw.strip()
            idx += 1
            if idx == 1:
                lines.append({"idx": idx, "zh": body, "en": ""})
                continue
            parts = re.split(r" {2,}", body, maxsplit=1)
            if len(parts) == 2:
                zh, en = parts[0].strip(), parts[1].strip()
            else:
                asc = sum(1 for c in body if ord(c) < 128)
                zh, en = (body, "") if asc / max(1, len(body)) <= 0.7 else ("", body)
            lines.append({"idx": idx, "zh": zh, "en": en})
    print(f"  [parse] {len(lines)} lines from {script_path.name}", flush=True)
    return lines


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [t for t in text.split() if len(t) >= 1]


def align_words_to_lines(segments, lines):
    """
    精确对齐：将 ASR word-level 时间戳匹配到每句剧本台词
    """

    class WordTok:
        __slots__ = ("word", "start", "end", "seg_idx")
        def __init__(self, word, start, end, seg_idx):
            self.word = word
            self.start = start
            self.end = end
            self.seg_idx = seg_idx

    # 构建 ASR 单词列表（带时间）
    asr_words: list[WordTok] = []
    for si, seg in enumerate(segments):
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                wt = w.get("word", "").strip()
                wt = re.sub(r"[^a-z0-9']", "", wt.lower())
                if wt:
                    asr_words.append(WordTok(
                        word=wt,
                        start=w.get("start", seg["start"]),
                        end=w.get("end", seg["end"]),
                        seg_idx=si,
                    ))
        else:
            toks = tokenize(seg.get("text", ""))
            dur = seg["end"] - seg["start"]
            for i, tok in enumerate(toks):
                asr_words.append(WordTok(
                    word=tok,
                    start=seg["start"] + dur * i / len(toks),
                    end=seg["start"] + dur * (i + 1) / len(toks),
                    seg_idx=si,
                ))

    # 构建台词单词列表
    script_words = []  # [(word, line_idx)]
    script_words_info = []  # 行号列表（从0开始）
    for li, line in enumerate(lines):
        en = line.get("en", "").strip()
        if not en:
            script_words_info.append({"li": li, "is_empty": True})
            continue
        toks = tokenize(en)
        for w in toks:
            script_words.append(w)
            script_words_info.append({"li": li, "is_empty": False})

    print(f"  ASR words: {len(asr_words)}, Script words: {len(script_words)}", flush=True)

    N = len(asr_words)
    M = len(script_words)

    # 构建反向索引：单词→剧本位置
    word_pos: dict[str, list[int]] = {}
    for si, w in enumerate(script_words):
        word_pos.setdefault(w, []).append(si)

    # === 阶段1: n-gram 锚点 ===
    # 使用 3-gram（连续3个词完全匹配来建立可靠锚点）
    NGRAM = 3

    # 构建剧本 3-gram 索引
    script_ngrams: dict[tuple, list[int]] = {}
    for i in range(M - NGRAM + 1):
        ngram = tuple(script_words[i + k] for k in range(NGRAM))
        script_ngrams.setdefault(ngram, []).append(i)

    # 匹配 3-gram
    used_script_pos = set()
    used_asr_pos = set()
    anchors = []  # [(asr_idx, script_idx)]
    # 同时记录分数
    for i in range(N - NGRAM + 1):
        ngram = tuple(asr_words[i + k].word for k in range(NGRAM))
        if ngram not in script_ngrams:
            continue
        for sj in script_ngrams[ngram]:
            # 检查是否与已使用的冲突
            conflict = False
            for k in range(NGRAM):
                if (i + k) in used_asr_pos or (sj + k) in used_script_pos:
                    conflict = True
                    break
            if conflict:
                continue
            # 计算覆盖度分数
            score = 0
            for k in range(NGRAM):
                # 加上单词概率或简单给分
                prob = asr_words[i + k].end - asr_words[i + k].start
                score += max(0, 2.0 - prob * 5)  # 短词高分（更可靠）
            for k in range(NGRAM):
                anchors.append((i + k, sj + k, score / NGRAM))
                used_asr_pos.add(i + k)
                used_script_pos.add(sj + k)
            break  # 每个ASR ngram 只匹配一次

    anchors.sort(key=lambda x: (x[0], x[1]))
    # 去掉重复和破坏单调性的
    clean_anchors = []
    last_a, last_s = -1, -1
    for a, s, _ in anchors:
        if a > last_a and s > last_s:
            clean_anchors.append((a, s))
            last_a, last_s = a, s
    anchors_pairs = [(a, s) for a, s in clean_anchors]

    print(f"  3-gram anchors: {len(anchors_pairs)} words matched", flush=True)

    # === 阶段2: 锚点间局部 Needleman-Wunsch ===
    extended = [(-1, -1)] + anchors_pairs + [(N, M)]

    def local_nw(asr_block, scr_block, max_dim=80):
        """局部 Needleman-Wunsch，带带状约束"""
        n = len(asr_block)
        m = len(scr_block)
        if n == 0 or m == 0:
            return []
        if n > max_dim or m > max_dim:
            return greedy_match(asr_block, scr_block)

        INF_NEG = -1e9
        dp = [[INF_NEG] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] - 0.3
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] - 0.3

        for i in range(1, n + 1):
            band = min(max_dim // 2, max(n, m) // 3 + 5)
            j_start = max(1, i - band)
            j_end = min(m, i + band)
            for j in range(j_start, j_end + 1):
                match = 2.0 if asr_block[i - 1].word == scr_block[j - 1] else -1.0
                diag = dp[i - 1][j - 1] + match
                up = dp[i - 1][j] - 0.3
                left = dp[i][j - 1] - 0.3
                dp[i][j] = max(diag, up, left)

        pairs = []
        i, j = n, m
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                match = 2.0 if asr_block[i - 1].word == scr_block[j - 1] else -1.0
                if dp[i][j] == dp[i - 1][j - 1] + match:
                    if asr_block[i - 1].word == scr_block[j - 1]:
                        pairs.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i - 1][j] - 0.3:
                    i -= 1
                else:
                    j -= 1
        pairs.reverse()
        return pairs

    def greedy_match(asr_block, scr_block):
        pairs = []
        si = 0
        for ai, aw in enumerate(asr_block):
            if si >= len(scr_block):
                break
            for sj in range(si, min(si + 8, len(scr_block))):
                if scr_block[sj] == aw.word:
                    pairs.append((ai, sj))
                    si = sj + 1
                    break
        return pairs

    all_pairs = list(anchors_pairs)

    for k in range(len(extended) - 1):
        a_lo, s_lo = extended[k]
        a_hi, s_hi = extended[k + 1]
        asr_block = asr_words[a_lo + 1:a_hi]
        scr_block = script_words[s_lo + 1:s_hi]

        if not asr_block or not scr_block:
            continue

        # 如果块太大，分层处理
        pairs = local_nw(asr_block, scr_block)

        for ai, si in pairs:
            all_pairs.append((a_lo + 1 + ai, s_lo + 1 + si))

    all_pairs.sort(key=lambda x: (x[0], x[1]))

    # === 聚合到行级时间戳 ===
    # 对每行剧本，取匹配到的 ASR 单词的 start 时间的第1/4分位作为开始时间
    line_starts = {}
    for ai, si in all_pairs:
        li = script_words_info[si]["li"]
        t = asr_words[ai].start
        if li not in line_starts:
            line_starts[li] = []
        line_starts[li].append(t)

    out_times = [None] * len(lines)
    for li, times in line_starts.items():
        if times:
            times.sort()
            # 取前1/4分位数作为该行的开始时间
            out_times[li] = times[max(0, len(times) // 4 - 1)]

    # 插值填补空行
    last_anchored = -1
    for li in range(len(lines)):
        if out_times[li] is not None:
            if li - last_anchored > 1 and last_anchored >= 0:
                t0 = out_times[last_anchored]
                t1 = out_times[li]
                for kk in range(last_anchored + 1, li):
                    out_times[kk] = round(t0 + (t1 - t0) * (kk - last_anchored) / (li - last_anchored), 3)
            last_anchored = li

    # 开头
    if out_times[0] is None:
        out_times[0] = 0.0
        for li in range(1, len(lines)):
            if out_times[li] is not None:
                break
    else:
        if out_times[0] > 0:
            out_times[0] = max(0.0, out_times[0] - 0.3)  # 稍微提前

    # 末尾
    if out_times[-1] is None and last_anchored >= 0:
        for li in range(last_anchored + 1, len(lines)):
            out_times[li] = out_times[last_anchored] + (li - last_anchored) * 0.5

    # 单调性
    for li in range(1, len(lines)):
        if out_times[li] is not None and out_times[li - 1] is not None:
            if out_times[li] < out_times[li - 1]:
                out_times[li] = out_times[li - 1] + 0.05

    anchored_count = sum(1 for t in out_times if t is not None)
    print(f"  Anchored lines: {anchored_count}/{len(lines)} ({anchored_count*100/len(lines):.1f}%)", flush=True)

    return out_times, all_pairs


def main():
    # 1. 读取已有剧本台词（1690行，包含标题和歌词）
    lines = json.loads(Path(LINES_IN).read_text(encoding="utf-8"))
    print(f"  [load] {len(lines)} lines from {LINES_IN.name}", flush=True)

    # 2. 重新转写（带 word-level timestamps）
    segments = transcribe_word_level(AUDIO_PATH, SEGMENTS_OUT)

    # 3. 精确对齐
    print(f"\n[align] precise word-level alignment...", flush=True)
    t0 = time.time()
    out_times, pairs = align_words_to_lines(segments, lines)
    elapsed = time.time() - t0
    print(f"  [done] alignment in {elapsed:.1f}s", flush=True)

    # 4. 构建 timeline
    timeline = []
    for i in range(len(lines)):
        timeline.append({
            "idx": lines[i]["idx"],
            "zh": lines[i]["zh"],
            "en": lines[i]["en"],
            "t": round(out_times[i], 3) if out_times[i] is not None else None,
        })

    # 5. 保存
    Path(TIMELINE_OUT).write_text(
        json.dumps(timeline, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8"
    )

    anchored = sum(1 for x in timeline if x["t"] is not None)
    print(f"\n=== Results ===")
    print(f"  Lines: {len(timeline)}")
    print(f"  Anchored: {anchored} ({anchored*100/len(timeline):.1f}%)")
    print(f"  Time range: {out_times[0]:.2f}s - {out_times[-1]:.2f}s")
    print(f"  Output → {TIMELINE_OUT}")

    # 验证：检查前几行
    print(f"\n--- 开头 5 行 ---")
    for x in timeline[:5]:
        print(f"  [{x['idx']:4d}] t={x['t']:7.2f}s | {x['en'][:50]}")

    # 验证 ASR 开头是否被正确识别
    print(f"\n--- ASR 前3段 ---")
    for i, seg in enumerate(segments[:3]):
        print(f"  [{i}] {seg['start']:.1f}s-{seg['end']:.1f}s | {seg['text'][:60]}")


if __name__ == "__main__":
    main()
