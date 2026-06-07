# -*- coding: utf-8 -*-
"""
对齐 whisper segments → 台词行
两阶段：
  阶段 1: LIS 全局锚点对齐（高置信度）
  阶段 2: 锚点之间未匹配段做局部 DP 细化（更松阈值，被锚点约束在小窗口内）
  阶段 3: 仍未匹配的行用线性插值
"""
import json
import re
import sys
from pathlib import Path


def tokens(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return set(t for t in s.split() if len(t) >= 2)


def align(segments_path, lines_path, out_path, verbose=True):
    SEG = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    LINES = json.loads(Path(lines_path).read_text(encoding="utf-8"))

    seg_tok = [tokens(s["text"]) for s in SEG]
    line_tok = [tokens(l["en"]) for l in LINES]
    N = len(SEG); M = len(LINES)
    if verbose: print(f"  segments={N} lines={M}", flush=True)

    # === 阶段 1: 候选 + LIS ===
    candidates = []
    for j in range(N):
        st = seg_tok[j]
        if not st: continue
        matches = []
        for i in range(M):
            lt = line_tok[i]
            if not lt: continue
            inter = len(lt & st)
            if inter == 0: continue
            cov = inter / len(lt)
            # 用更松阈值产生更多候选，靠 LIS 排序选出单调子集
            if len(lt) <= 2: thresh = 1.0      # 短句必须完全匹配
            elif len(lt) <= 4: thresh = 0.5
            else: thresh = 0.30
            if cov < thresh: continue
            # 分数 = 覆盖率 × 行长（长句更可信）
            score = cov * cov * min(len(lt), 12)  # 平方放大高 cov 的优势
            matches.append((i, score))
        matches.sort(key=lambda x: -x[1])
        for i, s in matches[:5]:  # 每 seg 多保留几个候选
            candidates.append((j, i, s))

    if verbose: print(f"  phase1 candidates={len(candidates)}", flush=True)

    candidates.sort(key=lambda x: (x[0], x[1]))
    K = len(candidates)
    dp = [0.0] * K
    prev = [-1] * K
    for k in range(K):
        j_k, i_k, s_k = candidates[k]
        dp[k] = s_k
        for p in range(k - 1, -1, -1):
            j_p, i_p, _ = candidates[p]
            if j_p < j_k and i_p <= i_k:
                cand = dp[p] + (0 if i_p == i_k else s_k)
                if cand > dp[k]:
                    dp[k] = cand
                    prev[k] = p

    if K == 0:
        anchors = []
    else:
        best_k = max(range(K), key=lambda k: dp[k])
        chosen = []
        while best_k >= 0:
            chosen.append(best_k)
            best_k = prev[best_k]
        chosen.reverse()
        anchors = [(candidates[k][1], candidates[k][0]) for k in chosen]
        seen = set(); uniq = []
        for li, sj in anchors:
            if li in seen: continue
            seen.add(li); uniq.append((li, sj))
        anchors = uniq

    if verbose: print(f"  phase1 anchors={len(anchors)} ({len(anchors)*100/M:.1f}%)", flush=True)

    line_t = [None] * M
    for li, sj in anchors:
        line_t[li] = SEG[sj]["start"]

    # === 阶段 2: 锚点间局部 DP ===
    extended = [(-1, -1)] + anchors + [(M, N)]
    LOCAL_THRESH = 0.20  # 局部 DP 用更松阈值（已被锚点约束）
    refined = 0

    for k in range(len(extended) - 1):
        l_a, s_a = extended[k]
        l_b, s_b = extended[k + 1]
        lr = list(range(l_a + 1, l_b))
        sr = list(range(s_a + 1, s_b))
        if not lr or not sr: continue
        if len(lr) > 80 or len(sr) > 120: continue

        nL = len(lr); nS = len(sr)
        dp2 = [[0.0] * (nS + 1) for _ in range(nL + 1)]
        back = [[0] * (nS + 1) for _ in range(nL + 1)]
        for i in range(1, nL + 1):
            li = lr[i - 1]
            lt = line_tok[li]
            for j in range(1, nS + 1):
                sj = sr[j - 1]
                st = seg_tok[sj]
                s = 0.0
                if lt and st:
                    inter = len(lt & st)
                    if inter > 0:
                        cov = inter / len(lt)
                        if cov >= LOCAL_THRESH:
                            s = cov * min(len(lt), 5)
                    # 即使无交集，对长台词允许小奖励（whisper 识别错误）
                    elif len(lt) >= 5 and len(st) >= 5:
                        # 字符级模糊：把 line en 整段和 seg text 比相似度
                        # 太慢，跳过
                        pass
                v_diag = dp2[i-1][j-1] + s
                # 跳过台词/segment 的轻微惩罚（鼓励对角线 = 一对一）
                v_up = dp2[i-1][j] - 0.05
                v_left = dp2[i][j-1] - 0.05
                best = v_diag; d = 0
                if v_up > best: best, d = v_up, 1
                if v_left > best: best, d = v_left, 2
                dp2[i][j] = best
                back[i][j] = d
        i, j = nL, nS
        while i > 0 and j > 0:
            if back[i][j] == 0:
                li = lr[i-1]; sj = sr[j-1]
                if line_t[li] is None:
                    lt = line_tok[li]; st = seg_tok[sj]
                    # 接受：有任意交集 OR 长台词通过位置启发式（fallback）
                    if lt and st and len(lt & st) > 0:
                        line_t[li] = SEG[sj]["start"]
                        refined += 1
                i -= 1; j -= 1
            elif back[i][j] == 1: i -= 1
            else: j -= 1

    total_anchored = sum(1 for t in line_t if t is not None)
    if verbose: print(f"  phase2 refined={refined}, total={total_anchored} ({total_anchored*100/M:.1f}%)", flush=True)

    # === 阶段 3: 边界 + 插值 + 单调修复 ===
    if line_t[0] is None: line_t[0] = 0.0
    if line_t[-1] is None: line_t[-1] = SEG[-1]["end"] if SEG else 0.0
    last_i = 0
    for i in range(1, M):
        if line_t[i] is not None:
            if i - last_i > 1:
                t0, t1 = line_t[last_i], line_t[i]
                if t1 < t0: t1 = t0
                gap = i - last_i
                for kk in range(last_i + 1, i):
                    line_t[kk] = round(t0 + (t1 - t0) * (kk - last_i) / gap, 3)
            last_i = i
    for i in range(1, M):
        if line_t[i] < line_t[i-1]:
            line_t[i] = line_t[i-1]

    out = []
    for i in range(M):
        out.append({
            "idx": LINES[i]["idx"],
            "zh": LINES[i]["zh"],
            "en": LINES[i]["en"],
            "t": round(line_t[i], 3),
        })
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    return {
        "segments": N,
        "lines": M,
        "phase1_anchors": len(anchors),
        "phase2_refined": refined,
        "anchored": total_anchored,
        "anchor_rate": total_anchored / M if M else 0.0,
    }


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        seg = Path(sys.argv[1])
        lines = Path(sys.argv[2])
        out = Path(sys.argv[3])
    else:
        HERE = Path(__file__).parent
        seg = HERE.parent / "疯狂动物城_app" / "segments.json"
        lines = HERE.parent / "疯狂动物城_app" / "lines.json"
        out = HERE / "timeline_v2.json"
    info = align(seg, lines, out)
    print(f"\n=== 结果 ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"output → {out}")
