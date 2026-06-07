# -*- coding: utf-8 -*-
"""
扫描音频和台词目录，建立 manifest.json：每部电影的音频+台词配对。
"""
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path(r"D:/OneDrive - IPowerBI/视频/外刊课/108部英文动画电影（音频+台词本）")
AUDIO_DIR = ROOT / "动画音频"
SCRIPT_DIR = ROOT / "台词本" / "台词本TXT格式"
OUT = Path(__file__).parent / "manifest.json"

def normalize_title(name: str) -> str:
    name = re.sub(r"\.(mp3|txt)$", "", name, flags=re.I)
    name = re.sub(r"^\d+\s*[\.\s]\s*", "", name)
    name = re.sub(r"^\d+\s*", "", name)
    return name.strip()

def extract_id(name: str):
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else None

def core_zh(name: str) -> str:
    """提取中文核心：移除所有非中文字符"""
    return re.sub(r"[^一-鿿]", "", name)

audios = sorted(AUDIO_DIR.glob("*.mp3"))
scripts = sorted(SCRIPT_DIR.glob("*.txt"))

audio_info = []
for p in audios:
    raw = p.stem
    audio_info.append({
        "id": extract_id(raw),
        "title": normalize_title(raw),
        "core": core_zh(normalize_title(raw)),
        "path": str(p),
        "raw": raw,
    })

script_info = []
for p in scripts:
    raw = p.stem
    script_info.append({
        "id": extract_id(raw),
        "title": normalize_title(raw),
        "core": core_zh(normalize_title(raw)),
        "path": str(p),
        "raw": raw,
    })

print(f"audios={len(audio_info)} scripts={len(script_info)}")

# 手工别名修正：把音频名映射到台词名（处理同片不同译名）
ALIASES = {
    "超人总动员1": "超人特工队",  # 同片，不同译名
    "超人特工队": None,  # 占位，让别名能找过来
    "爱宠大机密": None,
    # "恐龙世纪" 和 "冰河世纪1" 不是同一部，禁止匹配
}
BLOCKED_PAIRS = {
    ("冰河世纪1", "恐龙世纪"),
    ("冰河世纪", "恐龙世纪"),
}

# 全局最优匹配：先计算所有 audio×script 相似度，然后按相似度从高到低贪心
pairs = []  # (score, ai, si, how)
for ai, a in enumerate(audio_info):
    if not a["core"]: continue
    for si, s in enumerate(script_info):
        if not s["core"]: continue
        if (a["title"], s["title"]) in BLOCKED_PAIRS:
            continue
        # 别名映射
        alias = ALIASES.get(a["title"])
        if alias and s["title"] == alias:
            pairs.append((1.0, ai, si, "alias"))
            continue
        if a["core"] == s["core"]:
            score = 1.0; how = "exact"
        elif a["core"] in s["core"] or s["core"] in a["core"]:
            l1, l2 = len(a["core"]), len(s["core"])
            score = min(l1, l2) / max(l1, l2)
            how = "contain"
        else:
            score = SequenceMatcher(None, a["core"], s["core"]).ratio()
            if score < 0.6: continue
            how = "similar"
        pairs.append((score, ai, si, how))

# 按分数从高到低排序，贪心匹配
pairs.sort(reverse=True)
audio_to_script = {}  # ai -> (si, how, score)
script_to_audio = {}  # si -> ai
for score, ai, si, how in pairs:
    if ai in audio_to_script: continue
    if si in script_to_audio: continue
    audio_to_script[ai] = (si, how, score)
    script_to_audio[si] = ai

manifest = []
for ai, a in enumerate(audio_info):
    if ai in audio_to_script:
        si, how, score = audio_to_script[ai]
        s = script_info[si]
        manifest.append({
            "id": a["id"],
            "title": a["title"],
            "audio": a["path"],
            "script": s["path"],
            "match": how if score >= 0.95 else f"{how}_{score:.2f}",
            "script_title": s["title"],
        })
    else:
        manifest.append({
            "id": a["id"],
            "title": a["title"],
            "audio": a["path"],
            "script": None,
            "match": "no_match",
            "script_title": None,
        })

# 加上未配对的台词
for si, s in enumerate(script_info):
    if si not in script_to_audio:
        manifest.append({
            "id": None,
            "title": s["title"],
            "audio": None,
            "script": s["path"],
            "match": "script_only",
            "script_title": s["title"],
        })

# 按音频 id 排序
manifest.sort(key=lambda m: (m["id"] is None, m["id"] or 9999))

OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# 统计
n_exact = sum(1 for m in manifest if m["match"] == "exact")
n_fuzzy = sum(1 for m in manifest if m["match"].startswith("contain"))
n_similar = sum(1 for m in manifest if m["match"].startswith("similar"))
n_nomatch = sum(1 for m in manifest if m["match"] == "no_match")
n_scriptonly = sum(1 for m in manifest if m["match"] == "script_only")
print(f"\n=== 匹配统计 ===")
print(f"exact:       {n_exact}")
print(f"fuzzy:       {n_fuzzy}")
print(f"similar:     {n_similar}")
print(f"no_match:    {n_nomatch}  (音频有但台词找不到)")
print(f"script_only: {n_scriptonly}  (台词有但音频缺失)")
print(f"total:       {len(manifest)}")
print(f"\nsaved → {OUT}")

# 打印不确定的匹配
import sys
sys.stdout.reconfigure(encoding='utf-8')
print("\n=== 需要人工确认的匹配 ===")
for m in manifest:
    if m["match"].startswith("similar") or m["match"] in ("no_match", "script_only"):
        a_t = m["title"][:30] if m["title"] else "---"
        s_t = m["script_title"][:30] if m["script_title"] else "---"
        print(f"  [{m['match']:14}] audio={a_t!r:34}  script={s_t!r}")
