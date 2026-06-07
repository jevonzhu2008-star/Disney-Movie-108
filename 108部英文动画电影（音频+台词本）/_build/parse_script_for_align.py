# -*- coding: utf-8 -*-
import json, os, re

# Config
ROOT = "D:/OneDrive - IPowerBI/视频/外刊课/108部英文动画电影（音频+台词本）"
HERE = os.path.join(ROOT, "_build")
TMP = os.environ.get("TEMP", "C:/Temp") + "/zoo_align"
os.makedirs(TMP, exist_ok=True)

# Script
lines = []
with open(os.path.join(ROOT, "台词本", "台词本TXT格式", "004.疯狂动物城.txt"), encoding="utf-8-sig") as f:
    idx = 0
    for raw in f:
        raw = raw.rstrip("\r\n")
        if not raw.strip():
            continue
        body = raw.strip(); idx += 1
        if idx == 1:
            lines.append({"idx": idx, "zh": body, "en": ""}); continue
        parts = re.split(r" {2,}", body, maxsplit=1)
        if len(parts) == 2:
            zh, en = parts[0].strip(), parts[1].strip()
        else:
            asc = sum(1 for c in body if ord(c) < 128)
            zh, en = (body, "") if asc / max(1, len(body)) <= 0.7 else ("", body)
        lines.append({"idx": idx, "zh": zh, "en": en})

out_path = os.path.join(TMP, "lines.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(lines, f, ensure_ascii=False)
print(f"Lines: {len(lines)}")
print(f"First: {lines[0]}")
print(f"Last: {lines[-1]}")
print(f"Saved to {out_path}")
