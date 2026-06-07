# -*- coding: utf-8 -*-
"""
扫描 movies/ 目录下所有 meta.json，生成根目录的 index.html。
"""
import json
import urllib.parse
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
MOVIES_DIR = ROOT / "movies"
OUT = ROOT / "index.html"

entries = []
if MOVIES_DIR.exists():
    for sub in sorted(MOVIES_DIR.iterdir()):
        meta_path = sub / "meta.json"
        if not meta_path.exists(): continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # 找 html 文件
            html_files = list(sub.glob("*.html"))
            if not html_files: continue
            meta["html_path"] = f"movies/{sub.name}/{html_files[0].name}"
            entries.append(meta)
        except Exception as e:
            print(f"skip {sub}: {e}")

# 按 id 排序
entries.sort(key=lambda e: e.get("id") or 9999)

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>动画电影台词学习 — 索引</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif;
  background: #1a1a2e; color: #eee; min-height: 100vh;
}
.container { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 28px; margin-bottom: 4px; }
.subtitle { color: #aaa; font-size: 14px; margin-bottom: 24px; }
.stats { display: flex; gap: 16px; margin-bottom: 20px; font-size: 13px; color: #ccc; flex-wrap: wrap; }
.stats span { background: #16213e; padding: 6px 12px; border-radius: 6px; }
.search { width: 100%; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; background: #16213e; color: #eee; font-size: 15px; margin-bottom: 16px; }
.search:focus { outline: none; border-color: #e94560; }
.grid {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
}
.card {
  background: #16213e; border-radius: 10px; padding: 14px 16px;
  text-decoration: none; color: inherit; transition: all .2s;
  border: 1px solid #1f2a44;
}
.card:hover { background: #1f2a44; border-color: #e94560; transform: translateY(-1px); }
.card .id { font-size: 11px; color: #666; font-family: monospace; }
.card .title { font-size: 16px; font-weight: 600; margin: 2px 0 8px; color: #e94560; }
.card .meta-row { display: flex; gap: 10px; font-size: 12px; color: #aaa; flex-wrap: wrap; }
.card .meta-row span { white-space: nowrap; }
.anchor-bar {
  height: 4px; background: #222; border-radius: 2px; margin-top: 8px; overflow: hidden;
}
.anchor-bar > div { height: 100%; background: linear-gradient(90deg, #e94560, #f0a;); }
footer { margin-top: 32px; text-align: center; color: #555; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
  <h1>🎬 动画电影台词学习</h1>
  <p class="subtitle">音频 + 中英对照台词 + 时间轴对齐 · 点击卡片进入</p>
  <div class="stats">
    <span>📚 已处理: <b id="cnt">""" + str(len(entries)) + """</b> 部</span>
    <span>⏱️ 总时长: <b>""" + f"{sum(e.get('duration',0) for e in entries)/3600:.1f}" + """ 小时</b></span>
    <span>📊 平均锚点率: <b>""" + (f"{sum(e.get('anchor_rate',0) for e in entries)/len(entries)*100:.0f}%" if entries else "—") + """</b></span>
  </div>
  <input class="search" id="search" placeholder="搜索电影标题…" oninput="filt(this.value)">
  <div class="grid" id="grid">
"""

for e in entries:
    duration_min = (e.get("duration", 0) or 0) / 60
    anchor_pct = (e.get("anchor_rate", 0) or 0) * 100
    href = urllib.parse.quote(e["html_path"], safe="/.")
    html += f"""    <a class="card" href="{href}" data-title="{e['title']}">
      <div class="id">#{e['id']:03d}</div>
      <div class="title">{e['title']}</div>
      <div class="meta-row">
        <span>⏱️ {duration_min:.0f} 分钟</span>
        <span>📝 {int(e.get('lines',0))} 行</span>
        <span>📍 锚点 {anchor_pct:.0f}%</span>
      </div>
      <div class="anchor-bar"><div style="width:{anchor_pct:.0f}%"></div></div>
    </a>
"""

html += """  </div>
  <footer>local · 数据来自 faster-whisper 转录 + 序列对齐</footer>
</div>
<script>
function filt(q) {
  q = q.trim().toLowerCase();
  document.querySelectorAll(".card").forEach(c => {
    c.style.display = !q || c.dataset.title.toLowerCase().includes(q) ? "" : "none";
  });
}
</script>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"saved → {OUT}")
print(f"included {len(entries)} movies")
