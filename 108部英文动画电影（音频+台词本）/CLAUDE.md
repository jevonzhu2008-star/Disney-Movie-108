# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Builds standalone single-file HTML "audio + scrolling bilingual subtitle" apps for ~96 English animated movies. Each app embeds an aligned timeline so clicking a subtitle line seeks the audio, and the active line auto-scrolls. Input is raw mp3 audio + plain-text Chinese/English subtitle scripts; output is `<title>.html` + the original mp3 referenced by relative path.

Source data layout (do not move):
- `动画音频/` — 101 `.mp3` files named `<id>.<title>.mp3`
- `台词本/台词本TXT格式/` — 108 `.txt` files named `<id>.<title>.txt`, format is one Chinese-then-English line per spoken line separated by **two or more spaces** (no timestamps)
- The id schemes between audio and script directories **do not match** — pairing is done by Chinese title in `_build/build_manifest.py`

## Pipeline

The end-to-end pipeline is `_build/process_movie.py`, which orchestrates four stages per movie. Read its `process_one()` to understand the data flow:

1. **transcribe** — `faster-whisper medium` on CUDA. Outputs `segments.json` `[{start, end, text}]` (English ASR). About 6 min/movie on small, 12-18 min/movie on medium (RTX 4060).
2. **parse_script** — read the `.txt` with `utf-8-sig`, split each non-empty line on `r" {2,}"` into `(zh, en)`. Lines without that gap fall back to ASCII-ratio detection. Outputs `lines.json`.
3. **align** (delegated to `_build/align2.py` via subprocess) — two-phase alignment of `segments.json` to `lines.json`:
   - **Phase 1**: For each segment, compute token-coverage (`|line∩seg| / |line|`) against every line; emit top-5 candidates with `cov ≥ 0.30..1.0` (threshold scales with line length). Then run **weighted LIS** on `(seg_j, line_i, score)` triples to extract the longest monotone subset → these are the anchors (~75% of lines).
   - **Phase 2**: Between adjacent anchors, run a small local Needleman-Wunsch-style DP (`O(linesₖ × segsₖ)`, skipped when either dimension > 60/100) with looser threshold `cov ≥ 0.20`. Fills another ~5% of lines (final anchor rate ~80%).
   - **Phase 3**: Linear interpolation between anchored times, then enforce monotonicity.
4. **build_html** — substitute `__AUDIO_SRC__` and `__TIMELINE__` placeholders in `疯狂动物城_app/template.html` (this template is the canonical UI). Audio path is `os.path.relpath(audio, work_dir)` URL-quoted.

Per-movie output goes to `movies/<id:03d>_<title>/{segments,lines,timeline}.json + <title>.html + meta.json`. `meta.json` is the contract `_build/build_index.py` reads — it must contain `id, title, duration, lines, anchor_rate, html_path`.

`_build/build_index.py` walks `movies/*/meta.json` and writes `index.html` at the project root.

## Commands

All Python commands assume cwd = `_build/` (or absolute path). Use `python` (Windows Python 3.14 in PATH); `pip` is not aliased — use `python -m pip`.

```powershell
# Rebuild the audio↔script manifest (re-run if you fix an alias in build_manifest.py ALIASES)
python _build/build_manifest.py

# Process one movie by its AUDIO id (the leading number in 动画音频/N.title.mp3)
python _build/process_movie.py 5

# Batch with resume — skips movies whose meta.json exists
python _build/process_movie.py --resume

# Range
python _build/process_movie.py --range 1-20

# Regenerate the index after new movies finish
python _build/build_index.py

# Re-align an existing movie without re-transcribing
python _build/align2.py movies/005_海洋奇缘/segments.json movies/005_海洋奇缘/lines.json movies/005_海洋奇缘/timeline.json
```

`疯狂动物城_app/` is the **original prototype** kept as the canonical template source and as the legacy single-output `疯狂动物城.html` at the project root. Don't refactor it — `process_movie.py` reads `疯狂动物城_app/template.html` directly.

## Environment gotchas

- **CUDA DLLs**: `faster-whisper` (via `ctranslate2`) needs `cublas64_*.dll` + `cudnn*.dll`. The system CUDA Toolkit is not installed; instead, the pip packages `nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` provide them under `site-packages/nvidia/{cublas,cudnn,cuda_nvrtc}/bin`. Every entry-point script must `os.add_dll_directory()` those paths BEFORE importing `faster_whisper` — see `_add_cuda_dlls()` at the top of `process_movie.py`. Without this, CUDA silently falls back to CPU (10× slower).
- **HF model downloads from China**: `Systran/faster-whisper-medium` (~1.5GB) routes through `cas-bridge.xethub.hf.co` even via `hf-mirror.com`, which is slow. `process_movie.py` sets `HF_ENDPOINT=https://hf-mirror.com` and `HF_HUB_ENABLE_HF_TRANSFER=1` for the metadata calls. If `model.bin` stalls, manually `curl` it from the resolved CDN URL into `~/.cache/huggingface/hub/models--Systran--faster-whisper-medium/blobs/`.
- **Console encoding**: Windows shell is GBK. `print()`-ing CJK from Python typically errors with `UnicodeEncodeError` — write to files with `encoding="utf-8"` for results; use `sys.stdout.reconfigure(encoding='utf-8')` or `sys.stdout.buffer.write(...encode('utf-8'))` for human-readable debug output.
- **Script encoding**: subtitle `.txt` files are UTF-8 **with BOM** — always open with `encoding="utf-8-sig"`.

## Title-matching quirks

`_build/build_manifest.py` does global-greedy matching on the Chinese-character-only "core" of each filename. Same-movie different-translation cases (e.g. audio `超人总动员1` vs script `超人特工队`) live in the hand-curated `ALIASES` dict at the top of the file. The `BLOCKED_PAIRS` set vetoes false positives the similarity score would otherwise accept (e.g. `冰河世纪1` vs `恐龙世纪`). When `manifest.json` shows a wrong pair after a re-run, **fix the dict, don't post-edit the JSON** — the manifest is regenerated from disk on every run.

7 audios have no matching script (`了不起的狐狸爸爸`, `愤怒的小鸟`, `海洋之歌`, `小熊维尼`, etc.) and 11 scripts have no audio — these are unfixable from the dataset side; manifest reports them as `no_match` / `script_only` and `process_movie.py` skips them.

## Alignment quality knobs

When a movie's anchor rate is low (< 60%), it's almost always one of:
- **Phase 1 threshold too tight** for that movie's spoken style — lower `thresh` in `align2.py` candidate loop (current: 0.30 for ≥5-word lines).
- **Phase 2 segment too large** — adjust the `if len(lr) > 60 or len(sr) > 100: continue` guard. Going to 100/150 catches more refinements but is `O(n²)` per gap.
- **Short common words ("Yes", "No", "Yeah")** matching too many lines — Phase 1 enforces `thresh = 1.0` for ≤2-word lines specifically to prevent these from corrupting the LIS.

Interpolated lines (the ~20% without an anchor) still get a `t` value — the HTML player's "字幕偏移" slider (`#offsetSlider`, ±30s, persisted to localStorage as `zoo_offset`) lets the user nudge globally when systematic drift remains.
