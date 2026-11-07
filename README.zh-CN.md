# chorus-finder-rs

[English](README.md) | 简体中文

从歌词定位音频中副歌的起止时间，单个原生可执行文件，无需 Python 运行时。

副歌检测是确定性的：重复的歌词行按时间偏移配对并聚类，扩展到小节网格，再按重复支持度、歌词覆盖率、小节对齐和 RMS 能量打分。全程不使用 LLM。

## 下载

从 [Releases](https://github.com/Voemp/chorus-finder-rs/releases) 获取预编译包：

- Windows x86_64
- Linux x86_64
- macOS Apple Silicon / Intel

## 构建

要求：

- Rust（Windows 使用 MSVC 工具链）
- CMake 与 LLVM/libclang（仅用于构建 whisper.cpp）

Windows：

```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
cargo build --release
```

Linux：

```bash
sudo apt-get install -y cmake libclang-dev
cargo build --release
```

macOS（Xcode Command Line Tools 自带 libclang）：

```bash
brew install cmake
cargo build --release
```

可执行文件位于 `target/release/chorus-finder`（Windows 为 `chorus-finder.exe`）。

运行时，带时间戳的歌词路径不需要额外文件。无时间戳的歌词路径需要一个 Whisper ggml 模型：

- `ggml-base.bin`（约 142 MB）：https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
- `ggml-small.bin`（约 466 MB）：中文识别更准确

## 用法

```powershell
# 歌词来自显式文件、同名 .lrc/.txt，或音频内嵌标签
chorus-finder.exe "song.mp3"

# 返回最高分副歌段的重复出现位置，而不是它第一次出现的位置
chorus-finder.exe "song.mp3" --choose best

# 输出完整 JSON
chorus-finder.exe "song.mp3" --json

# 无时间戳歌词（纯文本）
chorus-finder.exe "song.mp3" --lyrics lyrics.txt --whisper-model models\ggml-base.bin
```

## 歌词来源

优先级：

1. `--lyrics <file>`
2. 音频旁的 `.lrc` / `.LRC` / `.txt` / `.TXT` 同名文件
3. 内嵌标签（lofty 读取的 `USLT`/unsynchronized lyrics）

支持的格式：

- 逐字 LRC：`[mm:ss.xx]字[mm:ss.xx]字...`
- 逐行 LRC：`[mm:ss.xx]整行歌词`
- 纯歌词：用 Whisper token 时间戳对齐；中文按无声调拼音比较，因此 ASR 同音字仍能匹配

## 输出

`--json` 输出完整结果；下方示例为节选。完整顶层字段为 `audio`、
`lyrics_source`、`lyrics_format`、`bpm`、`duration`、`chorus`、
`first_chorus`、`best_chorus`、`occurrences`、`repeat_support`、
`line_coverage`、`alignment_coverage` 和 `candidates`。每个 occurrence 还包含
数字形式的 `start` 与 `end` 秒数。不加 `--json` 时只打印四行纯文本。

```json
{
  "chorus": {
    "start": 60.81,
    "end": 95.66,
    "start_mmss": "01:00.81",
    "end_mmss": "01:35.66",
    "body_bars": 15.75,
    "confidence": 0.82
  },
  "occurrences": [
    { "start_mmss": "01:00.81", "end_mmss": "01:35.66" },
    { "start_mmss": "03:41.50", "end_mmss": "04:12.45" }
  ]
}
```

逐字 LRC 在行尾带时间戳时结束时间精确；若没有，最后一个字回退为起始时间 +
0.3 秒。逐行 LRC 和纯歌词只提供行起始时间，副歌结束时间由起音包络估计，这些格式通常有 1–3 秒误差。
