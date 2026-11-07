# chorus-finder-rs

English | [简体中文](README.zh-CN.md)

Find the chorus start/end of an audio file from its lyrics, as a single native
executable. No Python runtime is required.

The chorus detection is deterministic: repeated lyric lines are paired and
clustered by their time offset, expanded to the bar grid, and scored by repeat
support, line coverage, bar alignment, and RMS energy. No LLM is involved.

## Download

Prebuilt packages are published on [Releases](https://github.com/Voemp/chorus-finder-rs/releases):

- Windows x86_64
- Linux x86_64
- macOS Apple Silicon / Intel

## Build

Requirements:

- Rust (MSVC toolchain on Windows)
- CMake and LLVM/libclang (to build whisper.cpp)

Windows:

```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
cargo build --release
```

Linux:

```bash
sudo apt-get install -y cmake libclang-dev
cargo build --release
```

macOS (Xcode Command Line Tools provide libclang):

```bash
brew install cmake
cargo build --release
```

The executable is `target/release/chorus-finder` (`chorus-finder.exe` on Windows).

At runtime the timed-lyrics path needs no extra files. The untimed-lyrics path
needs a Whisper ggml model:

- `ggml-base.bin` (~142 MB): https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
- `ggml-small.bin` (~466 MB): better Chinese accuracy

## Usage

```powershell
# Lyrics from an explicit file, a same-name .lrc/.txt, or embedded tags
chorus-finder.exe "song.mp3"

# Return the repeat of the best chorus block instead of its first occurrence
chorus-finder.exe "song.mp3" --choose best

# Full JSON
chorus-finder.exe "song.mp3" --json

# Untimed lyrics (plain text)
chorus-finder.exe "song.mp3" --lyrics lyrics.txt --whisper-model models\ggml-base.bin
```

## Lyrics sources

Priority:

1. `--lyrics <file>`
2. Same-name `.lrc` / `.LRC` / `.txt` / `.TXT` beside the audio
3. Embedded tags (`USLT`/unsynchronized lyrics via lofty)

Supported formats:

- word-level LRC: `[mm:ss.xx]字[mm:ss.xx]字...`
- line-level LRC: `[mm:ss.xx]整行歌词`
- plain lyrics: aligned with Whisper token timestamps, using toneless pinyin
  for Chinese characters so ASR homophones still match

## Output

`--json` prints the full result; the example below is abridged. The complete
top-level fields are `audio`, `lyrics_source`, `lyrics_format`, `bpm`,
`duration`, `chorus`, `first_chorus`, `best_chorus`, `occurrences`,
`repeat_support`, `line_coverage`, `alignment_coverage`, and `candidates`.
Each occurrence also carries numeric `start` and `end` seconds. Without
`--json`, four plain-text lines are printed.

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

Word-level LRC has exact end times when the line ends with a timestamp; if it
does not, the last word falls back to start + 0.3 s. Line-level LRC and untimed
lyrics only provide line starts, so their chorus end is estimated from the
onset envelope; expect roughly 1-3 seconds of error for those formats.
