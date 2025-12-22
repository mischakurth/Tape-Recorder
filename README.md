# Bandsalat

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗  █████╗ ███╗   ██╗██████╗ ███████╗ █████╗ ██╗      █████╗ ████████╗║
║    ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔════╝██╔══██╗██║     ██╔══██╗╚══██╔══╝║
║    ██████╔╝███████║██╔██╗ ██║██║  ██║███████╗███████║██║     ███████║   ██║   ║
║    ██╔══██╗██╔══██║██║╚██╗██║██║  ██║╚════██║██╔══██║██║     ██╔══██║   ██║   ║
║    ██████╔╝██║  ██║██║ ╚████║██████╔╝███████║██║  ██║███████╗██║  ██║   ██║   ║
║    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ║
║                                                                               ║
║                   Machine Learning Emulation of a Tape Deck                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

> Transform clean digital audio to sound like it was recorded through a vintage Revox A77 reel-to-reel tape recorder.

![Revox A77 tape recorder setup with HS ORF tape](revox_a77_hs_orf_tape_recorder.jpg)

---

## What is Bandsalat?

In colloquial German, **"Bandsalat"** (literally "tape salad") refers to tangled magnetic tape within the mechanics of a playback device, such as a cassette or video recorder. The tape can become damaged due to rapidly switching between forward and rewind, uneven tape movement, or excessively loose or tight winding.

### Project Goal

We want to digitally reproduce the sound characteristics of an old tape recorder. The distinctive sonic character of vintage tape recordings—warmth, subtle saturation, and unique harmonic distortion—stands in contrast to the precision and clarity of modern digital recordings.

**Our approach**: Train a machine learning model on paired "before/after" audio samples to learn the acoustic transformation that analog tape imparts on audio.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BANDSALAT PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BEFORE (Clean)              TAPE RECORDING              AFTER (Tape)      │
│   ┌───────────┐              ┌───────────┐              ┌───────────┐       │
│   │  Digital  │    ────▶     │  Revox    │    ────▶     │   Tape    │       │
│   │   Audio   │              │    A77    │              │ Processed │       │
│   └───────────┘              └───────────┘              └───────────┘       │
│        │                                                      │             │
│        │                    ML TRAINING                       │             │
│        └──────────────────────┬───────────────────────────────┘             │
│                               │                                             │
│                               ▼                                             │
│                     ┌─────────────────┐                                     │
│                     │   Transformer   │                                     │
│                     │     Model       │                                     │
│                     └─────────────────┘                                     │
│                               │                                             │
│                               ▼                                             │
│                     ┌─────────────────┐                                     │
│                     │   Clean Audio   │──▶ "Virtual Tape" Output            │
│                     │     (New)       │                                     │
│                     └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

---

## Technical Specifications

| Specification | Value |
|---------------|-------|
| Sample Rate | 96 kHz |
| Bit Depth | 32-bit float |
| Channels | Stereo |

### Equipment

| Equipment | Model |
|-----------|-------|
| Tape Recorder | Revox A77 |
| Tape | HS ORF (broadcast quality) |

---

## Prerequisites

- **Python** 3.12 or higher
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **ffmpeg** - For audio format conversion

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

---


## Acknowledgments

- **Revox A77** - The iconic Swiss tape recorder used for data collection
- **Free Spoken Digit Dataset** - Open dataset for spoken digit recordings
- **MusicRadar** - Drum sample library

---

*Built for the Christmas Hackathon 2025*
