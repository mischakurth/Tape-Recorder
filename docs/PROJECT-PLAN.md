# Bandsalat Project Plan

> Machine learning emulation of a vintage tape deck recorder

---

## Project Vision

We want to digitally reproduce the sound characteristics of an old tape recorder. The distinctive sonic character of vintage tape recordings—warmth, subtle saturation, and unique harmonic distortion—stands in contrast to the precision and clarity of modern digital recordings.

**Goal**: Train a machine learning model that transforms clean digital audio to sound as if it was recorded and played back through an analog tape machine (specifically a Revox A77 reel-to-reel recorder).

---

## ML Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BANDSALAT ML PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │   1.     │   │   2.     │   │   3.     │   │   4.     │   │   5.     │  │
│  │  Data    │──▶│  Pre-    │──▶│ Compile  │──▶│  Tape    │──▶│  Align   │  │
│  │Collection│   │ process  │   │  Audio   │   │ Record   │   │  Pairs   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │                                                            │        │
│       ▼                                                            ▼        │
│  ┌─────────────────────┐                              ┌──────────────────┐  │
│  │ - YouTube drums     │                              │    ┌──────────┐  │  │
│  │ - MusicRadar samples│                              │    │   6.     │  │  │
│  │ - Spoken digits     │                              │    │ Chunking │  │  │
│  └─────────────────────┘                              │    │  50ms    │  │  │
│                                                       │    └────┬─────┘  │  │
│                                                       │         │        │  │
│                                                       │         ▼        │  │
│                                                       │    ┌──────────┐  │  │
│                                                       │    │   7.     │  │  │
│                                                       │    │  Model   │  │  │
│                                                       │    │ Training │  │  │
│                                                       │    └──────────┘  │  │
│                                                       └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Preparation

### 1.1 Audio Source Acquisition

**Sources**:
- Drum samples scraped from YouTube (38 URLs)
- MusicRadar drum sample library (~15,000 files)
- Free Spoken Digit Dataset (3,000 files for batch numbering)
- Custom acoustic drum samples

**Total raw data**: 14 GB (+20 GB additional), representing 90 minutes of recordings.

### 1.2 Preprocessing

**Amplitude Normalization**:
Create multiple amplitude variations for each source file to capture the tape's behavior at different input levels.

| Target dBFS | Purpose |
|-------------|---------|
| -0.1 dB | Near-peak levels |
| -3 dB | Moderate high |
| -6 dB | Standard level |
| -12 dB | Conservative |
| -24 dB | Low level |
| -48 dB | Very quiet |

**Result**: 972 source files → 4,850 normalized files

### 1.3 Audio Compilation

**Structure of compiled audio batches**:
```
┌───────────────────────────────────────────────────────────────────────────┐
│ BATCH STRUCTURE                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ [SPOKEN DIGITS: "zero zero one"]                                          │
│         │                                                                 │
│         ▼                                                                 │
│ ┌─────┬─────┬─────┬─────────────────────┬─────┬─────┬─────┬─────────────┐ │
│ │Sil. │Beep │Sil. │   AUDIO SEGMENT 1   │Sil. │Beep │Sil. │  SEGMENT 2  │ │
│ │0.5s │     │0.5s │                     │0.5s │     │0.5s │             │ │
│ └─────┴─────┴─────┴─────────────────────┴─────┴─────┴─────┴─────────────┘ │
│                                                                           │
│ Maximum batch duration: 720 seconds (12 minutes)                          │
└───────────────────────────────────────────────────────────────────────────┘
```

**Components**:
- **Spoken digits**: Identify batch number (e.g., "zero zero one" = batch 001)
- **Beep delimiters**: 3-beep sequences between segments for alignment
- **Silence gaps**: 500ms-1000ms surrounding delimiters
- **Audio segments**: Source audio samples

**Output**: Nine 10-minute WAV files per tape side.

### 1.4 Tape Recording

**Equipment**:
- **Recorder**: Revox A77 reel-to-reel
- **Tape**: HS ORF (high-speed, broadcast-quality)

**Process**:
1. Play compiled audio through tape recorder input
2. Record to tape
3. Play back tape while recording output
4. Capture both "before" (original) and "after" (tape-processed) at 96kHz

---

## Phase 2: Data Processing

### 2.1 Alignment

The key challenge is precisely aligning the "before" and "after" recordings, as tape introduces timing variations.

**Silence Detection Algorithm**:
```python
# Parameters
threshold_dbfs = -50.0    # dBFS level below which audio is silent
chunk_duration_ms = 50    # Scanning window size
```

**Process**:
1. Scan audio in chunks to detect non-silent regions
2. Use sliding window of chunk_duration_ms
3. Mark regions where max_dBFS > threshold
4. Refine boundaries to millisecond precision using 1ms steps

**Handling Tape Speed Variations**:
- **Wow**: Slow speed variations (< 4 Hz)
- **Flutter**: Fast speed variations (4-20 Hz)
- Solution: Use delimiter beeps as sync points, align each segment independently

See [ALIGNMENT-ALGORITHMS.md](ALIGNMENT-ALGORITHMS.md) for detailed algorithm documentation.

### 2.2 Chunking

**Parameters**:
| Parameter | Value |
|-----------|-------|
| Chunk duration | 50ms |
| Overlap | 50% (25ms) |
| Sample rate | 96 kHz |
| Bit depth | 32-bit float |
| Channels | Stereo |

**Output**:
- Matched before/after pairs saved as individual WAV files
- Naming: `chunk_{index:06d}.wav`
- **Total pairs**: 130,834 training examples

---

## Phase 3: Model Development

### 3.1 Feature Extraction

**Mel Spectrogram Transformation**:
```python
mel_transform = MelSpectrogram(
    sample_rate=96000,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)
```

**Normalization**:
```python
mel_spec = torch.log(mel_spec + 1e-9)  # Log transform
```

**Why Mel Spectrograms?**
- Reduces dimensionality while retaining perceptually relevant information
- Aligned with human auditory perception
- Efficient for transformer processing

### 3.2 Model Architecture

**Convolutional Transformer**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Mel Spectrogram (batch × 1 × 128 × time)               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           CONVOLUTIONAL FRONTEND                        │   │
│  │  Conv2d(1, 32, kernel=3, stride=1, padding=1)          │   │
│  │  ReLU + MaxPool2d(2)                                    │   │
│  │  [Additional conv layers as needed]                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼ Flatten + Permute                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           TRANSFORMER ENCODER                           │   │
│  │  Layers: 6-12                                           │   │
│  │  Model dimension: 512                                   │   │
│  │  Attention heads: 8                                     │   │
│  │  Positional encodings: Required                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  OUTPUT: Transformed features                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Architecture?**
- **CNN Frontend**: Extracts local features, reduces dimensionality
- **Transformer**: Models global/long-range dependencies
- **Combined**: Efficient processing of high-resolution spectrograms

### 3.3 Training Configuration

```python
# Loss function
criterion = nn.MSELoss()  # Mean Squared Error for regression

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data split
# - Training: 90%
# - Validation: 5%
# - Test: 5%
```

**Training Loop**:
1. Load batch of before/after Mel spectrogram pairs
2. Forward pass through model
3. Compute MSE loss between predicted and actual "after"
4. Backpropagate and update weights
5. Validate every N batches
6. Save best model checkpoint

---

## Phase 4: Evaluation

### 4.1 Objective Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **SNR** | Signal-to-Noise Ratio | Higher = better |
| **LSD** | Log-Spectral Distance | Lower = closer to reference |
| **MCD** | Mel Cepstral Distortion | Lower = better timbral fidelity |
| **SI-SNR** | Scale-Invariant SNR | Better perceptual correlation |
| **PEAQ** | Perceptual Evaluation of Audio Quality | Approximates MOS |

### 4.2 Subjective Evaluation

**Mean Opinion Score (MOS)**:
- Listeners rate audio quality on 1-5 scale
- Average scores across multiple listeners

**A/B Comparison (CMOS)**:
- Present original tape recording vs. model output
- Listeners score which sounds more "authentic"

---

## Phase 5: Inference

**Pipeline**:
```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│  Clean   │──▶│    Mel    │──▶│  Model   │──▶│  Inverse  │──▶│  "Tape"  │
│  Audio   │   │Spectrogram│   │ Forward  │   │ Transform │   │  Output  │
│  (WAV)   │   │           │   │  Pass    │   │           │   │  (WAV)   │
└──────────┘   └───────────┘   └──────────┘   └───────────┘   └──────────┘
```

**Usage**:
```python
# Load trained model
model = load_model("bandsalat_v1.pth")

# Load input audio
waveform, sr = librosa.load("input.wav", sr=96000)

# Transform to "tape" version
tape_output = model.inference(waveform)

# Save output
sf.write("output_tape.wav", tape_output, sr)
```

---

## Current Progress

### Completed

- [x] Project setup with uv package manager
- [x] Configuration system (`config.json` + `Config` class)
- [x] Directory structure established
- [x] Audio assets organized
  - [x] Audio delimiters (beeps, silence)
  - [x] Free Spoken Digit Dataset (3,000 files)
  - [x] Sample drum recordings
- [x] Dataset-berta with ~4,586 before/after chunk pairs
- [x] Demo notebooks
  - [x] `demo-read-audiofiles.ipynb` - Load and visualize WAV files
  - [x] `demo-loudness-normalisation.ipynb` - Normalize audio levels
  - [x] `demo-audio-chunking.ipynb` - Split audio into chunks
  - [x] `demo-play-before-after-audio.ipynb` - Compare before/after
- [x] Audio compiler notebook (`audio-compiler.ipynb`)
- [x] Alignment algorithms documentation

### In Progress

- [ ] Dataset expansion (alfred, charlie datasets)
- [ ] Training data generation from raw tape recordings

### Planned

- [ ] Test/train/validation split generation
- [ ] Model training notebook
- [ ] Model evaluation notebook
- [ ] Inference notebook
- [ ] Hyperparameter tuning
- [ ] Model comparison experiments

---

## Available Datasets

| Dataset | Description | Files | Status |
|---------|-------------|-------|--------|
| `dataset-alfred` | Before/after recording pairs | 10 pairs | Complete |
| `dataset-berta` | Chunked before/after pairs, multiple normalizations | ~4,586 pairs | Complete |
| `dataset-charlie` | Before/after recording pairs | 9 pairs | Complete |

**Dataset-berta Details**:
- Tape recorder: Revox A77
- Tape: HS ORF (broadcast quality)
- Sample rate: 96 kHz
- Bit depth: 32-bit float

---

## Audio Assets Inventory

| Asset | Location | Count | Purpose |
|-------|----------|-------|---------|
| Audio delimiters | `data/audio/assets/audio_delimiters/` | ~10 files | Markers for alignment |
| Spoken digits | `data/audio/assets/free-spoken-digit-dataset/` | 3,000 files | Batch numbering |
| Drum samples | `data/audio/assets/samples/` | ~15 files | Training source material |

**Audio Delimiters**:
| File | Purpose |
|------|---------|
| `beep-07a.wav` | Short beep marker |
| `beep-08b.wav` | Alternative beep |
| `beep-10.wav` | Longer beep |
| `silence-0.5s.wav` | 500ms silence |
| `silence-1s.wav` | 1s silence |

**Free Spoken Digit Dataset**:
- Format: WAV (8kHz, mono)
- Content: Digits 0-9, 6 speakers, 50 recordings each
- Naming: `{digit}_{speaker}_{index}.wav`
- Speakers: jackson, nicolas, theo, yweweler, george, lucas

---

## Deliverables

1. **Training Notebook** - One per model architecture, documenting:
   - Data loading and preprocessing
   - Model definition
   - Training loop with metrics
   - Checkpointing

2. **Inference Notebook** - Convert clean audio to tape-processed output:
   - Load trained model
   - Process arbitrary audio input
   - Export result as WAV

3. **Documentation**:
   - README.md (project overview)
   - PROJECT-PLAN.md (this document)
   - ALIGNMENT-ALGORITHMS.md (technical reference)

4. **GitHub Repository** - Complete codebase with:
   - Source code in `src/`
   - Notebooks in `notebooks/`
   - Tests in `tests/`
   - Configuration in `config.json`

---

*Last updated: December 2025*
