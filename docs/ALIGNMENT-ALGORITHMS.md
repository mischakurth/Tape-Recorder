# Audio Alignment Algorithms

This document describes the algorithms used to align tape recordings with their original input files. This is a critical step in preparing training data for the ML model.

## The Problem

When audio is recorded to tape and played back:
1. The tape recording contains the same audio segments as the input
2. But the timing is slightly different (tape speed variations)
3. We need to find exact segment boundaries to create matched before/after pairs

## Data Structure

The input audio is compiled with markers:
- **Spoken digits** at the start of each batch (e.g., "zero zero one")
- **Beep sequences** (3 beeps with silence) as delimiters between segments
- **Silence gaps** (0.5-1s) surrounding each delimiter

```
[silence][beep][silence][beep][silence][beep][silence][AUDIO SEGMENT][silence][beep]...
```

## Core Algorithms

### 1. Silence Detection

Detect silent vs non-silent regions using a dBFS threshold.

```python
from pydub import AudioSegment

def is_silent_chunk(audio_chunk: AudioSegment, threshold_dbfs: float) -> bool:
    """Check if an audio chunk is silent.

    Args:
        audio_chunk: Audio segment to check
        threshold_dbfs: Maximum dBFS level considered silent (e.g., -50.0)

    Returns:
        True if the chunk's peak level is below threshold
    """
    return audio_chunk.max_dBFS <= threshold_dbfs
```

### 2. Chunk Boundary Detection

Split audio into non-silent regions by scanning with a sliding window.

```python
def detect_audio_chunks(
    audio_sample: AudioSegment,
    threshold_dbfs: float,
    chunk_duration_ms: int
) -> list[list[int]]:
    """Detect continuous non-silent audio regions.

    Scans audio in chunks, identifies silent gaps, and returns
    boundaries of non-silent regions.

    Args:
        audio_sample: Full audio to analyze
        threshold_dbfs: Silence threshold (e.g., -50.0 dBFS)
        chunk_duration_ms: Scanning window size (e.g., 50ms)

    Returns:
        List of [chunk_start_idx, chunk_end_idx, start_ms, end_ms]
        for each non-silent region
    """
    result = []
    chunk_nr = 0
    duration_ms = len(audio_sample)

    index = 0
    while index + chunk_duration_ms <= duration_ms:
        chunk_nr += 1
        audio_chunk = audio_sample[index:index + chunk_duration_ms]

        if is_silent_chunk(audio_chunk, threshold_dbfs):
            # Silent chunk - refine the boundary of the previous region
            if result:
                start, end = refine_boundaries(
                    audio_sample,
                    result[-1][2],
                    result[-1][3],
                    threshold_dbfs
                )
                result[-1][2] = start
                result[-1][3] = end
        else:
            # Non-silent chunk
            if result and result[-1][1] == chunk_nr - 1:
                # Extend previous region
                result[-1][1] = chunk_nr
                result[-1][3] = index + chunk_duration_ms
            else:
                # Start new region
                result.append([chunk_nr, chunk_nr, index, index + chunk_duration_ms])

        index += chunk_duration_ms

    return result
```

### 3. Boundary Refinement

Refine chunk boundaries to millisecond precision using iterative narrowing.

```python
def refine_boundaries(
    audio_sample: AudioSegment,
    start: int,
    end: int,
    threshold_dbfs: float
) -> tuple[int, int]:
    """Refine boundaries to find exact start/end of non-silent audio.

    Uses 1ms steps to find precise boundaries where audio
    transitions from silent to non-silent.

    Args:
        audio_sample: Full audio
        start: Initial start boundary (ms)
        end: Initial end boundary (ms)
        threshold_dbfs: Silence threshold

    Returns:
        Tuple of (refined_start, refined_end) in milliseconds
    """
    width = 1  # 1ms precision

    # Find left boundary (first non-silent sample)
    left = start
    while left < end:
        if audio_sample[left:left + width].max_dBFS > threshold_dbfs:
            break
        left += 1

    # Find right boundary (last non-silent sample)
    right = end
    while right > start:
        if audio_sample[right - width:right].max_dBFS > threshold_dbfs:
            break
        right -= 1

    return left, right
```

### 4. Before/After Alignment

Apply boundaries detected in one file to extract matching segments from another.

```python
def align_and_extract_chunks(
    before_audio: AudioSegment,
    after_audio: AudioSegment,
    threshold_dbfs: float,
    chunk_duration_ms: int,
    output_dir_before: str,
    output_dir_after: str
):
    """Detect boundaries in 'before' audio and extract matching chunks from both.

    The key insight: we detect boundaries using the AFTER (tape) recording
    because that's where we need precise alignment, then use those same
    boundaries to extract from both files.

    Args:
        before_audio: Original input audio
        after_audio: Tape-recorded output audio
        threshold_dbfs: Silence detection threshold
        chunk_duration_ms: Detection window size
        output_dir_before: Where to save 'before' chunks
        output_dir_after: Where to save 'after' chunks
    """
    # Detect chunks in the before audio
    chunks = detect_audio_chunks(before_audio, threshold_dbfs, chunk_duration_ms)

    # Fine-tune boundaries using the after audio
    # (tape recording may have slightly different silence characteristics)
    for chunk in chunks:
        start, end = chunk[2], chunk[3]
        # Expand slightly then contract to find precise boundary
        end += chunk_duration_ms
        _, refined_end = refine_boundaries(after_audio, start, end, threshold_dbfs)
        chunk[3] = refined_end

    # Extract matching segments from both files
    for i, chunk in enumerate(chunks):
        start_ms, end_ms = chunk[2], chunk[3]

        before_chunk = before_audio[start_ms:end_ms]
        after_chunk = after_audio[start_ms:end_ms]

        before_chunk.export(f"{output_dir_before}/chunk_{i:06d}.wav", format="wav")
        after_chunk.export(f"{output_dir_after}/chunk_{i:06d}.wav", format="wav")
```

## Configuration Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `threshold_dbfs` | -50.0 | dBFS level below which audio is considered silent |
| `chunk_duration_ms` | 50 | Window size for scanning (smaller = more precise, slower) |
| `silence_gap_ms` | 500-1000 | Expected silence duration between segments |

## Algorithm Flow

```
1. Load before (original) and after (tape) audio files
                    │
                    ▼
2. Scan 'before' audio in chunks to detect non-silent regions
   - Use sliding window of chunk_duration_ms
   - Mark regions where max_dBFS > threshold
                    │
                    ▼
3. Refine boundaries to millisecond precision
   - For each region, narrow down start/end points
   - Use 1ms steps to find exact transitions
                    │
                    ▼
4. Apply boundaries to 'after' audio
   - Fine-tune using after audio's silence characteristics
   - Expand then contract to handle tape timing variations
                    │
                    ▼
5. Extract matched chunks from both files
   - Same time boundaries → matched training pairs
   - Export as individual WAV files
```

## Handling Tape Speed Variations

Tape recorders introduce timing variations:
- **Wow**: Slow speed variations (< 4 Hz)
- **Flutter**: Fast speed variations (4-20 Hz)

To handle this:
1. Use the delimiter beeps as sync points
2. Align each segment independently rather than assuming constant offset
3. The silence detection naturally handles small timing shifts

## Future Improvements

Ideas from the original research notes:

1. **Beep Detection**: Train a classifier to detect delimiter beeps
   - Use beep recordings as positive examples
   - Use other audio as negative examples
   - This would be more robust than silence-only detection

2. **Speed Measurement**: Record periodic reference tones
   - Use 16th note clicks or similar
   - Measure timing drift over tape length
   - Could model and compensate for speed variations

3. **Cross-correlation**: Use signal correlation to find optimal alignment
   - Compare before/after waveforms
   - Find offset that maximizes correlation
