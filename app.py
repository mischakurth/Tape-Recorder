import streamlit as st
import torch
import soundfile as sf
import numpy as np
import io
import os
import gc
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Importiere deine Projekt-Module
from src.model import AudioUNet
from src.audio_processor import AudioProcessor

# --- KONFIGURATION ---
MODEL_PATH = "models/berta_best.pth"
CHUNK_WIDTH = 256
TARGET_HEIGHT = 1040
WARN_DURATION = 60.0

# Layout auf 'wide' setzen
st.set_page_config(page_title="Tape Recorder Simulator", page_icon="📼", layout="wide")


@st.cache_resource
def load_engine():
    """
    Lädt Modell und Prozessor und hält sie im RAM.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # print(f"Lade Engine auf: {device}") # Optional für Log

    proc = AudioProcessor(sample_rate=96000)
    model = AudioUNet(n_channels=4, n_classes=4).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        except Exception as e:
            st.error(f"Fehler beim Laden der Gewichte: {e}")
            return None, None, None
    else:
        st.error(f"❌ Modell nicht gefunden unter: {MODEL_PATH}")
        return None, None, None

    return model, proc, device


def process_audio_pipeline(uploaded_file, model, proc, device):
    """
    Die komplette Pipeline: Audio -> STFT -> Chunking -> Model -> Stitching -> iSTFT -> Audio
    """
    # Audio Laden
    data, sr = sf.read(uploaded_file)

    # Dimensionen prüfen
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T

    y_tensor = torch.from_numpy(data).float()

    # Resampling
    if sr != proc.sample_rate:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, proc.sample_rate)
        y_tensor = resampler(y_tensor)

    # Mono aufblasen
    if y_tensor.shape[0] == 1:
        y_tensor = y_tensor.repeat(2, 1)

    y_np = y_tensor.numpy()

    # --- Pipeline ---
    # STFT
    stft_complex = proc.audio_to_stft(y_np)
    cnn_input = proc.stft_to_cnn_input(stft_complex)

    # Padding
    c, h, w = cnn_input.shape
    original_height = h
    pad_h = TARGET_HEIGHT - h
    if pad_h > 0:
        cnn_input = np.pad(cnn_input, ((0, 0), (0, pad_h), (0, 0)), mode='constant')

    # Inference
    output_list = []
    total_frames = cnn_input.shape[2]
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Simuliere Bandlauf...")

    with torch.no_grad():
        for start_idx in range(0, total_frames, CHUNK_WIDTH):
            end_idx = min(start_idx + CHUNK_WIDTH, total_frames)

            chunk = cnn_input[:, :, start_idx:end_idx]

            # Zeit-Padding
            current_w = chunk.shape[2]
            pad_w = CHUNK_WIDTH - current_w
            if pad_w > 0:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad_w)), mode='constant')

            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
            pred_tensor = model(chunk_tensor)
            pred_chunk = pred_tensor.cpu().numpy().squeeze(0)

            # Padding entfernen
            if pad_w > 0:
                pred_chunk = pred_chunk[:, :, :current_w]

            output_list.append(pred_chunk)
            progress_bar.progress(min(end_idx / total_frames, 1.0))

    cnn_output = np.concatenate(output_list, axis=2)

    # Un-Padding
    if pad_h > 0:
        cnn_output = cnn_output[:, :original_height, :]

    # iSTFT
    status_text.text("Rekonstruiere Audio...")
    stft_recon = proc.cnn_input_to_stft(cnn_output)
    y_recon = proc.stft_to_audio(stft_recon, original_length=None)

    # Cleanup
    status_text.empty()
    progress_bar.empty()
    del cnn_input, cnn_output, output_list
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    return y_recon, proc.sample_rate


def plot_advanced_analysis(clean_audio, taped_audio, sr):
    """
    Erstellt eine Heatmap der Veränderungen und einen Frequenzgang.
    """
    min_len = min(len(clean_audio), len(taped_audio))
    clean_cut = clean_audio[:min_len]
    taped_cut = taped_audio[:min_len]

    S_clean = np.abs(librosa.stft(clean_cut, n_fft=2048))
    S_tape = np.abs(librosa.stft(taped_cut, n_fft=2048))

    D_clean = librosa.amplitude_to_db(S_clean, ref=np.max)
    D_tape = librosa.amplitude_to_db(S_tape, ref=np.max)

    D_delta = D_tape - D_clean

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1)

    # 1. Heatmap
    ax1 = fig.add_subplot(gs[0])
    img1 = librosa.display.specshow(D_delta, sr=sr, x_axis='time', y_axis='hz', ax=ax1, cmap='bwr', vmin=-15, vmax=15)
    ax1.set_title("Heatmap der Veränderung (Rot = Hinzugefügt, Blau = Entfernt)")
    fig.colorbar(img1, ax=ax1, format="%+2.0f dB")

    # 2. Frequenzgang
    ax2 = fig.add_subplot(gs[1])
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean_clean = np.mean(S_clean, axis=1)
    mean_tape = np.mean(S_tape, axis=1)

    mean_clean_db = librosa.amplitude_to_db(mean_clean, ref=np.max)
    mean_tape_db = librosa.amplitude_to_db(mean_tape, ref=np.max)

    ax2.plot(freqs, mean_clean_db, label='Digital Input', color='gray', alpha=0.6, linestyle='--')
    ax2.plot(freqs, mean_tape_db, label='Tape Simulation', color='#ff4b4b', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlim(20, sr / 2)
    ax2.set_ylabel("Amplitude (dB)")
    ax2.set_xlabel("Frequenz (Hz)")
    ax2.set_title("Durchschnittlicher Frequenzgang (EQ-Kurve)")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # 3. Spektrogramm
    ax3 = fig.add_subplot(gs[2])
    img3 = librosa.display.specshow(D_tape, sr=sr, x_axis='time', y_axis='hz', ax=ax3, cmap='magma')
    ax3.set_title("Spektrogramm (Tape Output Resultat)")
    fig.colorbar(img3, ax=ax3, format="%+2.0f dB")

    plt.tight_layout()
    return fig


def main():
    st.title("📼 Tape Recorder Simulator")
    st.markdown("""
    Lade eine saubere, digitale `.wav` Datei hoch. 
    Die KI simuliert den analogen Charakter eines Bandgerätes (Sättigung, Rauschen, EQ).
    """)

    model, proc, device = load_engine()
    if model is None: st.stop()

    uploaded_file = st.file_uploader("Wähle eine WAV-Datei", type=["wav"])

    # --- STATE MANAGEMENT ---
    # Wenn ein neues File hochgeladen wird, löschen wir das alte Ergebnis aus dem Cache
    if uploaded_file:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if "last_file_id" not in st.session_state or st.session_state.last_file_id != file_id:
            # Reset, da neues File
            st.session_state.last_file_id = file_id
            st.session_state.processed_audio = None
            st.session_state.processed_sr = None
    else:
        # Kein File -> Alles resetten
        st.session_state.last_file_id = None
        st.session_state.processed_audio = None

    if uploaded_file is not None:
        # Input anzeigen
        uploaded_file.seek(0)
        data, sr = sf.read(uploaded_file)

        if data.ndim > 1:
            orig_mono = data[:, 0]
        else:
            orig_mono = data

        duration = len(orig_mono) / sr
        st.info(f"Input: {duration:.2f}s | {sr} Hz")

        st.subheader("1. Digitaler Input")
        uploaded_file.seek(0)
        st.audio(uploaded_file)

        # --- PROCESS BUTTON ---
        # Dieser Button triggert nur die Berechnung und speichert in session_state
        if st.button("🔴 Auf Tape aufnehmen (Simulieren)"):
            with st.spinner("Simuliere Bandlauf..."):
                try:
                    uploaded_file.seek(0)
                    taped_audio, out_sr = process_audio_pipeline(uploaded_file, model, proc, device)

                    # Ergebnis im State speichern!
                    st.session_state.processed_audio = taped_audio
                    st.session_state.processed_sr = out_sr
                    st.success("Aufnahme beendet!")

                except Exception as e:
                    st.error(f"Fehler: {e}")
                    st.exception(e)

        # --- RESULTAT ANZEIGE ---
        # Wir prüfen, ob ein Ergebnis im State liegt. Wenn ja, zeigen wir es an.
        # Das bleibt auch wahr, wenn der Download-Button geklickt wird (Script Rerun).
        if st.session_state.get("processed_audio") is not None:
            taped_audio = st.session_state.processed_audio
            out_sr = st.session_state.processed_sr

            # Buffer erstellen für Player & Download
            buffer_tape = io.BytesIO()
            sf.write(buffer_tape, taped_audio.T, out_sr, format='WAV')
            buffer_tape.seek(0)

            # --- A/B Vergleich ---
            st.divider()
            st.subheader("2. A/B Vergleich")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                uploaded_file.seek(0)
                st.audio(uploaded_file)
            with c2:
                st.markdown("**Tape Simulation**")
                st.audio(buffer_tape, format='audio/wav')

            # --- Analyse ---
            st.divider()
            st.subheader("3. Technische Analyse")

            tape_mono = taped_audio[0, :] if taped_audio.ndim > 1 else taped_audio

            # Plot erstellen (wird bei jedem Rerun neu generiert, geht aber schnell genug)
            fig = plot_advanced_analysis(orig_mono, tape_mono, sr)
            st.pyplot(fig)

            # --- Download ---
            st.divider()
            st.download_button(
                label="💾 Tape-Aufnahme speichern",
                data=buffer_tape,
                file_name="tape_simulation.wav",
                mime="audio/wav"
            )


if __name__ == "__main__":
    main()