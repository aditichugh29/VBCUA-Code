import streamlit as st
import io
import os
from speech_to_text import transcribe_audio
from semantic_eval import calculate_similarity
from scoring_engine import classify_understanding, calculate_filler_ratio
from report_generator import generate_pdf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Page Config (Enables widescreen mode)
st.set_page_config(page_title="VBCUA Enterprise", page_icon="🎙️", layout="wide")

def display_waveform(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#2E7D32", alpha=0.7)
        ax.set(title="Acoustic Amplitude Profile", xlabel="Time (seconds)", ylabel="Amplitude")
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('#fafafa')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Waveform Generation Suspended: {e}")

# 2. Hero Section Header
st.title("🎙️ Voice-Based Concept Understanding Analyser")
st.caption("Enterprise AI Assessment Platform | Audio Signal Processing & Semantic Evaluation")

# 3. Sidebar Input Control Panel
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    st.subheader("1. Data Ingestion")
    uploaded_audio = st.file_uploader("Upload Target Audio File", type=["wav", "mp3", "ogg"])
    
    st.subheader("2. Ground Truth Target")
    reference_concept = st.text_area("Reference Concept Script", height=180, placeholder="Input key benchmark evaluation concepts...")
    
    st.markdown("---")
    analyze_button = st.button("🚀 Execute AI Pipeline", use_container_width=True)

# 4. Processing Core & Tabbed Display
if analyze_button:
    if uploaded_audio and reference_concept:
        with st.status("Executing Cognitive Pipeline...", expanded=True) as status:
            st.write("🔄 Initializing audio stream extraction...")
            file_extension = uploaded_audio.name.split(".")[-1]
            temp_filename = f"temp_audio.{file_extension}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_audio.read())
            
            st.write("🤖 Transcribing vocal frequencies (OpenAI Whisper Engine)...")
            transcribed_text = transcribe_audio(temp_filename)
            
            st.write("🧠 Executing semantic mapping vectors (S-BERT Core)...")
            score = calculate_similarity(transcribed_text, reference_concept)
            level = classify_understanding(score)
            filler_ratio = calculate_filler_ratio(transcribed_text)
            
            st.session_state['report_data'] = {
                'score': score, 
                'level': level, 
                'transcription': transcribed_text
            }
            status.update(label="Analysis Pipeline Complete!", state="complete", expanded=False)

        # --- THE PREMIERE TAB LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["📊 Performance Analytics", "📝 Transcription Matrix", "📋 Generation Report"])

        with tab1:
            st.markdown("### Executive Dashboard Overview")
            
            # KPI Metric Grid
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Conceptual Alignment Score", value=f"{score * 100:.1f}%", delta=f"{score:.2f} / 1.00")
            m2.metric(label="Fluency Accuracy Index", value=level)
            m3.metric(label="Filler Velocity Ratio", value=f"{filler_ratio:.2f}")

            # Status Message
            if score >= 0.7:
                st.success(f"**Optimal Alignment Detected:** Explanatory proficiency aligns systematically with core reference structures.")
            elif score >= 0.4:
                st.warning(f"**Partial Alignment Detected:** Narrative vectors intersect core values but require granular elaboration.")
            else:
                st.error(f"**Structural Gap Detected:** Evaluated audio file demonstrates low convergence with standard requirements.")

            # Waveform inside a clean, interactive drop-down expander
            with st.expander("🔍 View Acoustic Structural Waveform Analysis", expanded=True):
               st.audio(uploaded_audio)
            display_waveform(uploaded_audio)

        with tab2:
            st.markdown("### AI Derived Transcription Matrix")
            st.info("The text below indicates the raw speech-to-text outputs processed through high-fidelity lexical modeling.")
            st.text_area("Decoded Audio Output String", value=transcribed_text, height=250, disabled=True)

        with tab3:
            st.markdown("### Compliance & Verification Export")
            st.markdown("Generate and compile immutable evaluation transcripts for submission documentation arrays.")
            
            buffer = io.BytesIO()
            generate_pdf(st.session_state['report_data'], buffer)
            
            st.download_button(
                label="⬇️ Export Official PDF Documentation",
                data=buffer.getvalue(),
                file_name="VBCUA_Evaluation_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # Secure workspace cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    else:
        st.sidebar.error("Error: Missing mandatory runtime parameters.")

# Persist download capability if context states are active
elif 'report_data' in st.session_state:
    st.markdown("---")
    with st.expander("📥 Download Report from Previous Active Session"):
        buffer = io.BytesIO()
        generate_pdf(st.session_state['report_data'], buffer)
        st.download_button(label="Download PDF Report File", data=buffer.getvalue(), file_name="VBCUA_Evaluation_Report.pdf", mime="application/pdf")