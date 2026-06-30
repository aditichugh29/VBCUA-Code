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

# 1. Page Config (Widescreen Dashboard)
st.set_page_config(page_title="VBCUA | AI Evaluation", page_icon="🎙️", layout="wide")

def display_waveform(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(12, 3))
        # Sleek indigo color for the waveform
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#4F46E5", alpha=0.8) 
        ax.set(title="Acoustic Signal Profile", xlabel="Time (seconds)", ylabel="Amplitude")
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        st.pyplot(fig)
    except Exception as e:
        st.warning("Waveform visualization unavailable for this file format.")

# 2. Hero Section
st.title("🎙️ Voice-Based Concept Understanding Analyser")
st.markdown("**Enterprise AI Assessment Platform** | Evaluate spoken knowledge against benchmark concepts with precision.")

# 3. Sidebar Input Control Panel
with st.sidebar:
    st.header("⚙️ Evaluation Setup")
    
    st.subheader("1. Audio Input")
    uploaded_audio = st.file_uploader("Upload student voice recording", type=["wav", "mp3", "ogg"])
    
    st.subheader("2. Benchmark Criteria")
    reference_concept = st.text_area("Ideal Reference Answer", height=180, placeholder="Paste the exact concept the student should be explaining...")
    
    st.markdown("---")
    analyze_button = st.button("🚀 Run AI Evaluation", use_container_width=True)

# 4. Processing Core & Dashboard
if analyze_button:
    if uploaded_audio and reference_concept:
        with st.status("Executing Cognitive Pipeline...", expanded=True) as status:
            st.write("🔄 Initializing audio stream...")
            file_extension = uploaded_audio.name.split(".")[-1]
            temp_filename = f"temp_audio.{file_extension}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_audio.read())
            
            st.write("🤖 Transcribing audio with AI...")
            transcribed_text = transcribe_audio(temp_filename)
            
            st.write("🧠 Analyzing semantic similarity...")
            score = calculate_similarity(transcribed_text, reference_concept)
            level = classify_understanding(score)
            filler_ratio = calculate_filler_ratio(transcribed_text)
            
            st.session_state['report_data'] = {
                'score': score, 
                'level': level, 
                'transcription': transcribed_text
            }
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # --- PREMIUM TABBED DASHBOARD ---
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📝 AI Transcription", "📄 Official Report"])

        with tab1:
            st.markdown("### Executive Overview")
            
            # KPI Metric Grid
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Conceptual Alignment", value=f"{score * 100:.1f}%", delta=f"{score:.2f} / 1.00")
            m2.metric(label="Assessed Level", value=level)
            m3.metric(label="Filler Word Ratio", value=f"{filler_ratio:.2f}")

            # Contextual Feedback
            if score >= 0.7:
                st.success("**High Proficiency:** The spoken explanation closely matches the required benchmark criteria.")
            elif score >= 0.4:
                st.warning("**Developing Proficiency:** Key concepts were mentioned, but deeper elaboration is required.")
            else:
                st.error("**Concept Gap:** The spoken explanation does not adequately cover the benchmark criteria.")

            # Interactive Audio & Waveform Expander (Bug Fixed Here!)
            with st.expander("🔍 Explore Audio Diagnostics", expanded=True):
                st.audio(uploaded_audio) 
                display_waveform(temp_filename)

        with tab2:
            st.markdown("### Decoded Speech Transcript")
            st.info("The text below represents exactly what the AI system heard and processed.")
            st.text_area("Raw AI Transcript", value=transcribed_text, height=250, disabled=True)

        with tab3:
            st.markdown("### Export Documentation")
            st.markdown("Generate a printable PDF evaluation report for your records.")
            
            buffer = io.BytesIO()
            generate_pdf(st.session_state['report_data'], buffer)
            
            st.download_button(
                label="⬇️ Download Official PDF Report",
                data=buffer.getvalue(),
                file_name="VBCUA_Evaluation_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # Safe cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    else:
        st.sidebar.error("Error: Please provide both an audio file and benchmark criteria.")

# Persist download button globally
elif 'report_data' in st.session_state:
    st.markdown("---")
    with st.expander("📥 Access Previous Session Report"):
        buffer = io.BytesIO()
        generate_pdf(st.session_state['report_data'], buffer)
        st.download_button(label="Download PDF Report", data=buffer.getvalue(), file_name="VBCUA_Evaluation_Report.pdf", mime="application/pdf")