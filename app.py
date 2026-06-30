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

# 1. Page Config MUST be the very first Streamlit command
st.set_page_config(page_title="VBCUA Platform", page_icon="🎙️", layout="wide")

def display_waveform(audio_path):
    try:
        # Load the audio file using Librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create a beautiful Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#4CAF50", alpha=0.8)
        
        # Format the graph
        ax.set(title="Acoustic Signal (Amplitude vs. Time)", xlabel="Time (seconds)", ylabel="Amplitude")
        ax.set_facecolor('#f0f2f6')
        fig.patch.set_facecolor('#f0f2f6')
        
        # Display the graph in Streamlit
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate waveform: {e}")

# 2. Main Title & Description
st.title("🎙️ Voice-Based Concept Understanding Analyser")
st.markdown("Upload your audio explaining a concept, and our AI will evaluate your understanding, transcription, and speech clarity.")

# 3. Sidebar for Inputs (The "Control Panel")
with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded_audio = st.file_uploader("1. Upload your audio", type=["wav", "mp3", "ogg"])
    reference_concept = st.text_area("2. Enter the reference concept:", height=150, placeholder="Type the concept you are explaining here...")
    analyze_button = st.button("🚀 Analyze Audio", use_container_width=True)

# 4. Main Processing Logic & Dashboard Layout
if analyze_button:
    if uploaded_audio and reference_concept:
        with st.spinner("Processing your audio with AI..."):
            # Save file with the correct extension
            file_extension = uploaded_audio.name.split(".")[-1]
            temp_filename = f"temp_audio.{file_extension}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_audio.read())
            
            # Run AI logic
            transcribed_text = transcribe_audio(temp_filename)
            score = calculate_similarity(transcribed_text, reference_concept)
            level = classify_understanding(score)
            filler_ratio = calculate_filler_ratio(transcribed_text)
            
            # Store for PDF report
            st.session_state['report_data'] = {
                'score': score, 
                'level': level, 
                'transcription': transcribed_text
            }

            # --- BEAUTIFUL RESULTS LAYOUT ---
            st.markdown("---")
            
            # Split screen into two wide columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Audio & Transcription")
                st.audio(temp_filename)
                
                st.markdown("### Audio Waveform Analysis")
                display_waveform(temp_filename)
                
                st.text_area("AI Transcription:", value=transcribed_text, height=150, disabled=True)
                
            with col2:
                st.subheader("🏆 Evaluation Results")
                
                # Big, sleek metrics side-by-side
                m1, m2 = st.columns(2)
                m1.metric(label="Final Score", value=f"{score:.2f}/1.0")
                m2.metric(label="Filler Word Ratio", value=f"{filler_ratio:.2f}")
                
                # Qualitative Feedback Boxes
                st.markdown("### Feedback Analysis")
                if score >= 0.7:
                    st.success(f"**Level: {level}**\n\n✅ Excellent! You covered the key concepts accurately.")
                elif score >= 0.4:
                    st.warning(f"**Level: {level}**\n\n⚠️ You touched on the main ideas, but your explanation needs more detail.")
                else:
                    st.error(f"**Level: {level}**\n\n❌ The explanation provided does not align with the reference concept.")

            # Cleanup temporary audio files securely
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    else:
        st.sidebar.error("Please provide both an audio file and a reference concept.")

# 5. Handle PDF Download beautifully at the bottom
if 'report_data' in st.session_state:
    st.markdown("---")
    st.subheader("📄 Formal Documentation")
    st.markdown("Generate a printable performance report of the current session.")
    
    buffer = io.BytesIO()
    generate_pdf(st.session_state['report_data'], buffer)
    
    st.download_button(
        label="⬇️ Download PDF Report",
        data=buffer.getvalue(),
        file_name="evaluation_report.pdf",
        mime="application/pdf"
    )