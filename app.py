import streamlit as st
import io
from speech_to_text import transcribe_audio
from semantic_eval import calculate_similarity
from scoring_engine import classify_understanding, calculate_filler_ratio
from report_generator import generate_pdf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def display_waveform(audio_path):
    st.write("### Audio Waveform Analysis")
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

st.set_page_config(page_title="VBCUA Platform", layout="wide")

st.title("🎙️ Voice-Based Concept Understanding Analyser")
uploaded_audio = st.file_uploader("Upload your audio", type=["wav", "mp3", "ogg"])
reference_concept = st.text_area("Enter the reference concept:")

if st.button("Analyze Audio"):
    if uploaded_audio and reference_concept:
        # Save file with the correct extension
        file_extension = uploaded_audio.name.split(".")[-1]
        temp_filename = f"temp_audio.{file_extension}"
        with open(temp_filename, "wb") as f:
         f.write(uploaded_audio.read())
        
        st.audio(temp_filename)
        display_waveform(temp_filename)
        
        # Run AI logic
        st.info("Transcribing and Evaluating...")
        transcribed_text = transcribe_audio(temp_filename)
        score = calculate_similarity(transcribed_text, reference_concept)
        
        # Display Results
        st.subheader("AI Transcription:")
        st.write(transcribed_text) # Displays what the AI heard
        
        st.subheader("Evaluation Results:")
        st.write(f"### Final Score: {score:.2f}")
        level = classify_understanding(score)
        st.write(f"**Level:** {level}")
        
        # Qualitative Feedback logic
        if score >= 0.7:
            st.success("✅ Excellent! You covered the key concepts accurately.")
        elif score >= 0.4:
            st.warning("⚠️ You touched on the main ideas, but your explanation needs more detail.")
        else:
            st.error("❌ The explanation provided does not align with the reference concept.")
        
        filler_ratio = calculate_filler_ratio(transcribed_text)
        st.metric("Filler Word Ratio", f"{filler_ratio:.2f}")
        
        # Store for PDF (including transcription for the report)
        st.session_state['report_data'] = {
            'score': score, 
            'level': level, 
            'transcription': transcribed_text
        }
    else:
        st.error("Please provide both an audio file and a reference concept.")
# Handle PDF Download correctly
if 'report_data' in st.session_state:
    buffer = io.BytesIO()
    generate_pdf(st.session_state['report_data'], buffer)
    
    st.download_button(
        label="Download PDF Report",
        data=buffer.getvalue(),
        file_name="evaluation_report.pdf",
        mime="application/pdf"
    )