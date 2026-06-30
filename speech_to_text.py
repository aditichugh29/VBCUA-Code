import whisper

def transcribe_audio(file_path):
    # Load the small model for faster processing
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]