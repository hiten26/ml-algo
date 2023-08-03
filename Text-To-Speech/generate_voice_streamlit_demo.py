import streamlit as st
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import io, os
import tempfile

st.title("Text-to-Speech App")

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

text_input = st.text_area("Enter the text:", "Hi, My name is Hitendra and I am a data scientist with 7 years of experience in ML/AI")
preset = st.selectbox("Select a voice preset:", ["v2/en_speaker_9", "v2/en_speaker_0"])

if st.button("Generate Audio"):
    inputs = processor(text_input, voice_preset=preset)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate

    # Create a temporary WAV file to save the generated audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filepath = temp_file.name
        sf.write(temp_filepath, audio_array, sample_rate)
    
    # Play the generated audio in Streamlit
    audio_bytes = open(temp_filepath, "rb").read()
    st.audio(audio_bytes, format="audio/wav")

    # Remove the temporary file
    os.remove(temp_filepath)

st.text("Output audio will be played above.")
