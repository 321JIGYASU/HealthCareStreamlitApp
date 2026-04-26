import streamlit as st
import wave
import os


from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

model = WhisperModel("small", device="cpu", compute_type="int8_float32")

def process_audio(audio_path):

    # Step 1: detect language + get native transcript
    segments, info = model.transcribe(audio_path)
    detected_lang = info.language

    native_text = " ".join([s.text for s in segments])

    # Step 2: translate to English using Whisper
    segments_en, _ = model.transcribe(
        audio_path,
        task="translate",
        language=detected_lang,
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False
    )

    english_text = " ".join([s.text for s in segments_en])

    # Step 3: optional back-translation (for UI clarity)
    native_script = GoogleTranslator(
        source="en",
        target=detected_lang
    ).translate(english_text)

    return detected_lang, native_text, english_text, native_script


st.title("Patient voice complaint")

audio_file = st.audio_input("Record patient's complaint")

# Save recording
if audio_file is not None:
    if st.button("Save Recording"):
        with open("patient_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())
        st.success("Recording saved")

# Show saved audio
if os.path.exists("patient_audio.wav"):
    st.audio("patient_audio.wav")

    with wave.open("patient_audio.wav", "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / sr

    st.write(f"Sample rate: {sr}")
    st.write(f"Audio duration: {duration:.2f} seconds")

    #Transcribe button
    if st.button("Transcribe & Translate"):
        lang, native_text, english_text, native_script = process_audio("patient_audio.wav")
        st.subheader("Detected language")
        st.write(lang)
        st.subheader("Patient Speech (Original)")
        st.write(native_text)
        st.subheader("English Translation")
        st.write(english_text)
        st.subheader("Back to Native (Cleaned Version)")
        st.write(native_script) 
    
       
    

    # Delete button
    if st.button("Delete Recording"):
        os.remove("patient_audio.wav")
        st.success("Recording deleted")
        st.rerun()

