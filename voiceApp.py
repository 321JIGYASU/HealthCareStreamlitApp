import streamlit as st
import wave
import os
import json
import re


from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import fuzz

SYMPTOMS = [
    ["fever", "high temperature"],
    ["cough"],
    ["chest pain", "heart pain", "chest discomfort","pain in chest","pain in heart"],
    ["breathing difficulty", "shortness of breath", "dyspnea"],
    ["headache"],
    ["fatigue", "tiredness"],
    ["vomiting","nausea"]
]

MEDICAL_HISTORY = [
    ["heart problem", "heart disease", "cardiac issue", "heart condition"],
    ["diabetes", "high sugar"],
    ["hypertension", "high bp", "blood pressure"],
    ["asthma"]
]

DURATION_PATTERNS = [
    r"\b\d+\s*(days|day|weeks|week|months|month|years|year)\b",
    r"\bsince\s+\d+\s*(days|weeks|months|years)\b",
    r"\bfor\s+\d+\s*(days|weeks|months|years)\b"
]


def match_grps(text,variants,threshold=80):
     """Match synonym groups instead of exact words"""
     for v in variants:
          score = fuzz.partial_ratio(v.lower(),text)
          if score>=threshold:
               return True
     return False

def extract_symptoms(text):

    text_lower = text.lower()

    found_symp = []

    found_history = []

    for group in SYMPTOMS:
         canonical = group[0]

         if match_grps(text,group):
              found_symp.append(canonical)
              
    for group in MEDICAL_HISTORY:
          canonical = group[0]

          if match_grps(text,group):
               found_history.append(canonical)
               


    duration = None
    for pattern in DURATION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            duration = match.group()
            break

    main_complaint = found_symp[0] if found_symp else None

    return {
        "main_complaint": main_complaint,
        "symptoms": list(set(found_symp)),
        "duration": duration,
        "medical_history": list(set(found_history))
    }
        


@st.cache_resource
def load_model():
     return WhisperModel("medium", device="cpu", compute_type="int8")
model = load_model()
     
    

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
    clinical_data_extract = extract_symptoms(english_text)

    # Step 3: optional back-translation (for UI clarity)
    native_script = GoogleTranslator(
        source="en",
        target=detected_lang
    ).translate(english_text)

    return detected_lang, native_text, english_text, native_script, clinical_data_extract


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
        lang, native_text, english_text, native_script,clinical_data_extract = process_audio("patient_audio.wav")
        st.subheader("Detected language")
        st.write(lang)
        st.subheader("Patient Speech (Original)")
        st.write(native_text)
        st.subheader("English Translation")
        st.write(english_text)
        st.subheader("Back to Native (Cleaned Version)")
        st.write(native_script)
        st.subheader("Summary Extract")
        st.write(clinical_data_extract)
    
       
    

    # Delete button
    if st.button("Delete Recording"):
        os.remove("patient_audio.wav")
        st.success("Recording deleted")
        st.rerun()

