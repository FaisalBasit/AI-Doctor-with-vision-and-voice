from dotenv import load_dotenv
load_dotenv()

import os
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import subprocess
from groq import Groq

# ================== CONFIGURATION ==================

# Configure FFmpeg path (Update this according to your local path)
FFMPEG_PATH = r"C:\ffmpeg\bin"
AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")

# Update system PATH for FFmpeg
os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

# Set your audio output path
AUDIO_FILE_PATH = "patient_voice_test.mp3"

# Your Whisper model
STT_MODEL = "whisper-large-v3"

# Get your Groq API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ================== FUNCTIONS ==================

def verify_ffmpeg():
    """Verify FFmpeg installation"""
    try:
        ffmpeg_path = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("✅ FFmpeg is installed and accessible.")
            return True
        logging.error("❌ FFmpeg is not found or misconfigured.")
        return False
    except Exception as e:
        logging.error(f"Error verifying FFmpeg: {e}")
        return False


def record_audio(file_path, timeout=60, phrase_time_limit=None):
    """Record audio from microphone and save as MP3."""
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("🎙️ Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("🗣️ Speak now... Recording will stop after a pause or the time limit.")
            
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("✅ Recording complete.")

            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"💾 Audio saved to: {file_path}")
    except Exception as e:
        logging.error(f"🎤 Recording failed: {e}")


def transcribe_with_groq(model_name, file_path, api_key):
    """Send MP3 to Groq's Whisper model for transcription."""
    try:
        client = Groq(api_key=api_key)
        with open(file_path, "rb") as audio_file:
            logging.info("📤 Sending audio to Groq for transcription...")
            transcription = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                language="en"
            )
        logging.info("📝 Transcription received successfully.")
        return transcription.text
    except Exception as e:
        logging.error(f"❌ Transcription failed: {e}")
        return None


# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    if not GROQ_API_KEY:
        logging.error("⚠️ GROQ_API_KEY not found in environment. Set it before running the script.")
        exit()

    if verify_ffmpeg():
        record_audio(file_path=AUDIO_FILE_PATH)
        transcription = transcribe_with_groq(STT_MODEL, AUDIO_FILE_PATH, GROQ_API_KEY)
        
        if transcription:
            print("\n🗒️ Transcribed Text:\n" + "-"*40)
            print(transcription)
            print("-"*40)
