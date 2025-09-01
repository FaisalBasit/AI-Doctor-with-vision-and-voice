# gradio_app.py

# 1. Environment Setup:
# It loads environment variables from a .env file using load_dotenv() (used for API keys like GROQ).
from dotenv import load_dotenv
load_dotenv()

# 2. Imports Custom Functions:
# • From brain_of_the_doctor: encode_image() and analyze_image_with_query() handle image processing and AI analysis.
# • From voice_of_the_patient: record_audio() and transcribe_with_groq() are used to record and convert speech to text.
# • From voice_of_the_doctor: text_to_speech_with_gtts() converts AI response into voice.
import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

# 3. Prompt Creation:
# A system_prompt is defined to instruct the AI model to behave like a human doctor.
system_prompt = """You are to act like a professional doctor reviewing medical images for educational purposes. 
Analyze the image and respond directly, concisely, and like you're speaking to a real patient. 
First, clearly state the name of the condition you think is present (e.g., no tumor detected, glioma detected, meningioma detected, pituitary tumor detected). 
Then explain your impression in a natural, human tone. Avoid using phrases like "In the image I see"—instead, begin with "With what I see, I think you have...". 
If appropriate, include possible differentials and brief suggestions for treatment or further evaluation. Do not use any numbers, bullet points, or special characters.
 Keep the response to no more than two sentences and write it as a single paragraph without any markdown or AI disclaimers.
"""

# 4. process_inputs() Function:
def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(
        model_name="whisper-large-v3",
        file_path=audio_filepath,
        api_key =os.environ.get("GROQ_API_KEY")
    )

    if image_filepath:
        encoded_img = encode_image(image_filepath)
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encoded_img,
            model="meta-llama/llama-4-maverick-17b-128e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    voice_of_doctor = text_to_speech_with_gtts(
        input_text=doctor_response,
        output_filepath="final.mp3"
    )

    return speech_to_text_output, doctor_response, voice_of_doctor

# 5. Gradio UI:
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Voice Output")
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)

# Visit the app at: http://127.0.0.1:7860
