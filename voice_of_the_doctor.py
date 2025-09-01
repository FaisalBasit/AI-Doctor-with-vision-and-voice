import os
import subprocess
from gtts import gTTS

def text_to_speech_with_gtts(input_text, output_filepath):
    language = "en"

    # Convert text to speech and save as mp3
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_filepath)

    try:
        # Play audio using ffplay (FFmpeg tool)
        subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath])
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")
    
    return output_filepath


# Example usage
# input_text = "Hi this is AI with Hassan, autoplay using ffplay test!"
# text_to_speech_with_gtts(input_text=input_text, output_filepath="gtts_testing_autoplay.mp3")
