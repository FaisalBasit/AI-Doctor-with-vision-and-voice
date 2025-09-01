# Step 1: Setup GROQ API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# Step 2: Convert image to required format
import base64

#image_path = "acne.jpg"
def encode_image(image_path):
    image_file = open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')



# Step 3: Setup Multimodal LLM
from groq import Groq

query = "Is there something wrong with my face?"  # Replace with your actual query
model = "meta-llama/llama-4-maverick-17b-128e-instruct"

def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=GROQ_API_KEY)  # Pass the API key directly
    

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return(chat_completion.choices[0].message.content)  # Print the response text
