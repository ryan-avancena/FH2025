import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

# debugging
# print(OPEN_AI_KEY)

client = OpenAI(api_key=os.getenv(OPEN_AI_KEY))

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        ],
    }],
)

print(response.output_text)