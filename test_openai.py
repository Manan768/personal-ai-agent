from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("Loaded key:", api_key[:10] + "...")

client = OpenAI(api_key=api_key)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}]
)

print("Response:", resp.choices[0].message.content)
