from dotenv import load_dotenv
import os
from openai import OpenAI

# .env laden
load_dotenv()

# API-Key auslesen
api_key = os.getenv("OPENAI_API_KEY")

# Debug (optional)
print("API KEY:", api_key)

# Client erstellen
client = OpenAI(api_key=api_key)

# Anfrage senden
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Sag Hallo"
)

# Ausgabe
print(response.output_text)