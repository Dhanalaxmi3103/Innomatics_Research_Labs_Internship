import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("Gemini_API")
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7
