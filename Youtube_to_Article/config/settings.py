import os
from dotenv import load_dotenv

load_dotenv()

# Load API key safely
os.environ['GOOGLE_API_KEY'] = os.getenv("Youtube_API")
