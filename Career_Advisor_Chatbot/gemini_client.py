from google import genai
import os
from config.settings import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE
from utils.logger import setup_logger

logger = setup_logger()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

class GeminiClient:
    def __init__(self):
        try:
            self.client = genai.Client()
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Client initialization failed: {str(e)}")

    def create_chat(self, system_prompt):
        try:
            chat = self.client.chats.create(
                model=MODEL_NAME,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE
                )
            )
            logger.info("Chat session created")
            return chat
        except Exception as e:
            logger.error(f"Chat creation failed: {str(e)}")

    def send_message(self, chat_session, message):
        try:
            logger.info(f"User Input: {message}")

            response = chat_session.send_message(message)

            reply = response.text if response else "No response"
            
            logger.info(f"Bot Response: {reply}")

            return reply

        except Exception as e:
            logger.error(f"Message sending failed: {str(e)}")
            return "Error occurred. Please try again later"