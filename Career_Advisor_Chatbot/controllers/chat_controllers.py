from services.gemini_client import GeminiClient
from core.prompt_manager import get_system_prompt
from utils.logger import setup_logger

logger = setup_logger()

class ChatController:
    def __init__(self):
        logger.info("Initializing Chat Controller")

        self.client = GeminiClient()
        self.system_prompt = get_system_prompt()
        self.chat_session = self.client.create_chat(self.system_prompt)

    def get_reply(self, user_input):
        logger.info("Processing user request")
        return self.client.send_message(self.chat_session, user_input)
