import logging
import os

# Create logs folder if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

def setup_logger():
    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    return logging.getLogger("GenAI-Chatbot")
