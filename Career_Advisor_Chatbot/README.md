# 💼 Career Advisor Chatbot 

##  Overview

This project is a domain-specific AI chatbot built using the Google Gemini GenAI API.
It provides intelligent career guidance such as career paths, required skills, tools, and internship advice through a conversational interface.

The application is designed using clean and modular architecture, making it scalable and maintainable.

---

##  Key Features

*  AI-Powered Responses using Gemini API
*  Multi-Turn Conversation Memory (context-aware chatbot)
*  Advanced Prompt Engineering for structured responses
*  Modular Architecture (separation of concerns)
*  Secure API Key Management using environment variables
*  Logging System for monitoring and debugging
*  Interactive UI built with Streamlit

---

##  System Architecture

User → Streamlit UI → Controller → Prompt Manager → Gemini Client → Gemini API → Response → UI

---

## 📁 Project Structure

```
genai-chatbot/
│
├── app.py                    # Streamlit UI
├── requirements.txt
│
├── config/                  # Configuration (API keys, settings)
├── services/                # Gemini API client
├── core/                    # Prompt management
├── controllers/             # Business logic
├── utils/                   # Logging module
```

---

##  Tech Stack

* Python
* Streamlit
* Google Gemini API
* python-dotenv

---

##  Sample Use Cases

* Career guidance for students
* Skill recommendations for tech roles
* Internship preparation advice
* Career switching suggestions

---
