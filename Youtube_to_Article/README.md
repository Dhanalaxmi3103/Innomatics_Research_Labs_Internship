# 🎥 YouTube to Article AI

An AI-powered application that converts YouTube videos into structured, readable article-style summaries using LangChain and Streamlit.

---

## 🚀 Overview

This project allows users to simply paste a YouTube URL and get:
- 📄 Clean transcript extraction  
- 🧠 AI-generated summary  
- 📰 Article-style formatted content  

It is designed to handle real-world issues like **blocked transcripts and missing captions** using a robust fallback mechanism.

---

## ✨ Features

- 🔗 Input: Paste any YouTube video link  
- 📜 Transcript extraction with fallback:
  - LangChain YoutubeLoader  
  - YouTube Transcript API  
  - yt-dlp (automatic captions)  
- 🧠 AI-powered summarization  
- 📰 Converts output into structured article format  
- ⚡ Built with modular and scalable architecture  
- 🖥 Interactive UI using Streamlit  

---

##  Architecture
User Input → Transcript Extraction → Fallback Handling → Text Processing → AI Summarization → Article Generation → Output

---

##  Tech Stack

- Python  
- LangChain (Runnables, Agents, Middleware)  
- Streamlit  
- YouTube Transcript API  
- yt-dlp  
- Google Gemini / Groq (LLMs)  

---
## Key Learnings
Handling real-world API failures with fallback strategies

Building modular and scalable AI systems

Working with LangChain pipelines and agents

Designing user-friendly AI applications


