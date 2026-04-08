import streamlit as st
from controllers.chat_controllers import ChatController
from utils.logger import setup_logger

logger = setup_logger()

logger.info("Streamlit app started")

st.set_page_config(page_title="Career Advisor", page_icon="💼")

st.title("💼 Career Advisor Chatbot")

# -------------------------------
# Initialize Controller
# -------------------------------
if "controller" not in st.session_state:
    st.session_state.controller = ChatController()

if "messages" not in st.session_state:
    st.session_state.messages = []

controller = st.session_state.controller

# -------------------------------
# Display Messages
# -------------------------------
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

# -------------------------------
# Input
# -------------------------------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = controller.get_reply(user_input)
            st.write(reply)

    st.session_state.messages.append(("assistant", reply))