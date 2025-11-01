import streamlit as st
from chatbot import chatbot_response

st.title("🤖 AI Chatbot (TensorFlow)")
user_input = st.text_input("You:")
if user_input:
    st.write("Bot:", chatbot_response(user_input))