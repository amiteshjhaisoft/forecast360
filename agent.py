# Author: Amitesh Jha | iSOFT

# agent.py
import streamlit as st

def render():
    st.title("🤖 AI Agent")
    st.write("This is the AI Agent tab. No sidebar is shown here.")
    prompt = st.text_area("Ask the agent:", placeholder="Type your question...")
    if st.button("Send"):
        if prompt.strip():
            st.info("Agent (demo): Thanks! I received your message.")
        else:
            st.warning("Please enter a prompt first.")
