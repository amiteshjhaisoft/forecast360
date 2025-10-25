# Author: Amitesh Jha | iSOFT

# gs.py
import streamlit as st

def render():
    st.title("🚀 Getting Started")
    st.write(
        "This tab includes helpful steps and shows a sidebar with quick actions and resources."
    )

def render_sidebar():
    with st.sidebar:
        st.header("Quick Start")
        st.markdown("- ✅ Install prerequisites")
        st.markdown("- ⚙️ Configure settings")
        st.markdown("- ▶️ Run your first task")

        st.divider()
        st.subheader("Settings")
        name = st.text_input("Project name", value="My Project")
        level = st.selectbox("Experience level", ["Beginner", "Intermediate", "Advanced"])
        st.write(f"Selected: **{name}** · **{level}**")

        st.divider()
        st.subheader("Actions")
        if st.button("Initialize Project"):
            st.success("Project initialized! 🎉")
