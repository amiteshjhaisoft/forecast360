import streamlit as st

st.set_page_config(page_title="Forecast360", page_icon="🟠", layout="wide", initial_sidebar_state="expanded")

from ui.home import render_home
from ui.getting_started import render_getting_started
from ui.decision_intelligence import render_decision_intelligence

def main():
    tab_home, tab_gs, tab_di = st.tabs(["Home", "Getting Started", "Decision Intelligence"])
    with tab_home:
        render_home()
    with tab_gs:
        render_getting_started()
    with tab_di:
        render_decision_intelligence()

if __name__ == "__main__":
    main()
