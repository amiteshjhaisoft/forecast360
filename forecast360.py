# app.py
import streamlit as st
import streamlit.components.v1 as components

from home import page_home
from gs import render as render_getting_started, render_sidebar as render_gs_sidebar
from agent import render_agent

st.set_page_config(
    page_title="Three-Tab App",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Tabs ---
tab_home, tab_gs, tab_agent = st.tabs(["Home", "Getting Started", "AI Agent"])

# --- Sidebar visibility controller (JS + CSS) ---
components.html(
    """
    <script>
    const setBodyTab = () => {
      const doc = window.parent.document;
      const tabs = doc.querySelectorAll('button[role="tab"]');
      if (!tabs || !tabs.length) return;
      const active = Array.from(tabs).find(t => t.getAttribute('aria-selected') === 'true');
      const label = active ? active.textContent.trim() : '';
      doc.body.setAttribute('data-tab', label);
    };
    setBodyTab();
    const obs = new MutationObserver(setBodyTab);
    obs.observe(window.parent.document, { subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
    </script>
    <style>
    body[data-tab="Home"] section[data-testid="stSidebar"] { display: none !important; }
    body[data-tab="AI Agent"] section[data-testid="stSidebar"] { display: none !important; }
    body[data-tab="Home"] [data-testid="collapsedControl"],
    body[data-tab="AI Agent"] [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    height=0,
)

# --- Render tabs ---
with tab_home:
    page_home()  # uses your home.py

with tab_gs:
    render_getting_started()
    render_gs_sidebar()  # sidebar shows only on this tab

with tab_agent:
    render_agent()
