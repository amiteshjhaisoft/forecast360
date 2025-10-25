# Author: Amitesh Jha | iSOFT

# app.py
import streamlit as st
import streamlit.components.v1 as components

from home import render as render_home
from gs import render as render_getting_started, render_sidebar as render_gs_sidebar
from agent import render as render_agent

st.set_page_config(
    page_title="Three-Tab App",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",  # start collapsed (no sidebar for Home by default)
)

# --- Tabs ---
tab_home, tab_gs, tab_agent = st.tabs(["Home", "Getting Started", "AI Agent"])

# --- Sidebar visibility controller (JS + CSS) ---
# Detects active tab and hides sidebar for Home & AI Agent, shows it for Getting Started.
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

    // Run now and whenever tab selection changes
    setBodyTab();
    const obs = new MutationObserver(setBodyTab);
    obs.observe(window.parent.document, { subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
    </script>
    <style>
    /* Hide sidebar for these tabs */
    body[data-tab="Home"] section[data-testid="stSidebar"] { display: none !important; }
    body[data-tab="AI Agent"] section[data-testid="stSidebar"] { display: none !important; }
    /* Optional: hide the collapse chevron when hidden (cosmetic) */
    body[data-tab="Home"] [data-testid="collapsedControl"] { display: none !important; }
    body[data-tab="AI Agent"] [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    height=0,
)

# --- Render tabs ---
with tab_home:
    render_home()

with tab_gs:
    render_getting_started()
    # Sidebar content ONLY for Getting Started (still defined here, but CSS keeps it hidden on other tabs)
    render_gs_sidebar()

with tab_agent:
    render_agent()
