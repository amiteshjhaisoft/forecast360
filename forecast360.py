# app.py
import streamlit as st
import streamlit.components.v1 as components

from home import page_home
from gs import render as render_getting_started, render_sidebar as render_gs_sidebar
import agent  # uses your provided agent.py; we will call agent.run()

# --- Tabs ---
tab_home, tab_gs, tab_agent = st.tabs(["Home", "Getting Started", "AI Agent"])

# --- Sidebar visibility controller (JS + CSS) ---
# Hides sidebar for Home & AI Agent; shows it for Getting Started
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
    page_home()  # your home.py function (no sidebar)

with tab_gs:
    render_getting_started()
    render_gs_sidebar()  # only tab that shows sidebar

with tab_agent:
    # IMPORTANT: your agent.py calls st.set_page_config inside run()
    # We purposely did NOT call st.set_page_config in this file to avoid duplicate calls.
    agent.run()  # uses your provided agent.py unchanged
