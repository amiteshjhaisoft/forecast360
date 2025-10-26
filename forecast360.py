# app.py
import streamlit as st
import streamlit.components.v1 as components

# --- Page config (call once, at top) ---
st.set_page_config(
    page_title="Forecast360",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"  # JS/CSS will hide it for Home & AI Agent
)

# --- Imports for the three tabs ---
from home import page_home

from gs import render_getting_started, render_sidebar 
from agent import render_agent  # must expose render_agent() in agent.py

# --- Sidebar visibility controller (JS + CSS) ---
# Shows sidebar ONLY on "Getting Started", hides it on "Home" and "AI Agent".
components.html(
    """
    <script>
    (function () {
      const doc = window.parent.document;

      function activeLabel() {
        const btn = doc.querySelector('div[role="tablist"] button[role="tab"][aria-selected="true"]');
        return btn ? btn.textContent.trim() : '';
      }

      function toggleSidebar() {
        const label = activeLabel();
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        const chevron = doc.querySelector('[data-testid="collapsedControl"]');

        // Tag body so CSS can act as a fallback
        doc.body.setAttribute('data-tab', label);

        if (!sidebar) return;
        if (label === "Getting Started") {
          sidebar.style.display = "";
          if (chevron) chevron.style.display = "";
        } else {
          sidebar.style.display = "none";
          if (chevron) chevron.style.display = "none";
        }
      }

      // Observe tab selection changes
      const obs = new MutationObserver(toggleSidebar);
      obs.observe(doc, { subtree: true, attributes: true, attributeFilter: ["aria-selected"] });

      window.addEventListener("hashchange", toggleSidebar);
      window.addEventListener("load", toggleSidebar);
      toggleSidebar();
    })();
    </script>

    <style>
      /* CSS fallback to guarantee hiding on Home & AI Agent */
      body[data-tab="Home"] section[data-testid="stSidebar"],
      body[data-tab="AI Agent"] section[data-testid="stSidebar"] { display: none !important; }

      body[data-tab="Home"] [data-testid="collapsedControl"],
      body[data-tab="AI Agent"] [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    height=0,
)

# --- Tabs ---
tab_home, tab_gs, tab_agent = st.tabs(["Home", "Getting Started", "AI Agent"])

# --- Render each tab ---
with tab_home:
    page_home()  # Sidebar hidden

with tab_gs:
    render_getting_started()
    render_gs_sidebar()     # Sidebar visible only here

with tab_agent:
    render_agent()          # Sidebar hidden
