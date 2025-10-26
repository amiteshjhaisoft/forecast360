# app.py
import streamlit as st
import streamlit.components.v1 as components

# --- Page config (call once, at top) ---
st.set_page_config(
    icon_path = Path("assets/forecast360.png")
    page_icon = Image.open(icon_path) if icon_path.is_file() else "📈"

    st.set_page_config(
        page_title="Forecast360",
        page_icon=page_icon,
        layout="wide",
    )

# Ensure the flag exists (used by GS CTA + JS visibility)
st.session_state.setdefault("show_sidebar", False)

# --- Imports for the three tabs ---
from home import page_home
from gs import getting_started_tab        # <-- wrapper that shows CTA first, then sidebar + page
from agent import render_agent            # remains as-is

# --- Sidebar visibility controller (JS + CSS) ---
# Shows sidebar ONLY on "Getting Started" AND ONLY after CTA sets show_sidebar=True.
_show_sidebar_flag = "true" if st.session_state.get("show_sidebar", False) else "false"

components.html(
    f"""
    <script>
    (function () {{
      const doc = window.parent.document;
      const SHOW_SIDEBAR = {_show_sidebar_flag};  // injected from Python session_state

      function activeLabel() {{
        const btn = doc.querySelector('div[role="tablist"] button[role="tab"][aria-selected="true"]');
        return btn ? btn.textContent.trim() : '';
      }}

      function toggleSidebar() {{
        const label = activeLabel();
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        const chevron = doc.querySelector('[data-testid="collapsedControl"]');

        // Tag body so CSS can act as a fallback
        doc.body.setAttribute('data-tab', label);

        if (!sidebar) return;

        // Only show sidebar if we're on "Getting Started" AND the CTA has been pressed (SHOW_SIDEBAR=true)
        const shouldShow = (label === "Getting Started") && SHOW_SIDEBAR;

        sidebar.style.display = shouldShow ? "" : "none";
        if (chevron) chevron.style.display = shouldShow ? "" : "none";
      }}

      // Observe tab selection changes
      const obs = new MutationObserver(toggleSidebar);
      obs.observe(doc, {{ subtree: true, attributes: true, attributeFilter: ["aria-selected"] }});

      window.addEventListener("hashchange", toggleSidebar);
      window.addEventListener("load", toggleSidebar);
      toggleSidebar();
    }})();
    </script>

    <style>
      /* CSS fallback to guarantee hiding on Home & AI Agent */
      body[data-tab="Home"] section[data-testid="stSidebar"],
      body[data-tab="AI Agent"] section[data-testid="stSidebar"] {{
        display: none !important;
      }}

      body[data-tab="Home"] [data-testid="collapsedControl"],
      body[data-tab="AI Agent"] [data-testid="collapsedControl"] {{
        display: none !important;
      }}
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
    # This wrapper (in gs.py) shows the CTA first; once clicked, it renders sidebar + page.
    getting_started_tab()

with tab_agent:
    render_agent()  # Sidebar hidden
