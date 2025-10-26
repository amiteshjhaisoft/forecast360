# Author: Amitesh Jha | iSOFT
# home.py
from pathlib import Path
from datetime import datetime
from PIL import Image
import base64
import streamlit as st

def page_home():
    # -- Cache image -> base64 so reruns are fast
    @st.cache_data(show_spinner=False)
    def _img_to_base64_first(paths):
        for p in paths:
            fp = Path(p)
            if fp.is_file():
                try:
                    return base64.b64encode(fp.read_bytes()).decode("utf-8")
                except Exception:
                    continue
        return None

    img_b64 = _img_to_base64_first(["assets/forecast360.png"])

    # -- Scoped CSS (home only)
    # -- Styles
    st.markdown(
        """
        <style>
        .home-wrap{
            background: radial-gradient(1200px 600px at 10% -10%, rgba(0,183,255,.10), transparent 60%),
                        radial-gradient(1200px 600px at 110% 110%, rgba(255,79,160,.08), transparent 60%),
                        linear-gradient(135deg, rgba(255,136,0,.06), rgba(0,183,255,.06) 50%, rgba(255,79,160,.06));
            border: 1px solid #eaeaea; border-radius: 22px; padding: 22px 22px 14px; margin-bottom: 14px;
            box-shadow: 0 10px 24px rgba(0,0,0,.04);
        }
        .home-cols{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 26px; align-items: center; }
        .home-left h1{ margin: 0 0 8px; font-weight: 800; letter-spacing: .2px; }
        .home-left h5{ margin: 0 0 10px; font-weight: 600; color:#0f172a; opacity:.85; }
        .home-left p{ margin: 0 0 10px; color: #334155; line-height:1.5; }

        .home-right{ display:flex; align-items:center; justify-content:center; }
        .logo-wrap{
            width: min(360px, 90%);
            aspect-ratio: 1 / 1;
            display:flex; align-items:center; justify-content:center;
            background: radial-gradient(60% 60% at 50% 45%, rgba(255,255,255,.25), transparent 70%);
            border-radius: 50%;
            box-shadow: 0 20px 40px rgba(2, 6, 23, 0.12), inset 0 1px 0 rgba(255,255,255,.3);
        }
        .logo-wrap img{
            width: 100%; height: auto; display:block;
            filter: drop-shadow(0 10px 24px rgba(2, 6, 23, 0.16));
            border-radius: 50%;
        }

        .kpis{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; }
        .kcard{
            background:#fff; border:1px solid #eee; border-radius:16px; padding:14px 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,.05);
        }
        .kcard h4{ margin:0 0 6px; font-size:18px; }
        .kcard p{ margin:0; color:#475569; font-size:13px; }

        /* Small, subtle footer inside the hero */
        .home-footer{
            margin-top: 8px;
            text-align: center;
            font-size: 12px;
            line-height: 1.4;
            opacity: .75;
        }
        @media (max-width: 1100px){
            .home-cols{ grid-template-columns: 1fr; gap: 18px; }
            .logo-wrap{ width:min(300px, 70%); margin: 6px auto 0; }
            .kpis{ grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 680px){
            .kpis{ grid-template-columns: 1fr; }
            .home-footer{ font-size: 11px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -- Hero
    st.markdown(
        f"""
        <div class="home-wrap">
        <div class="home-cols">
            <div class="home-left">
            <h1 class="hero-title">Forecast360</h1>
            <h5>AI Powered Forecasting. No Code. Decisions in Minutes.</h5>
            <p>Upload any time series, auto-profile, compare models, forecast with confidence intervals,
                and turn it into <b>actionable decisions</b> with an AI analyst, speech and a talking avatar.</p>
            <p>Bring your own local LLM via <b>Ollama</b> or connect <b>Claude.ai</b>.
                Data stays local; artifacts live in your local <b>Knowledge Base</b>.</p>
            </div>
            <div class="home-right">
            <div class="logo-wrap">
                {("<img src='data:image/png;base64," + img_b64 + "' alt='Forecast360 logo'/>") if img_b64 else ""}
            </div>
            </div>
        </div>

        <div class="home-footer">© {datetime.now().year} iSOFT ANZ Pvt Ltd. All rights reserved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # icon_path = Path("assets/forecast360.png")
    # page_icon = Image.open(icon_path) if icon_path.is_file() else "📈"

    # st.set_page_config(
    #     page_title="Forecast360",
    #     page_icon=page_icon,
    #     layout="wide",
    # )
    page_home()  # make sure page_home() is imported/defined
if __name__ == "__main__":
    main()
