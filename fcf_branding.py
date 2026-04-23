"""Filter Coffee Finance shared branding for Strategy Lab tools.

Drop this file into any Streamlit app in the Strategy Lab and call:

    from fcf_branding import apply_theme, render_header, render_footer
    apply_theme()
    render_header("Tool Name", "optional subtitle")
    # ... your app code ...
    render_footer()
"""
import streamlit as st

LOGO_URL = "https://filtercoffeefinance.com/filtercoffee_bgremove.png"
SITE_URL = "https://filtercoffeefinance.com"
STRATEGY_LAB_URL = "https://filtercoffeefinance.com/strategy-lab.html"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.stApp {
    background-color: #0a0a0a;
    color: #e6e6e6;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1280px;
}

/* FCF header banner */
.fcf-banner {
    background: linear-gradient(135deg, #141414 0%, #1f1f1f 100%);
    padding: 1.1rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #FFD700;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.fcf-banner-left {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex: 1;
    min-width: 0;
}
.fcf-banner-logo {
    width: 56px;
    height: 56px;
    flex-shrink: 0;
}
.fcf-banner-text { min-width: 0; }
.fcf-banner-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFD700;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.2;
}
.fcf-banner-sub {
    font-size: 0.85rem;
    color: #aaa;
    margin-top: 0.25rem;
}
.fcf-banner-brand {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    opacity: 0.85;
    text-align: right;
}
.fcf-banner-brand a {
    color: #FFD700;
    text-decoration: none;
    font-weight: 600;
}
.fcf-banner-brand a:hover { color: #E5BC47; }

/* Headings */
h1, h2, h3, h4 {
    color: #f0f0f0 !important;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #161616;
    padding: 1rem 1.1rem;
    border-radius: 8px;
    border: 1px solid #262626;
    border-left: 3px solid #FFD700;
}
[data-testid="stMetricValue"] {
    color: #FFD700 !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #aaa !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] {
    color: #d4d4d4 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f0f;
    border-right: 1px solid #242424;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #d0d0d0;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFD700 !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background-color: #FFD700;
    color: #0a0a0a;
    border: none;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    transition: background-color 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #E5BC47;
    color: #0a0a0a;
}

/* Inputs */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input, .stTextInput input {
    background-color: #161616;
    color: #e6e6e6;
}

/* Dataframes */
.stDataFrame, [data-testid="stTable"] {
    background-color: #141414;
    border-radius: 8px;
}

/* Expanders / alerts */
.streamlit-expanderHeader {
    background-color: #161616;
    color: #e6e6e6;
    border: 1px solid #262626;
    border-radius: 8px;
}
.stAlert {
    background-color: #161616;
    border-left: 3px solid #FFD700;
    color: #e6e6e6;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #262626;
}
.stTabs [data-baseweb="tab"] {
    color: #aaa;
    background-color: transparent;
}
.stTabs [aria-selected="true"] {
    color: #FFD700 !important;
    border-bottom-color: #FFD700 !important;
}

/* Footer */
.fcf-footer {
    margin-top: 3rem;
    padding: 1.5rem 1rem;
    border-top: 1px solid #262626;
    text-align: center;
    font-size: 0.8rem;
    color: #888;
}
.fcf-footer-logo {
    width: 32px;
    height: 32px;
    opacity: 0.7;
    margin-bottom: 0.5rem;
}
.fcf-footer a {
    color: #FFD700;
    text-decoration: none;
    font-weight: 600;
}
.fcf-footer a:hover { color: #E5BC47; }
.fcf-footer .disclaimer {
    margin-top: 0.6rem;
    font-size: 0.7rem;
    color: #666;
    line-height: 1.5;
}
</style>
"""


def apply_theme() -> None:
    """Inject the Filter Coffee Finance dark theme CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(tool_title: str, tool_subtitle: str = "") -> None:
    """Render the standard FCF header banner with brand logo."""
    subtitle_html = f'<div class="fcf-banner-sub">{tool_subtitle}</div>' if tool_subtitle else ""
    st.markdown(
        f"""
<div class="fcf-banner">
  <div class="fcf-banner-left">
    <img src="{LOGO_URL}" alt="Filter Coffee Finance" class="fcf-banner-logo" />
    <div class="fcf-banner-text">
      <div class="fcf-banner-title">{tool_title}</div>
      {subtitle_html}
    </div>
  </div>
  <div class="fcf-banner-brand">
    <a href="{STRATEGY_LAB_URL}" target="_blank">Strategy Lab</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the standard FCF footer with logo, link back, and disclaimer."""
    st.markdown(
        f"""
<div class="fcf-footer">
  <img src="{LOGO_URL}" alt="Filter Coffee Finance" class="fcf-footer-logo" /><br/>
  Part of the
  <a href="{STRATEGY_LAB_URL}" target="_blank">Filter Coffee Finance Strategy Lab</a>
  <div class="disclaimer">
    For educational purposes only. Past performance does not guarantee future results.<br/>
    Not SEBI-registered investment advice.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
