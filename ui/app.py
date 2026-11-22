from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

import streamlit as st

st.set_page_config(page_title="AgentRestoration · 图像修复", page_icon="✦", layout="wide")
st.html("""<style>
.stApp{background:#f5f7fb;color:#25334a}
.stMainBlockContainer{padding-top:2rem;max-width:1800px}
h1{font-size:2rem!important;letter-spacing:-.6px} h3{font-size:1.05rem!important}
div[data-testid="stColumn"]>div[data-testid="stVerticalBlockBorderWrapper"]{background:white;border-radius:16px}
.eyebrow{font:600 11px system-ui;letter-spacing:2px;color:#6483ac;margin-bottom:10px}
.intro{color:#728198;font-size:14px;margin-top:-8px;margin-bottom:28px}
button[kind="primary"]{background:#2878e3;border:0;border-radius:9px}
.st-key-shutdown_workbench button{background:#dc2626!important;color:#fff!important;border:1px solid #dc2626!important;border-radius:9px}
.st-key-shutdown_workbench button:hover{background:#b91c1c!important;border-color:#b91c1c!important}
.st-key-shutdown_workbench button:disabled{opacity:.45}
div[data-testid="stImage"] img{border-radius:10px;object-fit:contain}
</style>""")
from ui.workbench import main

main()
