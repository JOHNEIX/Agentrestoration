from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

import streamlit as st

st.set_page_config(page_title="AgentRestoration · 图像修复", page_icon="✦", layout="wide")
st.html("""<style>
.stApp{background:#f5f7fb;color:#25334a}
.stMainBlockContainer{padding-top:1.4rem;max-width:1800px}
h1{font-size:2rem!important;letter-spacing:-.6px} h3{font-size:1.05rem!important}
.st-key-settings_panel,.st-key-image_panel,.st-key-flow_panel{border-radius:18px;background:#fff}
.st-key-settings_panel,.st-key-image_panel,.st-key-flow_panel{border-color:#e1e7ef}
.st-key-settings_panel div[data-testid="stMetricValue"]{font-size:1.35rem}
.eyebrow{font:600 11px system-ui;letter-spacing:2px;color:#6483ac;margin-bottom:10px}
.intro{color:#728198;font-size:14px;margin-top:-8px;margin-bottom:18px}
button[kind="primary"]{background:#2878e3;border:0;border-radius:9px}
.st-key-shutdown_workbench button{background:#dc2626!important;color:#fff!important;border:1px solid #dc2626!important;border-radius:9px}
.st-key-shutdown_workbench button:hover{background:#b91c1c!important;border-color:#b91c1c!important}
.st-key-shutdown_workbench button:disabled{opacity:.45}
div[data-testid="stImage"] img{border-radius:10px;object-fit:contain}
.st-key-image_panel div[data-testid="stImage"]{width:100%}
.st-key-image_panel div[data-testid="stImage"] img{width:100%;height:380px;background:#f7f9fc}
.st-key-image_panel div[data-testid="stExpander"] div[data-testid="stImage"] img{height:auto;max-height:300px}
.image-empty{height:380px;box-sizing:border-box;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;background:#f7f9fc;border:1px dashed #dbe3ee;border-radius:12px;color:#8a98ad;font:14px system-ui}
.image-empty span{font-size:32px;color:#a7b8d0;font-weight:300}
.flow-status-wrap{display:flex;justify-content:flex-end;width:100%;min-width:0}
.flow-status{box-sizing:border-box;display:inline-flex;align-items:center;gap:6px;max-width:min(100%,320px);padding:5px 10px;border:1px solid #e1e7ef;border-radius:14px;background:#f6f8fb;color:#64748b;font:500 12px/1.5 system-ui}
.flow-status .status-text{min-width:0;overflow-wrap:anywhere}
.flow-status .status-dot{flex:none;width:6px;height:6px;border-radius:50%;background:currentColor}
.flow-status.running{background:#eaf2ff;border-color:#cbdfff;color:#175bb5}
.flow-status.success{background:#edf8f1;border-color:#c7e8d4;color:#216942}
.flow-status.failed{background:#fff0ef;border-color:#f2cdca;color:#a52c2c}
@media(max-width:700px){.stMainBlockContainer{padding:1rem}}
</style>""")
from ui.workbench import main

main()
