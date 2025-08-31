import io
import json
from pathlib import Path
import textwrap

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from io import BytesIO
from PIL import Image

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as _pdf_canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parents) else Path('.')
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI Audit Dashboard", layout="wide")

# ---------- CONSTANT AUTHOR INFO ----------
AUTHOR_NAME = "Thanh Trung Le"

# ---------- SIDEBAR: Artifacts & Export ----------
st.sidebar.header("Artifacts")
metrics_base_p = st.sidebar.text_input("Metrics (original)", str(REPORTS / "metrics.json"))
metrics_rep_p  = st.sidebar.text_input("Metrics (repaired)",  str(REPORTS / "metrics_repaired.json"))

bias_base_p = st.sidebar.text_input("Bias summary (original)", str(REPORTS / "bias_summary.json"))
bias_rep_p  = st.sidebar.text_input("Bias summary (repaired)", str(REPORTS / "bias_summary_repaired.json"))

rob_base_p = st.sidebar.text_input("Robustness (original)", str(REPORTS / "robustness_summary.json"))
rob_rep_p  = st.sidebar.text_input("Robustness (repaired)", str(REPORTS / "robustness_summary_repaired.json"))

sim_pairs_p = st.sidebar.text_input("Similarity pairs CSV", str(REPORTS / "similarity_pairs.csv"))

fairness_p = st.sidebar.text_input("Fairness JSON (optional)", str(REPORTS / "fairness.json"))

drift_html_p = Path("reports/drift_report.html")

shap_img_p = st.sidebar.text_input("SHAP image (optional)", str(REPORTS / "shap_summary.png"))
lime_img_p = st.sidebar.text_input("LIME image (optional)", str(REPORTS / "lime_example.png"))

audit_md_p = st.sidebar.text_input("Audit summary (MD)", str(REPORTS / "audit_summary.md"))

st.sidebar.header("Export")
exp_name = st.sidebar.text_input("PDF filename", "audit_summary.pdf")
if not exp_name.lower().endswith(".pdf"):
    exp_name += ".pdf"

if REPORTLAB_OK and st.sidebar.button("📄 Export audit_summary.md → PDF"):
    md_path = Path(audit_md_p)
    if not md_path.exists():
        st.sidebar.error(f"Not found: {md_path}")
    else:
        pdf_bytes = io.BytesIO()
        c = _pdf_canvas.Canvas(pdf_bytes, pagesize=letter)
        width, height = letter
        margin = 54
        y = height - margin
        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, "AI Audit Summary")
        y -= 20
        # Author
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, f"Prepared by: {AUTHOR_NAME}")
        y -= 16
        # Body
        c.setFont("Helvetica", 11)
        text = Path(md_path).read_text(encoding="utf-8")
        for raw_line in text.splitlines():
            wrapped = textwrap.wrap(raw_line, width=95) or [""]
            for line in wrapped:
                if y < margin + 20:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y = height - margin
                c.drawString(margin, y, line)
                y -= 14
        c.save()
        pdf_bytes.seek(0)
        st.sidebar.download_button("Download PDF", data=pdf_bytes, file_name=exp_name, mime="application/pdf")
elif not REPORTLAB_OK:
    st.sidebar.info("Install `reportlab` to enable PDF export: pip install reportlab")

# ---------- HEADER / HERO ----------
col_left, col_right = st.columns([6, 2])
with col_left:
    st.markdown(
        """
        <div style='display:flex;flex-direction:column;gap:6px;'>
          <h1 style='margin:0;'>🛡️ AI Audit Framework</h1>
          <div style='opacity:0.85;'>Dataset & Model Pre-Flight — duplicates • bias/toxicity • robustness • repairs • summary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    try:
        AVATAR_URL = "https://media.licdn.com/dms/image/v2/D5603AQGGjnTXj02LXw/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1701709248038?e=1759363200&v=beta&t=JgQ1PUM51Ci5zo15qSYmDL9XkqLcE7GxaavPbjCbzfg"  # 👈 replace with your avatar URL
        response = requests.get(AVATAR_URL, timeout=10)
        img = Image.open(BytesIO(response.content))

        # Convert to base64 for embedding in HTML
        import base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{img_b64}"
                     style="width:50%; border-radius:50%; object-fit:cover;"/>
                <div style="font-size:1.3em; font-weight:bold; margin-top:8px;">
                    Thanh Trung Le
                </div>
                <div style="font-size:0.8em; opacity:0.7;">
                    Software Engineering - AWS ML
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.info(f"Could not load avatar from URL: {e}")

st.divider()

# ---------- HELPERS ----------
def read_json(path: str | Path):
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def read_csv(path: str | Path):
    p = Path(path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.warning(f"Could not read CSV: {p}: {e}")
    return None

PLOTLY_TEMPLATE = "plotly"  # always light theme

# ---------- METRICS ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Model Quality — Original")
    m = read_json(metrics_base_p)
    if m:
        st.metric("Accuracy", f"{m.get('accuracy', '—')}")
        st.metric("F1", f"{m.get('f1', '—')}")
    else:
        st.info("No original metrics found.")
with col2:
    st.subheader("📈 Model Quality — Repaired")
    m = read_json(metrics_rep_p)
    if m:
        st.metric("Accuracy", f"{m.get('accuracy', '—')}")
        st.metric("F1", f"{m.get('f1', '—')}")
    else:
        st.info("No repaired metrics found.")

st.divider()

# ---------- BIAS ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("⚖️ Bias & Toxicity — Original")
    b = read_json(bias_base_p)
    if b:
        st.write({k: b.get(k) for k in ["n_scored", "toxicity_mean", "toxicity_p95"]})
    else:
        st.info("No original bias summary found.")
with col2:
    st.subheader("⚖️ Bias & Toxicity — Repaired")
    b = read_json(bias_rep_p)
    if b:
        st.write({k: b.get(k) for k in ["n_scored", "toxicity_mean", "toxicity_p95"]})
    else:
        st.info("No repaired bias summary found.")

st.divider()

# ---------- ROBUSTNESS ----------
def robustness_table(path: str | Path):
    data = read_json(path)
    if not data or "perturbations" not in data:
        return None
    df = pd.DataFrame(data["perturbations"])  # columns: name, flip_rate, avg_conf_change
    return df

col1, col2 = st.columns(2)
with col1:
    st.subheader("🛡️ Robustness — Original")
    df = robustness_table(rob_base_p)
    if df is not None and not df.empty:
        st.dataframe(df)
        fig = px.bar(df, x="name", y="flip_rate", title="Flip Rate by Perturbation (Original)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No original robustness summary found.")

with col2:
    st.subheader("🛡️ Robustness — Repaired")
    df = robustness_table(rob_rep_p)
    if df is not None and not df.empty:
        st.dataframe(df)
        fig = px.bar(df, x="name", y="flip_rate", title="Flip Rate by Perturbation (Repaired)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No repaired robustness summary found.")

st.divider()

# ---------- SIMILARITY ----------
st.subheader("🔁 High-Similarity Pairs (Near-Duplicates)")
sim_df = read_csv(sim_pairs_p)
if sim_df is not None and not sim_df.empty:
    st.write(f"Pairs: {len(sim_df):,}")
    st.dataframe(sim_df.head(100))
else:
    st.info("No similarity_pairs.csv found.")

st.divider()

# ---------- FAIRNESS (optional) ----------
st.subheader("🧭 Fairness (Optional)")
fair = read_json(fairness_p)
if fair:
    st.json(fair)
else:
    st.info("No fairness.json found (optional).")

st.divider()

# ---------- IMAGES ----------
st.subheader("🖼️ Explainability Visuals")
left, right = st.columns(2)
with left:
    if Path(shap_img_p).exists():
        st.image(str(shap_img_p), caption="SHAP Summary", use_container_width=True)
    else:
        st.info("shap_summary.png not found.")
with right:
    if Path(lime_img_p).exists():
        st.image(str(lime_img_p), caption="LIME Example", use_container_width=True)
    else:
        st.info("lime_example.png not found.")

# ---------- DRIFT LINK ----------
st.divider()
st.subheader("🌊 Drift Report (Optional)")
if Path(drift_html_p).exists():
    st.success(f"Found: {drift_html_p}")
    html = drift_html_p.read_text(encoding="utf-8")
    components.html(html, height=900, scrolling=True)

    st.download_button(
        "⬇️ Download drift_report.html",
        data=html,
        file_name="drift_report.html",
        mime="text/html",
    )
else:
    st.info("No drift_report.html found (optional).")

st.divider()

# Footer
st.caption("© 2025 AI Audit Framework — local-only visualization. No data leaves your machine.")
