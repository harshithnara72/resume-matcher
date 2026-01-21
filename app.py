import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="centered")

# ------------------ Background Image + Blur CSS ------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: none;
        }}

        /* Background Image */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            filter: blur(8px);
            transform: scale(1.1);
            z-index: -2;
        }}

        /* Dark overlay for readability */
        .stApp::after {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background: rgba(0,0,0,0.65);
            z-index: -1;
        }}

        /* Title */
        h1 {{
            color: white !important;
            text-align: center;
            font-size: 42px !important;
            font-weight: 800 !important;
        }}

        /* Normal text */
        p, label, div {{
            color: white !important;
            font-size: 16px !important;
        }}

        /* Input boxes */
        textarea, input {{
            background-color: rgba(255,255,255,0.15) !important;
            color: white !important;
            border-radius: 12px !important;
        }}

        /* File uploader box */
        section[data-testid="stFileUploader"] {{
            background-color: rgba(255,255,255,0.12);
            padding: 15px;
            border-radius: 15px;
        }}

        /* Button */
        div.stButton > button {{
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 12px 20px;
            font-size: 18px;
            font-weight: 700;
            width: 100%;
            transition: 0.3s;
        }}

        div.stButton > button:hover {{
            transform: scale(1.03);
        }}

        /* Result card */
        .result-box {{
            background: rgba(255,255,255,0.15);
            padding: 18px;
            border-radius: 16px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #00ffcc;
            margin-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background (make sure bg.jpg exists in repo)
add_bg_from_local("bg.jpg")

# ------------------ Load Model ------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Functions ------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def calculate_match(resume_text, job_desc):
    resume_embedding = model.encode([resume_text])
    jd_embedding = model.encode([job_desc])
    similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
    return round(similarity * 100, 2)

# ------------------ UI ------------------
st.title("📄 Resume Screening & Job Matching System")
st.write("Upload your resume and paste the job description to check the match score.")

uploaded_resume = st.file_uploader("📌 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description Here", height=180)

if st.button("🔍 Check Match Percentage"):
    if uploaded_resume is None:
        st.warning("⚠️ Please upload a resume PDF.")
    elif job_description.strip() == "":
        st.warning("⚠️ Please enter the job description.")
    else:
        resume_text = extract_text_from_pdf(uploaded_resume)
        match_score = calculate_match(resume_text, job_description)

        st.markdown(
            f"<div class='result-box'>✅ Resume Match Score: {match_score}%</div>",
            unsafe_allow_html=True
        )

        with st.expander("📄 View Extracted Resume Text"):
            st.write(resume_text)
