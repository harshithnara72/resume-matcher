import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Page Config ------------------
st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="centered")

# ------------------ Background CSS ------------------
def add_bg():
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            background-attachment: fixed;
        }

        /* Title style */
        h1 {
            color: white !important;
            text-align: center;
            font-size: 45px !important;
            font-weight: 700 !important;
        }

        /* Text style */
        p, label, div {
            color: white !important;
            font-size: 16px !important;
        }

        /* Input box background */
        textarea, input {
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border-radius: 10px !important;
        }

        /* File uploader */
        section[data-testid="stFileUploader"] {
            background-color: rgba(255, 255, 255, 0.12);
            padding: 15px;
            border-radius: 12px;
        }

        /* Button style */
        div.stButton > button {
            background: linear-gradient(90deg, #ff512f, #dd2476);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 20px;
            font-size: 18px;
            font-weight: 600;
            width: 100%;
            transition: 0.3s;
        }

        div.stButton > button:hover {
            transform: scale(1.03);
            background: linear-gradient(90deg, #dd2476, #ff512f);
        }

        /* Result box */
        .result-box {
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #00ffcc;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

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
st.write("Upload your resume and paste the job description to check how well it matches.")

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

        st.markdown(f"<div class='result-box'>✅ Resume Match Score: {match_score}%</div>", unsafe_allow_html=True)

        with st.expander("📄 View Extracted Resume Text"):
            st.write(resume_text)
