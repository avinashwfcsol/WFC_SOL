import json
import PyPDF2
import streamlit as st
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

# ==========================================
# DEFINE STRICT AI OUTPUT SCHEMA (PYDANTIC)
# ==========================================
class CandidateEvaluation(BaseModel):
    reasoning: str = Field(description="A brief, 2-sentence explanation of why this candidate fits or fails.")
    missing_critical_skills: List[str] = Field(description="List of required skills from the JD missing in the resume.")
    match_score: int = Field(description="Integer out of 100 based on core requirements met.")
    is_match: bool = Field(description="Must be true ONLY if the match_score is 75 or higher.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def evaluate_resume(client, resume_text, jd_text):
    prompt = f"""
    You are an expert, highly critical IT Recruiter. Evaluate a candidate's resume against a Job Description (JD).
    You must not make errors or assumptions. If a skill is not in the resume, assume the candidate does not have it.
    
    Job Description:
    {jd_text}
    
    Resume:
    {resume_text}
    """
    try:
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateEvaluation,
                temperature=0.0,
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"is_match": False, "reasoning": f"System Error: {str(e)}", "match_score": 0, "missing_critical_skills": []}

# ==========================================
# STREAMLIT WEB APP UI
# ==========================================
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI-Powered Resume Screener")

# Securely grab the API key from Streamlit Cloud Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found! Please configure it in Streamlit Secrets.")
    st.stop()

jd_text = st.text_area("Paste the Job Description (JD) here", height=200)
uploaded_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

if st.button("Analyze Resumes", type="primary"):
    if not jd_text.strip():
        st.warning("Please paste a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        client = genai.Client(api_key=API_KEY)
        st.write("---")
        st.subheader("Evaluation Results")
        
        for file in uploaded_files:
            with st.expander(f"Processing: {file.name}", expanded=True):
                resume_text = extract_text_from_pdf(file)
                if not resume_text.strip():
                    st.error("Could not extract text from this PDF.")
                    continue
                    
                with st.spinner("Analyzing with Gemini AI..."):
                    evaluation = evaluate_resume(client, resume_text, jd_text)
                
                if evaluation.get("is_match"):
                    st.success(f"✅ MATCH! (Score: {evaluation.get('match_score')}/100)")
                    st.write(f"**Reasoning:** {evaluation.get('reasoning')}")
                else:
                    st.error(f"❌ REJECTED (Score: {evaluation.get('match_score')}/100)")
                    st.write(f"**Missing Skills:** {', '.join(evaluation.get('missing_critical_skills', []))}")
                    st.write(f"**Reasoning:** {evaluation.get('reasoning')}")
