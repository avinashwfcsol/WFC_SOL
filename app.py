import json
import time
import PyPDF2
import requests
import streamlit as st

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

def evaluate_resume(api_keys, resume_text, jd_text):
    # We must explicitly instruct Grok to return raw JSON matching our app's logic
    system_prompt = """
    You are an expert, highly critical IT Recruiter. Evaluate a candidate's resume against a Job Description (JD).
    You must not make errors or assumptions. If a skill is not in the resume, assume the candidate does not have it.
    
    You MUST return your evaluation strictly as a valid JSON object with the following exact keys:
    - "reasoning": A brief, 2-sentence explanation of why this candidate fits or fails.
    - "missing_critical_skills": A list of strings containing required skills from the JD missing in the resume.
    - "match_score": An integer out of 100 based on core requirements met.
    - "is_match": A boolean (true or false). Must be true ONLY if the match_score is 75 or higher.
    
    Do not include markdown formatting like ```json, just output the raw JSON object.
    """
    
    user_prompt = f"Job Description:\n{jd_text}\n\nResume:\n{resume_text}"

    # API KEY ROTATION LOGIC
    for i, key in enumerate(api_keys):
        try:
            # We use xAI's standard chat completions endpoint
            url = "[https://api.x.ai/v1/chat/completions](https://api.x.ai/v1/chat/completions)"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            }
            payload = {
                "model": "grok-4.5",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            # Catch Rate Limits (429) or Server Errors (500+) and rotate keys
            if response.status_code == 429 or response.status_code >= 500:
                if i == len(api_keys) - 1:
                    return {"is_match": False, "reasoning": f"All API keys exhausted. Last Error Code: {response.status_code}", "match_score": 0, "missing_critical_skills": []}
                continue # Instantly try the next key in the list
                
            response.raise_for_status() # Catch any other errors
            
            # Extract and clean the JSON text from Grok's response
            result_text = response.json()["choices"][0]["message"]["content"]
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(result_text)
            
        except Exception as e:
            error_message = str(e)
            
            # If we just failed on the VERY LAST key in our list, give up
            if i == len(api_keys) - 1:
                return {"is_match": False, "reasoning": f"System Error: {error_message}", "match_score": 0, "missing_critical_skills": []}
            
            # Otherwise, move to the next key
            continue

# ==========================================
# STREAMLIT WEB APP UI
# ==========================================
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 Grok-Powered Resume Screener")

# Securely grab ALL API keys from Streamlit Secrets
api_keys = []
for idx in range(1, 10): 
    key = st.secrets.get(f"GROK_API_KEY_{idx}")
    if key:
        api_keys.append(key)

if not api_keys:
    st.error("No API keys found! Please configure GROK_API_KEY_1 in Streamlit Secrets.")
    st.stop()

jd_text = st.text_area("Paste the Job Description (JD) here", height=200)
uploaded_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

if st.button("Analyze Resumes", type="primary"):
    if not jd_text.strip():
        st.warning("Please paste a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        st.write("---")
        st.subheader("Evaluation Results")
        
        for file in uploaded_files:
            with st.expander(f"Processing: {file.name}", expanded=True):
                resume_text = extract_text_from_pdf(file)
                if not resume_text.strip():
                    st.error("Could not extract text from this PDF.")
                    continue
                    
                with st.spinner("Analyzing with Grok AI..."):
                    evaluation = evaluate_resume(api_keys, resume_text, jd_text)
                
                if "All API keys exhausted" in evaluation.get("reasoning", ""):
                    st.warning("⚠️ ERROR: All provided API keys have hit their rate limits.")
                    st.write(f"**Details:** {evaluation.get('reasoning')}")
                elif evaluation.get("is_match"):
                    st.success(f"✅ MATCH! (Score: {evaluation.get('match_score')}/100)")
                    st.write(f"**Reasoning:** {evaluation.get('reasoning')}")
                else:
                    st.error(f"❌ REJECTED (Score: {evaluation.get('match_score', 0)}/100)")
                    st.write(f"**Missing Skills:** {', '.join(evaluation.get('missing_critical_skills', []))}")
                    st.write(f"**Reasoning:** {evaluation.get('reasoning')}")
                
                # Pace the requests to avoid overwhelming the xAI servers
                time.sleep(3)
