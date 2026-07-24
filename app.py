import json
import time
import PyPDF2
import requests
import streamlit as st

# ==========================================
# CONFIG
# ==========================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use a free model by default to avoid 402 errors.
# You can change this in Streamlit secrets as OPENROUTER_MODEL if you want.
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

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


def evaluate_resume(api_keys, resume_text, jd_text, model_name):
    system_prompt = """
You are an expert, highly critical IT Recruiter. Evaluate a candidate's resume against a Job Description (JD).
You must not make errors or assumptions. If a skill is not in the resume, assume the candidate does not have it.

You MUST return your evaluation strictly as a valid JSON object with the following exact keys:
- "reasoning": A brief, 2-sentence explanation of why this candidate fits or fails.
- "missing_critical_skills": A list of strings containing required skills from the JD missing in the resume.
- "match_score": An integer out of 100 based on core requirements met.
- "is_match": A boolean (true or false). Must be true ONLY if the match_score is 75 or higher.

Do not include markdown formatting like ```json, just output the raw JSON object.
""".strip()

    user_prompt = f"Job Description:\n{jd_text}\n\nResume:\n{resume_text}"

    for i, key in enumerate(api_keys):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "AI Resume Screener",
            }

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
            }

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=90,
            )

            # Print exact API error details for debugging
            if not response.ok:
                error_text = response.text
                if response.status_code in (429, 500, 502, 503, 504):
                    if i == len(api_keys) - 1:
                        return {
                            "is_match": False,
                            "reasoning": f"All API keys exhausted or server error. Status: {response.status_code}. Details: {error_text}",
                            "match_score": 0,
                            "missing_critical_skills": [],
                        }
                    continue
                else:
                    if i == len(api_keys) - 1:
                        return {
                            "is_match": False,
                            "reasoning": f"OpenRouter error {response.status_code}: {error_text}",
                            "match_score": 0,
                            "missing_critical_skills": [],
                        }
                    continue

            data = response.json()
            result_text = data["choices"][0]["message"]["content"].strip()

            # Remove code fences if the model adds them anyway
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(result_text)

            return {
                "reasoning": parsed.get("reasoning", ""),
                "missing_critical_skills": parsed.get("missing_critical_skills", []),
                "match_score": int(parsed.get("match_score", 0)),
                "is_match": bool(parsed.get("is_match", False)),
            }

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            if i == len(api_keys) - 1:
                return {
                    "is_match": False,
                    "reasoning": f"System Error: {str(e)}",
                    "match_score": 0,
                    "missing_critical_skills": [],
                }
            continue


# ==========================================
# STREAMLIT WEB APP UI
# ==========================================
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 OpenRouter-Powered Resume Screener")

# Get model from secrets if present, otherwise use free model
model_name = st.secrets.get("OPENROUTER_MODEL", DEFAULT_MODEL)

# Securely grab ALL API keys from Streamlit Secrets
api_keys = []
for idx in range(1, 10):
    key = st.secrets.get(f"OPENROUTER_API_KEY_{idx}")
    if key:
        api_keys.append(key)

if not api_keys:
    st.error("No API keys found! Please configure OPENROUTER_API_KEY_1 in Streamlit Secrets.")
    st.stop()

st.caption(f"Using model: {model_name}")

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

                with st.spinner("Analyzing via OpenRouter..."):
                    evaluation = evaluate_resume(api_keys, resume_text, jd_text, model_name)

                if "All API keys exhausted" in evaluation.get("reasoning", ""):
                    st.warning("⚠️ API limit or server issue.")
                    st.write(f"**Details:** {evaluation.get('reasoning')}")
                elif evaluation.get("is_match"):
                    st.success(f"✅ MATCH! (Score: {evaluation.get('match_score')}/100)")
                    st.write(f"**Reasoning:** {evaluation.get('reasoning')}")
                    missing = evaluation.get("missing_critical_skills", [])
                    if missing:
                        st.write(f"**Missing Skills:** {', '.join(missing)}")
                else:
                    st.error(f"❌ REJECTED (Score: {evaluation.get('match_score', 0)}/100)")
                    missing = evaluation.get("missing_critical_skills", [])
                    st.write(f"**Missing Skills:** {', '.join(missing) if missing else 'None found'}")
                    st.write(f"**Reasoning:** {evaluation.get('reasoning')}")

                time.sleep(1)
