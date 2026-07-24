import csv
import io
import json
import re
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage

import PyPDF2
import requests
import streamlit as st

# ==========================================
# CONFIG
# ==========================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"   # free routing model

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

def extract_candidate_name(resume_text, filename):
    """
    Best-effort candidate name extraction.
    """
    patterns = [
        r"(?i)\bname[:\s-]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
        r"(?i)\bcandidate name[:\s-]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
    ]
    for pat in patterns:
        m = re.search(pat, resume_text)
        if m:
            return m.group(1).strip()

    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]  
    for line in lines[:8]:  
        if re.fullmatch(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}", line):  
            return line.strip()  

    base = filename.rsplit(".", 1)[0]  
    base = re.sub(r"[_\-]+", " ", base).strip()  
    return base[:60] if base else filename

def normalize_list(value):
    """
    Forces the AI output strictly into a list of strings to prevent UI/CSV crashes.
    """
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(value)]

def classify_recommendation(score):
    if score >= 90:
        return "Highly Recommended"
    if score >= 75:
        return "Recommended"
    if score >= 60:
        return "Review"
    return "Not Recommended"

def confidence_label(score):
    if score >= 85:
        return "high"
    if score >= 70:
        return "medium"
    return "low"

def evaluate_resume(api_keys, resume_text, jd_text, model_name):
    system_prompt = """
You are an expert, highly critical IT Recruiter.
Compare the candidate's resume against the Job Description carefully.
Be strict and job-relevant. Do not assume skills that are not explicitly present in the resume.

Return ONLY valid JSON with these exact keys:
"candidate_name": string
"match_score": integer from 0 to 100
"is_match": boolean (true only if match_score is 75 or higher)
"matched_skills": array of strings (WARNING: Only include skills explicitly written in the resume. Do not invent matches. If none, return an empty array [])
"missing_critical_skills": array of strings (List the mandatory JD skills that are missing from the resume)
"why_matched": string (Provide detail ONLY if is_match is true. If false, leave as empty string)
"why_not_matched": string (Provide detail ONLY if is_match is false. If true, leave as empty string)
"overall_summary": string
"confidence_level": string ("low", "medium", or "high")

Rules:
If a skill is missing from the resume, treat it as missing.
Be careful and do not make assumptions.
No markdown. No code fences. Only raw JSON.
""".strip()

    user_prompt = f"""Job Description:\n{jd_text}\n\nResume:\n{resume_text}"""

    last_error = ""
    # Fallback Strategy: Loop through available OpenRouter keys sequentially
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
                timeout=120,  
            )  

            if not response.ok:  
                last_error = f"OpenRouter status {response.status_code}: {response.text}"
                # If this is not the last key, fallback to the next OpenRouter key
                continue  

            data = response.json()  
            result_text = data["choices"][0]["message"]["content"].strip()  

            result_text = result_text.replace("```json", "").replace("```", "").strip()  

            start_idx = result_text.find("{")  
            end_idx = result_text.rfind("}")  

            if start_idx != -1 and end_idx != -1:  
                clean_json_str = result_text[start_idx:end_idx + 1]  
                parsed = json.loads(clean_json_str)  
            else:  
                raise ValueError(f"Model output did not contain valid JSON: {result_text}")  

            score = int(parsed.get("match_score", 0))  

            return {  
                "candidate_name": parsed.get("candidate_name", ""),  
                "match_score": score,  
                "is_match": bool(parsed.get("is_match", False)),  
                "matched_skills": normalize_list(parsed.get("matched_skills", [])),  
                "missing_critical_skills": normalize_list(parsed.get("missing_critical_skills", [])),  
                "why_matched": parsed.get("why_matched", ""),  
                "why_not_matched": parsed.get("why_not_matched", ""),  
                "overall_summary": parsed.get("overall_summary", ""),  
                "confidence_level": parsed.get("confidence_level", confidence_label(score)),  
            }  

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:  
            last_error = f"System Error: {str(e)}"
            continue

    # If all OpenRouter keys fail, return standard error payload
    return {  
        "candidate_name": "",  
        "is_match": False,  
        "reasoning": f"All OpenRouter API keys failed. Last error: {last_error}",  
        "match_score": 0,  
        "missing_critical_skills": [],  
        "matched_skills": [],  
        "why_matched": "",  
        "why_not_matched": "",  
        "overall_summary": "",  
        "confidence_level": "low",  
    }

def build_csv_bytes(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "rank",
        "scan_date",
        "candidate_name",
        "file_name",
        "match_score",
        "match_percentage",
        "match_status",
        "recommendation",
        "confidence_level",
        "matched_skills",
        "missing_critical_skills",
        "why_matched",
        "why_not_matched",
        "overall_summary",
    ])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")

def send_email_with_csv(csv_bytes, to_email, subject, body):
    smtp_host = st.secrets.get("SMTP_HOST")
    smtp_port = int(st.secrets.get("SMTP_PORT", 587))
    smtp_username = st.secrets.get("SMTP_USERNAME")
    smtp_password = st.secrets.get("SMTP_PASSWORD")
    email_from = st.secrets.get("EMAIL_FROM", smtp_username)

    if not smtp_host or not smtp_username or not smtp_password:  
        raise ValueError("SMTP settings are missing in Streamlit Secrets.")  

    msg = EmailMessage()  
    msg["From"] = email_from  
    msg["To"] = to_email  
    msg["Subject"] = subject  
    msg.set_content(body)  

    msg.add_attachment(  
        csv_bytes,  
        maintype="text",  
        subtype="csv",  
        filename="resume_screening_report.csv",  
    )  

    with smtplib.SMTP(smtp_host, smtp_port) as server:  
        server.starttls()  
        server.login(smtp_username, smtp_password)  
        server.send_message(msg)

# ==========================================
# STREAMLIT WEB APP UI
# ==========================================
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# This hides the anchor link (🔗) next to headers
st.markdown(
    """
    <style>
    .st-emotion-cache-10trblm {display: none;}
    a.header-anchor {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 AI-Powered Resume Screener")

model_name = st.secrets.get("OPENROUTER_MODEL", DEFAULT_MODEL)

api_keys = []
for idx in range(1, 10):
    key = st.secrets.get(f"OPENROUTER_API_KEY_{idx}")
    if key:
        api_keys.append(key)

if not api_keys:
    st.error("No API keys found! Please configure OPENROUTER_API_KEY_1 in Streamlit Secrets.")
    st.stop()

jd_text = st.text_area("Paste the Job Description (JD) here", height=200)
uploaded_files = st.file_uploader(
    "Upload Resumes (PDFs)",
    type="pdf",
    accept_multiple_files=True
)

st.markdown("### Email Report")
send_report_email = st.checkbox("Send CSV report by email after scanning", value=False)

report_email_to = ""
if send_report_email:
    report_email_to = st.text_input("Recipient email", placeholder="recruiter@company.com")

if st.button("Analyze Resumes", type="primary"):
    if not jd_text.strip():
        st.warning("Please paste a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        st.write("---")
        st.subheader("Evaluation Results")

        scan_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
        rows = []  

        progress = st.progress(0)  
        total_files = len(uploaded_files)  

        for idx, file in enumerate(uploaded_files, start=1):  
            resume_text = extract_text_from_pdf(file)  

            if not resume_text.strip():  
                st.error(f"Could not extract text from this PDF: {file.name}")  
                progress.progress(idx / total_files)  
                continue  

            fallback_name = extract_candidate_name(resume_text, file.name)  
            
            with st.spinner(f"Analyzing {file.name} via AI..."):  
                evaluation = evaluate_resume(api_keys, resume_text, jd_text, model_name)  

            candidate_name = evaluation.get("candidate_name") or fallback_name  
            score = int(evaluation.get("match_score", 0))  
            is_match = bool(evaluation.get("is_match", False))  
            matched_skills = evaluation.get("matched_skills", [])  
            missing_skills = evaluation.get("missing_critical_skills", [])  
            why_matched = evaluation.get("why_matched", "")  
            why_not_matched = evaluation.get("why_not_matched", "")  
            overall_summary = evaluation.get("overall_summary", "")  
            confidence_level = evaluation.get("confidence_level", confidence_label(score))  
            recommendation = classify_recommendation(score)  

            expander_title = f"{'✅ MATCH' if is_match else '❌ REJECTED'} - {candidate_name} ({score}/100)"
            
            with st.expander(expander_title, expanded=is_match):
                if "OpenRouter error" in evaluation.get("reasoning", "") or "System Error" in evaluation.get("reasoning", "") or "All OpenRouter API keys failed" in evaluation.get("reasoning", ""):  
                    st.warning("⚠️ Error while processing this resume.")  
                    st.write(f"**Details:** {evaluation.get('reasoning')}")  
                else:  
                    st.write(f"**Candidate Name:** {candidate_name}")  
                    st.write(f"**Recommendation:** {recommendation}")  
                    st.write(f"**Confidence:** {confidence_level}")  
                    
                    if is_match:
                        st.write(f"**Why Matched:** {why_matched or 'Details not provided'}")
                    else:
                        st.write(f"**Why Not Matched:** {why_not_matched or 'Details not provided'}")
                        
                    st.write(f"**Summary:** {overall_summary or 'Not provided'}")  

                    if matched_skills:  
                        st.write(f"**Matched Skills:** {', '.join(matched_skills)}")  
                    else:  
                        st.write("**Matched Skills:** None found")  

                    if missing_skills:  
                        st.write(f"**Missing Skills:** {', '.join(missing_skills)}")  
                    else:  
                        st.write("**Missing Skills:** None found")  

            rows.append({  
                "scan_date": scan_date,  
                "candidate_name": candidate_name,  
                "file_name": file.name,  
                "match_score": score,  
                "match_percentage": f"{score}%",  
                "match_status": "Matched" if is_match else "Rejected",  
                "recommendation": recommendation,  
                "confidence_level": confidence_level,  
                "matched_skills": ", ".join(matched_skills) if matched_skills else "",  
                "missing_critical_skills": ", ".join(missing_skills) if missing_skills else "",  
                "why_matched": why_matched if is_match else "",  
                "why_not_matched": why_not_matched if not is_match else "",  
                "overall_summary": overall_summary,  
            })  

            time.sleep(5)  

            progress.progress(idx / total_files)  

        rows.sort(key=lambda x: x["match_score"], reverse=True)  

        for i, row in enumerate(rows, start=1):  
            row["rank"] = i  

        st.write("---")  
        st.subheader("Ranked Report (Highest Match to Lowest Match)")  

        if rows:  
            csv_bytes = build_csv_bytes(rows)  

            st.download_button(  
                label="⬇️ Download CSV Report",  
                data=csv_bytes,  
                file_name="resume_screening_report.csv",  
                mime="text/csv",  
            )  

            try:  
                import pandas as pd  
                df = pd.DataFrame(rows)  
                st.dataframe(df, use_container_width=True)  
            except Exception:  
                st.json(rows)  

            if send_report_email:  
                if not report_email_to.strip():  
                    st.warning("Please enter recipient email.")  
                else:  
                    try:  
                        subject = "Resume Screening Report"  
                        body = (  
                            f"Hello,\n\n"  
                            f"Attached is the resume screening CSV report.\n"  
                            f"Scan Date: {scan_date}\n\n"  
                            f"Regards,\nAI Resume Screener"  
                        )  
                        send_email_with_csv(  
                            csv_bytes=csv_bytes,  
                            to_email=report_email_to.strip(),  
                            subject=subject,  
                            body=body,  
                        )  
                        st.success(f"📧 CSV report emailed successfully to {report_email_to.strip()}")  
                    except Exception as e:  
                        st.error(f"Failed to send email: {e}")  
        else:  
            st.warning("No results to export.")
