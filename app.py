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
DEFAULT_MODEL = "openrouter/free"

REQUIRED_WEIGHT = 88
PREFERRED_WEIGHT = 12

REVIEW_REQUIRED_COVERAGE_MIN = 0.60  # 60%+ of required skills present => review if not full match

# ==========================================
# SMALL UTILITIES
# ==========================================
def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        item = clean_text(item)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

def normalize_list(value):
    """
    Normalize any AI output into a clean list[str].
    """
    if isinstance(value, list):
        return dedupe_preserve_order([str(x).strip() for x in value if str(x).strip()])
    if value is None:
        return []
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        return dedupe_preserve_order(parts)
    return [str(value).strip()]

def safe_intersection(source_list, allowed_list):
    allowed_map = {clean_text(x).lower(): clean_text(x) for x in allowed_list}
    out = []
    for item in normalize_list(source_list):
        key = item.lower()
        if key in allowed_map:
            out.append(allowed_map[key])
    return dedupe_preserve_order(out)

def parse_json_object(text):
    text = clean_text(text)
    text = text.replace("```json", "").replace("```", "").strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(f"Model output did not contain valid JSON: {text}")
    return json.loads(text[start_idx:end_idx + 1])

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

def confidence_label(score, final_decision):
    if final_decision == "MATCH" and score >= 90:
        return "high"
    if score >= 70 or final_decision == "REVIEW":
        return "medium"
    return "low"

def recommendation_from_decision(final_decision, score):
    if final_decision == "MATCH":
        if score >= 90:
            return "Highly Recommended"
        return "Recommended"
    if final_decision == "REVIEW":
        return "Review"
    return "Not Recommended"

def heuristic_jd_split(jd_text):
    """
    Hidden fallback for unstructured JDs.
    This is not shown in the frontend.
    """
    required = []
    preferred = []
    current = None

    for raw_line in jd_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower().rstrip(":").strip()

        if any(k in lower for k in ["required", "must have", "mandatory"]):
            current = "required"
            continue
        if any(k in lower for k in ["preferred", "good to have", "nice to have", "bonus"]):
            current = "preferred"
            continue

        if current in ("required", "preferred"):
            cleaned = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", line).strip()
            if not cleaned:
                continue
            parts = re.split(r"\s+(?:or|/)\s+", cleaned, flags=re.I)
            parts = [p.strip(" ,;.") for p in parts if p.strip(" ,;.")]
            if current == "required":
                required.extend(parts)
            else:
                preferred.extend(parts)

    return dedupe_preserve_order(required), dedupe_preserve_order(preferred)

def compute_final_assessment(model_data, jd_text):
    """
    Single source of truth for score + decision.
    Preferred skills contribute only a small bonus.
    Required skills drive the actual decision.
    """
    required_skills = normalize_list(model_data.get("required_skills", []))
    preferred_skills = normalize_list(model_data.get("preferred_skills", []))

    # Hidden fallback if the model fails to split the JD.
    if not required_skills and not preferred_skills:
        fallback_required, fallback_preferred = heuristic_jd_split(jd_text)
        required_skills = fallback_required
        preferred_skills = fallback_preferred

    matched_required = safe_intersection(model_data.get("matched_required_skills", []), required_skills)
    matched_preferred = safe_intersection(model_data.get("matched_preferred_skills", []), preferred_skills)

    if required_skills:
        missing_required = [s for s in required_skills if s.lower() not in {x.lower() for x in matched_required}]
    else:
        missing_required = normalize_list(model_data.get("missing_required_skills", []))

    if preferred_skills:
        missing_preferred = [s for s in preferred_skills if s.lower() not in {x.lower() for x in matched_preferred}]
    else:
        missing_preferred = normalize_list(model_data.get("missing_preferred_skills", []))

    # If the model returned matched_skills, keep it as a union fallback.
    matched_skills = normalize_list(model_data.get("matched_skills", []))
    if not matched_skills:
        matched_skills = dedupe_preserve_order(matched_required + matched_preferred)

    req_total = len(required_skills)
    pref_total = len(preferred_skills)

    req_coverage = (len(matched_required) / req_total) if req_total else 0.0
    pref_coverage = (len(matched_preferred) / pref_total) if pref_total else 0.0

    if req_total > 0 and pref_total > 0:
        score = round((req_coverage * REQUIRED_WEIGHT) + (pref_coverage * PREFERRED_WEIGHT))
    elif req_total > 0:
        score = round(req_coverage * 100)
    elif pref_total > 0:
        score = round(pref_coverage * 100)
    else:
        score = 0

    if req_total == 0:
        final_decision = "REVIEW" if score >= 60 else "REJECTED"
    elif len(missing_required) == 0:
        final_decision = "MATCH"
    elif req_coverage >= REVIEW_REQUIRED_COVERAGE_MIN:
        final_decision = "REVIEW"
    else:
        final_decision = "REJECTED"

    if final_decision == "MATCH":
        if pref_total and len(missing_preferred) > 0:
            decision_reason = "All required skills are present; preferred skills are partially missing but do not block the match."
        else:
            decision_reason = "All required skills are present."
    elif final_decision == "REVIEW":
        missing_text = ", ".join(missing_required[:4]) if missing_required else "some required skills"
        decision_reason = f"The profile is close, but still missing required skills such as {missing_text}."
    else:
        missing_text = ", ".join(missing_required[:4]) if missing_required else "multiple required skills"
        decision_reason = f"The profile is missing too many required skills, especially {missing_text}."

    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "matched_skills": matched_skills,
        "match_score": score,
        "required_coverage": round(req_coverage * 100, 1),
        "preferred_coverage": round(pref_coverage * 100, 1),
        "final_decision": final_decision,
        "is_match": final_decision == "MATCH",
        "decision_reason": decision_reason,
        "recommendation": recommendation_from_decision(final_decision, score),
        "confidence_level": confidence_label(score, final_decision),
    }

def evaluate_resume(api_keys, resume_text, jd_text, model_name):
    system_prompt = """
You are an expert recruiter.

Your job:
- Read the job description carefully.
- Identify which skills are required and which are preferred.
- Compare the resume against the JD.
- Do not assume a skill unless it is explicitly supported by the resume text.
- Treat preferred skills as bonus only.
- If the resume is close but missing one or two required items, describe it as a REVIEW case.
- Return ONLY valid JSON. No markdown. No code fences.

Return exactly these keys:
"candidate_name": string
"required_skills": array of strings
"preferred_skills": array of strings
"matched_required_skills": array of strings
"missing_required_skills": array of strings
"matched_preferred_skills": array of strings
"missing_preferred_skills": array of strings
"matched_skills": array of strings
"why_matched": string
"why_review": string
"why_not_matched": string
"overall_summary": string
"confidence_level": string ("low", "medium", or "high")

Rules:
- Only include skills that are supported by the resume.
- Keep the lists concise and job-relevant.
- If the result is a review case, put the explanation in "why_review".
- If the result is a match, leave "why_review" and "why_not_matched" empty.
- If the result is a reject, leave "why_matched" and "why_review" empty.
""".strip()

    user_prompt = f"""Job Description:
{jd_text}

Resume:
{resume_text}"""

    last_error = ""

    for key in api_keys:
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
                continue

            data = response.json()
            result_text = data["choices"][0]["message"]["content"].strip()
            parsed = parse_json_object(result_text)

            model_data = {
                "candidate_name": clean_text(parsed.get("candidate_name", "")),
                "required_skills": normalize_list(parsed.get("required_skills", [])),
                "preferred_skills": normalize_list(parsed.get("preferred_skills", [])),
                "matched_required_skills": normalize_list(parsed.get("matched_required_skills", [])),
                "missing_required_skills": normalize_list(parsed.get("missing_required_skills", [])),
                "matched_preferred_skills": normalize_list(parsed.get("matched_preferred_skills", [])),
                "missing_preferred_skills": normalize_list(parsed.get("missing_preferred_skills", [])),
                "matched_skills": normalize_list(parsed.get("matched_skills", [])),
                "why_matched": clean_text(parsed.get("why_matched", "")),
                "why_review": clean_text(parsed.get("why_review", "")),
                "why_not_matched": clean_text(parsed.get("why_not_matched", "")),
                "overall_summary": clean_text(parsed.get("overall_summary", "")),
                "confidence_level": clean_text(parsed.get("confidence_level", "")),
            }

            computed = compute_final_assessment(model_data, jd_text)
            confidence = model_data["confidence_level"] or computed["confidence_level"]

            return {
                **model_data,
                **computed,
                "confidence_level": confidence,
                "reasoning": "",
            }

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            last_error = f"System Error: {str(e)}"
            continue

    return {
        "candidate_name": "",
        "required_skills": [],
        "preferred_skills": [],
        "matched_required_skills": [],
        "missing_required_skills": [],
        "matched_preferred_skills": [],
        "missing_preferred_skills": [],
        "matched_skills": [],
        "why_matched": "",
        "why_review": "",
        "why_not_matched": "",
        "overall_summary": "",
        "confidence_level": "low",
        "match_score": 0,
        "required_coverage": 0.0,
        "preferred_coverage": 0.0,
        "final_decision": "REJECTED",
        "is_match": False,
        "decision_reason": "",
        "recommendation": "Not Recommended",
        "reasoning": f"All OpenRouter API keys failed. Last error: {last_error}",
    }

def build_csv_bytes(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "rank",
        "scan_date",
        "candidate_name",
        "file_name",
        "match_score",
        "required_coverage",
        "preferred_coverage",
        "final_decision",
        "match_percentage",
        "match_status",
        "recommendation",
        "confidence_level",
        "matched_required_skills",
        "missing_required_skills",
        "matched_preferred_skills",
        "missing_preferred_skills",
        "matched_skills",
        "why_matched",
        "why_review",
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
            final_decision = evaluation.get("final_decision", "REJECTED")
            recommendation = evaluation.get("recommendation", recommendation_from_decision(final_decision, score))
            confidence_level = evaluation.get("confidence_level", confidence_label(score, final_decision))

            matched_required_skills = evaluation.get("matched_required_skills", [])
            missing_required_skills = evaluation.get("missing_required_skills", [])
            matched_preferred_skills = evaluation.get("matched_preferred_skills", [])
            missing_preferred_skills = evaluation.get("missing_preferred_skills", [])
            matched_skills = evaluation.get("matched_skills", [])

            why_matched = evaluation.get("why_matched", "")
            why_review = evaluation.get("why_review", "")
            why_not_matched = evaluation.get("why_not_matched", "")
            overall_summary = evaluation.get("overall_summary", "")

            status_map = {
                "MATCH": ("✅", "MATCH"),
                "REVIEW": ("⚠️", "REVIEW"),
                "REJECTED": ("❌", "REJECTED"),
            }
            status_icon, status_label = status_map.get(final_decision, ("❌", "REJECTED"))

            expander_title = f"{status_icon} {status_label} - {candidate_name} ({score}/100)"

            with st.expander(expander_title, expanded=(final_decision == "MATCH")):
                if evaluation.get("reasoning"):
                    st.warning("⚠️ Error while processing this resume.")
                    st.write(f"**Details:** {evaluation.get('reasoning')}")
                else:
                    st.write(f"**Candidate Name:** {candidate_name}")
                    st.write(f"**Final Decision:** {final_decision}")
                    st.write(f"**Recommendation:** {recommendation}")
                    st.write(f"**Confidence:** {confidence_level}")
                    st.write(f"**Required Coverage:** {evaluation.get('required_coverage', 0)}%")
                    st.write(f"**Preferred Coverage:** {evaluation.get('preferred_coverage', 0)}%")

                    if final_decision == "MATCH":
                        st.write(f"**Why Matched:** {why_matched or evaluation.get('decision_reason', 'Details not provided')}")
                    elif final_decision == "REVIEW":
                        st.write(f"**Why Review:** {why_review or evaluation.get('decision_reason', 'Details not provided')}")
                    else:
                        st.write(f"**Why Not Matched:** {why_not_matched or evaluation.get('decision_reason', 'Details not provided')}")

                    st.write(f"**Summary:** {overall_summary or 'Not provided'}")

                    st.write(f"**Matched Required Skills:** {', '.join(matched_required_skills) if matched_required_skills else 'None found'}")
                    st.write(f"**Missing Required Skills:** {', '.join(missing_required_skills) if missing_required_skills else 'None found'}")
                    st.write(f"**Matched Preferred Skills:** {', '.join(matched_preferred_skills) if matched_preferred_skills else 'None found'}")
                    st.write(f"**Missing Preferred Skills:** {', '.join(missing_preferred_skills) if missing_preferred_skills else 'None found'}")
                    st.write(f"**Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None found'}")

            rows.append({
                "scan_date": scan_date,
                "candidate_name": candidate_name,
                "file_name": file.name,
                "match_score": score,
                "required_coverage": evaluation.get("required_coverage", 0),
                "preferred_coverage": evaluation.get("preferred_coverage", 0),
                "final_decision": final_decision,
                "match_percentage": f"{score}%",
                "match_status": final_decision,
                "recommendation": recommendation,
                "confidence_level": confidence_level,
                "matched_required_skills": ", ".join(matched_required_skills) if matched_required_skills else "",
                "missing_required_skills": ", ".join(missing_required_skills) if missing_required_skills else "",
                "matched_preferred_skills": ", ".join(matched_preferred_skills) if matched_preferred_skills else "",
                "missing_preferred_skills": ", ".join(missing_preferred_skills) if missing_preferred_skills else "",
                "matched_skills": ", ".join(matched_skills) if matched_skills else "",
                "why_matched": why_matched if final_decision == "MATCH" else "",
                "why_review": why_review if final_decision == "REVIEW" else "",
                "why_not_matched": why_not_matched if final_decision == "REJECTED" else "",
                "overall_summary": overall_summary,
            })

            # Time limit added here to prevent hitting API rate limits
            time.sleep(5)

            progress.progress(idx / total_files)

        # Highest score first
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
