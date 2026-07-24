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

# Weights:
# Required skills dominate the score.
# Preferred skills only add a small bonus.
REQUIRED_WEIGHT = 85
PREFERRED_WEIGHT = 15

# REVIEW logic thresholds
REVIEW_MIN_REQUIRED_COVERAGE = 0.60   # enough required skills to deserve review
MATCH_MIN_REQUIRED_COVERAGE = 1.00    # all required skills present => match

# ==========================================
# SMALL HELPERS
# ==========================================
def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()

def normalize_list(value):
    """
    Forces any AI output into a clean list of strings.
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
    allowed = {str(x).strip().lower(): str(x).strip() for x in allowed_list}
    out = []
    for item in normalize_list(source_list):
        key = item.lower()
        if key in allowed:
            out.append(allowed[key])
    return dedupe_preserve_order(out)

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

def classify_recommendation(final_decision, score):
    if final_decision == "MATCH":
        if score >= 90:
            return "Highly Recommended"
        return "Recommended"
    if final_decision == "REVIEW":
        return "Review"
    return "Not Recommended"

def confidence_label(score, final_decision=None):
    if score >= 85:
        return "high"
    if score >= 70:
        return "medium"
    return "low"

def parse_jd_sections(jd_text):
    """
    Heuristic parser to split JD into required and preferred skills.
    Works well for structured JDs with headings like:
    Required:, Preferred:, Good to have:, Nice to have:
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

            # Split only on "or" and "/" to avoid over-splitting phrases.
            parts = re.split(r"\s+(?:or|/)\s+", cleaned, flags=re.I)
            parts = [p.strip(" ,;.") for p in parts if p.strip(" ,;.")]

            if current == "required":
                required.extend(parts)
            else:
                preferred.extend(parts)

    required = dedupe_preserve_order(required)
    preferred = dedupe_preserve_order(preferred)
    return required, preferred

def compute_final_assessment(parsed, required_skills, preferred_skills):
    """
    Final score + decision is computed outside the model so preferred skills
    do not drag down the score too much.
    """
    matched_required = safe_intersection(parsed.get("matched_required_skills", []), required_skills)
    missing_required = safe_intersection(parsed.get("missing_required_skills", []), required_skills)
    matched_preferred = safe_intersection(parsed.get("matched_preferred_skills", []), preferred_skills)
    missing_preferred = safe_intersection(parsed.get("missing_preferred_skills", []), preferred_skills)

    # Fallbacks when model returns incomplete lists
    if required_skills and not matched_required and parsed.get("matched_skills"):
        matched_required = safe_intersection(parsed.get("matched_skills", []), required_skills)
    if preferred_skills and not matched_preferred and parsed.get("matched_skills"):
        matched_preferred = safe_intersection(parsed.get("matched_skills", []), preferred_skills)

    # Derive missing lists from required/preferred pools when needed
    if required_skills:
        if not missing_required:
            missing_required = [s for s in required_skills if s.lower() not in {x.lower() for x in matched_required}]
    if preferred_skills:
        if not missing_preferred:
            missing_preferred = [s for s in preferred_skills if s.lower() not in {x.lower() for x in matched_preferred}]

    req_total = len(required_skills)
    pref_total = len(preferred_skills)

    req_coverage = (len(matched_required) / req_total) if req_total else 1.0
    pref_coverage = (len(matched_preferred) / pref_total) if pref_total else 1.0

    # Weighted score:
    # - Required skills dominate
    # - Preferred skills add only a small bonus
    score = round((req_coverage * REQUIRED_WEIGHT) + (pref_coverage * PREFERRED_WEIGHT))

    # Decision logic:
    # - All required skills => Match
    # - Most required skills but some gap => Review
    # - Too many gaps => Rejected
    if req_total == 0:
        # If JD parsing fails, keep a safer fallback
        final_decision = "REVIEW" if score >= 60 else "REJECTED"
    elif req_coverage >= MATCH_MIN_REQUIRED_COVERAGE:
        final_decision = "MATCH"
    elif req_coverage >= REVIEW_MIN_REQUIRED_COVERAGE:
        final_decision = "REVIEW"
    else:
        final_decision = "REJECTED"

    # Human-readable reason labels
    if final_decision == "MATCH":
        if pref_total and pref_coverage < 0.5:
            decision_reason = "All required skills are present; preferred skills are partially missing but do not block the match."
        else:
            decision_reason = "All required skills are present."
    elif final_decision == "REVIEW":
        missing_msg = ", ".join(missing_required[:3]) if missing_required else "some required skills"
        decision_reason = f"The profile is close, but still has gaps in required skills such as {missing_msg}."
    else:
        missing_msg = ", ".join(missing_required[:3]) if missing_required else "multiple required skills"
        decision_reason = f"The profile is missing too many required skills, especially {missing_msg}."

    return {
        "match_score": score,
        "required_coverage": round(req_coverage * 100, 1),
        "preferred_coverage": round(pref_coverage * 100, 1),
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "final_decision": final_decision,
        "decision_reason": decision_reason,
    }

def evaluate_resume(api_keys, resume_text, jd_text, required_skills, preferred_skills, model_name):
    system_prompt = """
You are an expert recruiter screening a resume against a job description.

Strict rules:
- Only use skills explicitly supported by the resume text.
- Do not invent skills.
- Treat preferred skills as bonus only.
- Focus most on required skills.
- If a candidate is missing only some preferred skills, do not over-penalize them.
- If the candidate is missing one required skill but otherwise looks close, explain that this should be a REVIEW case, not an automatic hard reject.
- Return ONLY valid JSON. No markdown. No code fences.

Return these exact keys:
"candidate_name": string
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

Notes:
- Use the required/preferred skill lists supplied in the prompt.
- "matched_skills" can include all explicitly present matched skills from both required and preferred categories.
- If the result should be a review case, put the explanation in "why_review".
""".strip()

    user_prompt = f"""
Job Description:
{jd_text}

Required skills list:
{json.dumps(required_skills, ensure_ascii=False)}

Preferred skills list:
{json.dumps(preferred_skills, ensure_ascii=False)}

Resume:
{resume_text}
""".strip()

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
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}")
            if start_idx == -1 or end_idx == -1:
                raise ValueError(f"Model output did not contain valid JSON: {result_text}")

            clean_json_str = result_text[start_idx:end_idx + 1]
            parsed = json.loads(clean_json_str)

            model_payload = {
                "candidate_name": clean_text(parsed.get("candidate_name", "")),
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

            computed = compute_final_assessment(model_payload, required_skills, preferred_skills)

            final_decision = computed["final_decision"]
            score = computed["match_score"]

            return {
                **model_payload,
                **computed,
                "is_match": final_decision == "MATCH",
                "recommendation": classify_recommendation(final_decision, score),
                "confidence_level": model_payload["confidence_level"] or confidence_label(score, final_decision),
                "reasoning": "",
            }

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            last_error = f"System Error: {str(e)}"
            continue

    return {
        "candidate_name": "",
        "is_match": False,
        "match_score": 0,
        "required_coverage": 0.0,
        "preferred_coverage": 0.0,
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
        "recommendation": "Not Recommended",
        "final_decision": "REJECTED",
        "decision_reason": "",
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
required_skills, preferred_skills = parse_jd_sections(jd_text)

with st.expander("Parsed JD skills", expanded=False):
    st.write(f"**Required:** {', '.join(required_skills) if required_skills else 'Could not parse required skills automatically'}")
    st.write(f"**Preferred:** {', '.join(preferred_skills) if preferred_skills else 'Could not parse preferred skills automatically'}")

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
                evaluation = evaluate_resume(
                    api_keys=api_keys,
                    resume_text=resume_text,
                    jd_text=jd_text,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills,
                    model_name=model_name,
                )

            candidate_name = evaluation.get("candidate_name") or fallback_name
            score = int(evaluation.get("match_score", 0))
            final_decision = evaluation.get("final_decision", "REJECTED")
            recommendation = evaluation.get("recommendation", classify_recommendation(final_decision, score))
            confidence_level = evaluation.get("confidence_level", confidence_label(score, final_decision))

            matched_required = evaluation.get("matched_required_skills", [])
            missing_required = evaluation.get("missing_required_skills", [])
            matched_preferred = evaluation.get("matched_preferred_skills", [])
            missing_preferred = evaluation.get("missing_preferred_skills", [])

            why_matched = evaluation.get("why_matched", "")
            why_review = evaluation.get("why_review", "")
            why_not_matched = evaluation.get("why_not_matched", "")
            overall_summary = evaluation.get("overall_summary", "")

            status_icon = {"MATCH": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(final_decision, "❌")
            status_label = {"MATCH": "MATCH", "REVIEW": "REVIEW", "REJECTED": "REJECTED"}.get(final_decision, "REJECTED")

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
                        st.write(f"**Why Matched:** {why_matched or 'Details not provided'}")
                    elif final_decision == "REVIEW":
                        st.write(f"**Why Review:** {why_review or evaluation.get('decision_reason', 'Details not provided')}")
                    else:
                        st.write(f"**Why Not Matched:** {why_not_matched or evaluation.get('decision_reason', 'Details not provided')}")

                    st.write(f"**Summary:** {overall_summary or 'Not provided'}")

                    st.write(
                        f"**Matched Required Skills:** {', '.join(matched_required) if matched_required else 'None found'}"
                    )
                    st.write(
                        f"**Missing Required Skills:** {', '.join(missing_required) if missing_required else 'None found'}"
                    )
                    st.write(
                        f"**Matched Preferred Skills:** {', '.join(matched_preferred) if matched_preferred else 'None found'}"
                    )
                    st.write(
                        f"**Missing Preferred Skills:** {', '.join(missing_preferred) if missing_preferred else 'None found'}"
                    )

            rows.append({
                "scan_date": scan_date,
                "candidate_name": candidate_name,
                "file_name": file.name,
                "match_score": score,
                "required_coverage": evaluation.get("required_coverage", 0),
                "preferred_coverage": evaluation.get("preferred_coverage", 0),
                "final_decision": final_decision,
                "match_percentage": f"{score}%",
                "match_status": final_decision.title(),
                "recommendation": recommendation,
                "confidence_level": confidence_level,
                "matched_required_skills": ", ".join(matched_required) if matched_required else "",
                "missing_required_skills": ", ".join(missing_required) if missing_required else "",
                "matched_preferred_skills": ", ".join(matched_preferred) if matched_preferred else "",
                "missing_preferred_skills": ", ".join(missing_preferred) if missing_preferred else "",
                "why_matched": why_matched if final_decision == "MATCH" else "",
                "why_review": why_review if final_decision == "REVIEW" else "",
                "why_not_matched": why_not_matched if final_decision == "REJECTED" else "",
                "overall_summary": overall_summary,
            })

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
