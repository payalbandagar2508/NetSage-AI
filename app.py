import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NetSage AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cases.csv"
PROMPT_PATH = BASE_DIR / "diagnose_prompt.md"


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if GROQ_API_KEY:
    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
else:
    groq_client = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b1120;
        color: white;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1400px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* Sidebar title */
    .sidebar-logo {
        font-size: 28px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.5;
    }

    .sidebar-section {
        color: #60a5fa;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Header */
    .page-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .page-title span {
        color: #2196f3;
    }

    .page-subtitle {
        font-size: 17px;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        min-height: 125px;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: white;
        font-size: 34px;
        font-weight: 800;
    }

    /* Workflow */
    .workflow-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 22px;
        height: 100%;
        text-align: center;
    }

    .workflow-number {
        font-size: 14px;
        color: #60a5fa;
        font-weight: 700;
    }

    .workflow-title {
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin-top: 8px;
    }

    .workflow-text {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 8px;
    }

    /* Status boxes */
    .pass-box {
        background: #123d34;
        border: 1px solid #17634f;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
    }

    .warn-box {
        background: #403f18;
        border: 1px solid #615e1d;
        border-left: 5px solid #eab308;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
    }

    .fail-box {
        background: #461f2b;
        border: 1px solid #713142;
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
    }

    .info-box {
        background: #172554;
        border: 1px solid #1d4ed8;
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
    }

    /* Section heading */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: white;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 12px;
        right: 25px;
        color: #64748b;
        font-size: 13px;
        z-index: 999;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
    }

    /* Hide Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CASES
# ============================================================

@st.cache_data
def load_cases():

    if not DATA_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(DATA_PATH)

    except Exception as e:
        st.error(f"Unable to load cases.csv: {e}")
        return pd.DataFrame()


cases_df = load_cases()


# ============================================================
# LOAD PROMPT
# ============================================================

def load_prompt():

    if not PROMPT_PATH.exists():

        return """
You are NetSage AI, a network troubleshooting assistant.

Analyze only the supplied network evidence.

Do not invent:
- IP addresses
- interfaces
- VLANs
- routes
- configurations
- command outputs

Explain the likely root cause and provide practical
troubleshooting actions.

Return valid JSON.
"""

    try:

        return PROMPT_PATH.read_text(
            encoding="utf-8"
        ).strip()

    except Exception:

        return """
You are a network troubleshooting assistant.
Analyze only the supplied evidence.
Do not invent network information.
Return a clear JSON diagnosis.
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    for col in df.columns:

        clean = (
            str(col)
            .lower()
            .strip()
            .replace(" ", "_")
        )

        for name in possible_names:

            if clean == name:
                return col

    return None


def get_case_id(row, index):

    column = find_column(
        cases_df,
        ["case_id", "id", "case"]
    )

    if column:

        value = row[column]

        if not pd.isna(value):
            return str(value)

    return f"CASE_{index + 1:03d}"


def get_case_text(row):

    parts = []

    for column in cases_df.columns:

        value = row[column]

        if not pd.isna(value):

            parts.append(
                f"{column}: {value}"
            )

    return "\n".join(parts)


# ============================================================
# RULE CHECKER
# ============================================================

def run_rule_checks(row):

    evidence = get_case_text(row)
    text = evidence.lower()

    results = []

    # --------------------------------------------------------
    # Interface
    # --------------------------------------------------------

    if (
        "interface down" in text
        or "interface is down" in text
        or "administratively down" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "Interface Status",
            "message": "Interface appears to be down."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "Interface Status",
            "message": "Interfaces appear operational."
        })

    # --------------------------------------------------------
    # DHCP
    # --------------------------------------------------------

    if (
        "169.254." in text
        or "apipa" in text
        or "dhcp failure" in text
        or "dhcp not working" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "DHCP & APIPA Validation",
            "message": "Possible DHCP or APIPA problem detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "DHCP & APIPA Validation",
            "message": "No DHCP/APIPA issue detected."
        })

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    if (
        "no route" in text
        or "missing route" in text
        or "routing failure" in text
        or "routing problem" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "Routing",
            "message": "Possible missing or incorrect route detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "Routing",
            "message": "No obvious routing failure detected."
        })

    # --------------------------------------------------------
    # DNS
    # --------------------------------------------------------

    if (
        "dns failure" in text
        or "dns not working" in text
        or "cannot resolve" in text
        or "dns error" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "DNS",
            "message": "Possible DNS resolution problem detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "DNS",
            "message": "No obvious DNS failure detected."
        })

    # --------------------------------------------------------
    # VLAN
    # --------------------------------------------------------

    if (
        "vlan mismatch" in text
        or "wrong vlan" in text
        or "incorrect vlan" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "VLAN",
            "message": "Possible VLAN configuration mismatch detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "VLAN",
            "message": "No obvious VLAN mismatch detected."
        })

    # --------------------------------------------------------
    # ACL
    # --------------------------------------------------------

    if (
        "acl deny" in text
        or "access-list deny" in text
        or "blocked by acl" in text
        or "acl blocking" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "ACL",
            "message": "Possible ACL blocking traffic detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "ACL",
            "message": "No obvious ACL blocking detected."
        })

    # --------------------------------------------------------
    # NAT
    # --------------------------------------------------------

    if (
        "nat failure" in text
        or "nat not working" in text
        or "nat translation" in text
    ):

        results.append({
            "status": "FAIL",
            "check": "NAT",
            "message": "Possible NAT configuration problem detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "NAT",
            "message": "No obvious NAT problem detected."
        })

    # --------------------------------------------------------
    # Subnet
    # --------------------------------------------------------

    if (
        "mask mismatch" in text
        or "subnet mask mismatch" in text
        or "subnet mismatch" in text
    ):

        results.append({
            "status": "WARN",
            "check": "Subnet Mask Validation",
            "message": "Subnet mask mismatch detected."
        })

    else:

        results.append({
            "status": "PASS",
            "check": "Subnet Mask Validation",
            "message": "Subnet mask appears consistent."
        })

    return results


# ============================================================
# RULE STATUS
# ============================================================

def get_rule_summary(results):

    passed = sum(
        1 for r in results
        if r["status"] == "PASS"
    )

    failed = sum(
        1 for r in results
        if r["status"] == "FAIL"
    )

    warnings = sum(
        1 for r in results
        if r["status"] == "WARN"
    )

    return passed, failed, warnings


# ============================================================
# GROQ AI DIAGNOSIS
# ============================================================

def run_ai_diagnosis(
    case_id,
    case_information,
    rule_results
):

    if groq_client is None:

        return {
            "error":
            "Groq API key is not configured. "
            "Please add GROQ_API_KEY to the .env file."
        }

    system_prompt = load_prompt()

    user_prompt = f"""
CASE ID:
{case_id}

NETWORK CASE INFORMATION:
{case_information}

DETERMINISTIC RULE CHECK RESULTS:
{json.dumps(rule_results, indent=2)}

IMPORTANT:

1. Analyze only the supplied evidence.
2. Do not invent IP addresses.
3. Do not invent interfaces.
4. Do not invent VLANs.
5. Do not invent routes.
6. Do not invent configurations.
7. Respect deterministic checker results.
8. Explain the most likely root cause.
9. Provide practical recommended actions.
10. Provide verification commands.
11. If evidence is insufficient, set needs_human_review to true.
12. Return ONLY valid JSON.

Use exactly this structure:

{{
    "case_id": "{case_id}",
    "diagnosis": "short diagnosis",
    "fault_category": "VLAN/DHCP/DNS/Routing/ACL/NAT/Wireless/Unknown",
    "severity": "Low/Medium/High/Critical/Unknown",
    "confidence": 0.0,
    "root_cause": "root cause",
    "explanation": "clear explanation",
    "recommended_actions": [],
    "verification_commands": [],
    "evidence": [],
    "deterministic_status": "PASS/FAIL/WARN/NOT_AVAILABLE",
    "needs_human_review": false
}}
"""

    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.2,

            response_format={
                "type": "json_object"
            }
        )

        if not response.choices:

            return {
                "error": "Groq returned no response."
            }

        content = response.choices[0].message.content

        if not content:

            return {
                "error": "Groq returned an empty response."
            }

        result = json.loads(content)

        return result

    except json.JSONDecodeError:

        return {
            "error": "Groq returned invalid JSON."
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# SESSION STATE
# ============================================================

if "selected_case" not in st.session_state:
    st.session_state.selected_case = 0

if "rule_completed" not in st.session_state:
    st.session_state.rule_completed = False

if "rule_results" not in st.session_state:
    st.session_state.rule_results = []

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            🌐 NetSage AI
        </div>
        <div class="sidebar-subtitle">
            Network Troubleshooting Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section">Main Menu</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Diagnose Case",
            "📋 Rule Checker",
            "📁 Network Cases",
            "👤 Human Review",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section">System Status</div>',
        unsafe_allow_html=True
    )

    if GROQ_API_KEY:

        st.success("🟢 Groq AI Connected")

    else:

        st.error("🔴 Groq API Key Missing")

    st.markdown(
        f"""
        <div class="card" style="padding:15px;">
            <b>Network Cases</b><br>
            <span style="font-size:24px;">
                {len(cases_df)}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption("CISCO Network Troubleshooting Project")


# ============================================================
# COMMON HEADER
# ============================================================

st.markdown(
    """
    <div class="page-title">
        NetSage <span>AI</span>
    </div>

    <div class="page-subtitle">
        Network Troubleshooting Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Dashboard</div>',
        unsafe_allow_html=True
    )

    total_cases = len(cases_df)

    total_checks = 0
    total_pass = 0
    total_fail = 0
    total_warn = 0

    if not cases_df.empty:

        for _, row in cases_df.iterrows():

            results = run_rule_checks(row)

            total_checks += len(results)

            for result in results:

                if result["status"] == "PASS":
                    total_pass += 1

                elif result["status"] == "FAIL":
                    total_fail += 1

                elif result["status"] == "WARN":
                    total_warn += 1

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Network Cases
                </div>
                <div class="metric-value">
                    {total_cases}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Total Checks
                </div>
                <div class="metric-value">
                    {total_checks}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    PASS
                </div>
                <div class="metric-value">
                    {total_pass}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    FAIL
                </div>
                <div class="metric-value">
                    {total_fail}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🚀 System Workflow</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The system follows a controlled troubleshooting process. "
        "Rule checking is completed before AI diagnosis."
    )

    w1, w2, w3, w4, w5 = st.columns(5)

    with w1:

        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-number">STEP 1</div>
                <div class="workflow-title">
                    📁 Network Evidence
                </div>
                <div class="workflow-text">
                    Select a network troubleshooting case.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w2:

        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-number">STEP 2</div>
                <div class="workflow-title">
                    🔎 Rule Checker
                </div>
                <div class="workflow-text">
                    Python checks the supplied evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w3:

        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-number">STEP 3</div>
                <div class="workflow-title">
                    ✅ Verified Facts
                </div>
                <div class="workflow-text">
                    Deterministic results become verified evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w4:

        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-number">STEP 4</div>
                <div class="workflow-title">
                    🤖 Groq AI Diagnosis
                </div>
                <div class="workflow-text">
                    AI analyzes the verified evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w5:

        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-number">STEP 5</div>
                <div class="workflow-title">
                    👤 Human Review
                </div>
                <div class="workflow-text">
                    Expert validates the final solution.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DIAGNOSE CASE
# ============================================================

elif page == "🔍 Diagnose Case":

    st.markdown(
        '<div class="section-title">🔍 Diagnose Network Case</div>',
        unsafe_allow_html=True
    )

    if cases_df.empty:

        st.error(
            "No network cases found. "
            "Please check data/cases.csv."
        )

    else:

        # ----------------------------------------------------
        # CASE SELECTION
        # ----------------------------------------------------

        case_options = []

        for i, row in cases_df.iterrows():

            case_options.append(
                get_case_id(row, i)
            )

        selected_case = st.selectbox(
            "Select a Network Case",
            range(len(case_options)),
            format_func=lambda x: case_options[x],
            key="diagnose_case_selector"
        )

        selected_row = cases_df.iloc[selected_case]

        case_id = get_case_id(
            selected_row,
            selected_case
        )

        # Reset workflow if case changes
        if st.session_state.selected_case != selected_case:

            st.session_state.selected_case = selected_case
            st.session_state.rule_completed = False
            st.session_state.rule_results = []
            st.session_state.ai_result = None

        st.markdown(
            f"""
            <div class="card">
                <h3>📁 {case_id}</h3>
                <p style="color:#94a3b8;">
                    Network evidence loaded successfully.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # NETWORK EVIDENCE
        # ----------------------------------------------------

        with st.expander(
            "📄 View Network Evidence",
            expanded=False
        ):

            for column in cases_df.columns:

                value = selected_row[column]

                if not pd.isna(value):

                    st.markdown(
                        f"**{column}**"
                    )

                    st.write(str(value))

        # ----------------------------------------------------
        # STEP 1 - RULE CHECKER
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">1️⃣ Rule Checker</div>',
            unsafe_allow_html=True
        )

        st.info(
            "First, run the deterministic Python Rule Checker. "
            "AI diagnosis becomes available only after the "
            "network evidence has been checked."
        )

        if st.button(
            "🔎 Run Rule Checker",
            type="primary",
            use_container_width=False
        ):

            with st.spinner(
                "Checking network evidence..."
            ):

                results = run_rule_checks(
                    selected_row
                )

            st.session_state.rule_results = results
            st.session_state.rule_completed = True
            st.session_state.ai_result = None

            st.success(
                "Rule checking completed successfully."
            )

        # ----------------------------------------------------
        # RULE RESULTS
        # ----------------------------------------------------

        if st.session_state.rule_completed:

            rule_results = st.session_state.rule_results

            passed, failed, warnings = get_rule_summary(
                rule_results
            )

            a, b, c = st.columns(3)

            with a:
                st.metric("PASS", passed)

            with b:
                st.metric("FAIL", failed)

            with c:
                st.metric("WARN", warnings)

            for result in rule_results:

                status = result["status"]

                if status == "PASS":

                    st.markdown(
                        f"""
                        <div class="pass-box">
                            🟢 <b>PASS — {result["check"]}</b>
                            <br><br>
                            {result["message"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif status == "WARN":

                    st.markdown(
                        f"""
                        <div class="warn-box">
                            🟡 <b>WARN — {result["check"]}</b>
                            <br><br>
                            {result["message"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="fail-box">
                            🔴 <b>FAIL — {result["check"]}</b>
                            <br><br>
                            {result["message"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ------------------------------------------------
            # VERIFIED FACTS
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">2️⃣ Verified Facts</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="info-box">
                    <b>Evidence Verification Complete</b>
                    <br><br>
                    The deterministic rule checker has completed.
                    These results will now be supplied to Groq AI
                    for diagnosis.
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # AI DIAGNOSIS
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">3️⃣ AI Diagnosis</div>',
                unsafe_allow_html=True
            )

            if not GROQ_API_KEY:

                st.error(
                    "Groq API key is missing. "
                    "Add GROQ_API_KEY to your .env file."
                )

            else:

                st.success(
                    "✅ Rule Checker completed. "
                    "AI Diagnosis is now ready."
                )

                if st.button(
                    "🤖 Run Groq AI Diagnosis",
                    type="primary"
                ):

                    with st.spinner(
                        "Groq AI is analyzing the verified network evidence..."
                    ):

                        result = run_ai_diagnosis(
                            case_id,
                            get_case_text(selected_row),
                            rule_results
                        )

                    st.session_state.ai_result = result

            # ------------------------------------------------
            # AI RESULT
            # ------------------------------------------------

            result = st.session_state.ai_result

            if result:

                if "error" in result:

                    st.error(
                        "AI Diagnosis Failed: "
                        + str(result["error"])
                    )

                else:

                    st.success(
                        "🎉 AI diagnosis completed successfully."
                    )

                    # Diagnosis
                    st.markdown(
                        '<div class="section-title">📌 Diagnosis</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>
                                {result.get(
                                    "diagnosis",
                                    "Not available"
                                )}
                            </h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Metrics
                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Fault Category",
                            result.get(
                                "fault_category",
                                "Unknown"
                            )
                        )

                    with c2:

                        st.metric(
                            "Severity",
                            result.get(
                                "severity",
                                "Unknown"
                            )
                        )

                    with c3:

                        confidence = result.get(
                            "confidence",
                            0
                        )

                        try:
                            confidence = float(
                                confidence
                            )
                        except:
                            confidence = 0

                        if confidence <= 1:
                            confidence *= 100

                        st.metric(
                            "Confidence",
                            f"{confidence:.1f}%"
                        )

                    # Root cause
                    st.markdown(
                        '<div class="section-title">🎯 Root Cause</div>',
                        unsafe_allow_html=True
                    )

                    st.info(
                        result.get(
                            "root_cause",
                            "Not available"
                        )
                    )

                    # Explanation
                    st.markdown(
                        '<div class="section-title">📖 Explanation</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        result.get(
                            "explanation",
                            "Not available"
                        )
                    )

                    # Actions
                    st.markdown(
                        '<div class="section-title">🛠️ Recommended Actions</div>',
                        unsafe_allow_html=True
                    )

                    actions = result.get(
                        "recommended_actions",
                        []
                    )

                    if isinstance(actions, list):

                        for i, action in enumerate(
                            actions,
                            start=1
                        ):

                            st.write(
                                f"**{i}.** {action}"
                            )

                    else:

                        st.write(actions)

                    # Verification
                    st.markdown(
                        '<div class="section-title">✅ Verification Steps</div>',
                        unsafe_allow_html=True
                    )

                    commands = result.get(
                        "verification_commands",
                        []
                    )

                    if isinstance(commands, list):

                        for command in commands:

                            st.write(
                                f"• {command}"
                            )

                    else:

                        st.write(commands)

                    # Evidence
                    st.markdown(
                        '<div class="section-title">🔎 Evidence Used</div>',
                        unsafe_allow_html=True
                    )

                    evidence = result.get(
                        "evidence",
                        []
                    )

                    if isinstance(evidence, list):

                        for item in evidence:

                            if isinstance(item, dict):

                                source = item.get(
                                    "source",
                                    "Evidence"
                                )

                                detail = item.get(
                                    "detail",
                                    ""
                                )

                                st.write(
                                    f"**{source}:** {detail}"
                                )

                            else:

                                st.write(
                                    f"• {item}"
                                )

                    # Human review
                    if result.get(
                        "needs_human_review",
                        False
                    ):

                        st.warning(
                            "⚠️ Human review is required "
                            "before accepting the solution."
                        )

                    else:

                        st.success(
                            "✅ AI diagnosis can proceed "
                            "to human review."
                        )


# ============================================================
# RULE CHECKER PAGE
# ============================================================

elif page == "📋 Rule Checker":

    st.markdown(
        '<div class="section-title">📋 Rule Checker</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Run deterministic network checks before using AI diagnosis."
    )

    if cases_df.empty:

        st.warning("No cases available.")

    else:

        selected = st.selectbox(
            "Select Case",
            range(len(cases_df)),
            format_func=lambda x:
                get_case_id(
                    cases_df.iloc[x],
                    x
                )
        )

        row = cases_df.iloc[selected]

        if st.button(
            "🔎 Run Rule Checker",
            type="primary"
        ):

            results = run_rule_checks(row)

            st.session_state.rule_results = results
            st.session_state.rule_completed = True

        if st.session_state.rule_completed:

            for result in st.session_state.rule_results:

                if result["status"] == "PASS":

                    st.success(
                        f"PASS — {result['check']}: "
                        f"{result['message']}"
                    )

                elif result["status"] == "WARN":

                    st.warning(
                        f"WARN — {result['check']}: "
                        f"{result['message']}"
                    )

                else:

                    st.error(
                        f"FAIL — {result['check']}: "
                        f"{result['message']}"
                    )


# ============================================================
# NETWORK CASES
# ============================================================

elif page == "📁 Network Cases":

    st.markdown(
        '<div class="section-title">📁 Network Cases</div>',
        unsafe_allow_html=True
    )

    if cases_df.empty:

        st.warning(
            "cases.csv was not found."
        )

    else:

        st.info(
            f"Total Network Cases: {len(cases_df)}"
        )

        st.dataframe(
            cases_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# HUMAN REVIEW
# ============================================================

elif page == "👤 Human Review":

    st.markdown(
        '<div class="section-title">👤 Human Review</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Review the AI diagnosis and validate the proposed solution."
    )

    st.markdown(
        """
        <div class="info-box">
            <b>Human Review Process</b>
            <br><br>
            The network engineer reviews the AI diagnosis,
            recommended actions and evidence before accepting
            the final solution.
        </div>
        """,
        unsafe_allow_html=True
    )

    review = st.radio(
        "Review Decision",
        [
            "Pending",
            "Approved",
            "Needs Correction"
        ]
    )

    comments = st.text_area(
        "Reviewer Comments",
        placeholder="Enter your review comments..."
    )

    if st.button(
        "Submit Review",
        type="primary"
    ):

        st.success(
            "Human review submitted successfully."
        )

        st.write(
            f"**Decision:** {review}"
        )

        if comments:

            st.write(
                f"**Comments:** {comments}"
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">ℹ️ About NetSage AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>🌐 NetSage AI</h2>

        <p>
        NetSage AI is a Network Troubleshooting Intelligence
        Platform designed to assist engineers in identifying
        common network problems.
        </p>

        <h3>System Process</h3>

        <p>
        📁 Network Evidence
        →
        🔎 Rule Checker
        →
        ✅ Verified Facts
        →
        🤖 Groq AI Diagnosis
        →
        👤 Human Review
        →
        ✅ Final Solution
        </p>

        <h3>Technologies</h3>

        <p>
        Python • Streamlit • Pandas • Groq API • CSV
        </p>

        <h3>Supported Network Problems</h3>

        <p>
        VLAN • DHCP • DNS • Routing • ACL • NAT • Wireless
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Created by <b>Payal Bandagar</b>
    </div>
    """,
    unsafe_allow_html=True
)