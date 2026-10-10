"""
Personal Document Finder — Streamlit UI (Phase 13)

Full-featured interface for:
  - Login / Register
  - Document upload & list view with category/status badges
  - Chat-style Q&A (hitting /api/v1/query)
  - Document comparison
  - Timeline visualization
  - Alerts / Insights dashboard
  - Knowledge Graph viewer

All UI calls go through the REST API with the user's JWT.
"""

import json
import requests
import streamlit as st
from datetime import datetime

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Personal Document Finder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS for a polished look
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Cards */
    .doc-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .doc-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-ready { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-processing { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-uploaded { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-failed { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Category badges */
    .badge-insurance { background: rgba(139, 92, 246, 0.2); color: #8b5cf6; }
    .badge-banking { background: rgba(14, 165, 233, 0.2); color: #0ea5e9; }
    .badge-government { background: rgba(234, 179, 8, 0.2); color: #eab308; }
    .badge-property { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
    .badge-tax { background: rgba(249, 115, 22, 0.2); color: #f97316; }
    .badge-medical { background: rgba(236, 72, 153, 0.2); color: #ec4899; }

    /* Alert severity */
    .alert-critical { border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.08); padding: 12px; border-radius: 8px; margin: 8px 0; }
    .alert-warning { border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.08); padding: 12px; border-radius: 8px; margin: 8px 0; }
    .alert-info { border-left: 4px solid #3b82f6; background: rgba(59, 130, 246, 0.08); padding: 12px; border-radius: 8px; margin: 8px 0; }

    /* Chat messages */
    .chat-user {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .chat-assistant {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin: 8px 0;
    }

    /* Timeline item */
    .timeline-item {
        position: relative;
        padding: 12px 12px 12px 28px;
        margin: 4px 0;
        border-left: 2px solid rgba(99, 102, 241, 0.4);
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -5px;
        top: 18px;
        width: 8px;
        height: 8px;
        background: #6366f1;
        border-radius: 50%;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85em;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 4px;
    }

    /* Header styling */
    .main-header {
        font-size: 2.2em;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    .sub-header {
        color: rgba(255, 255, 255, 0.5);
        font-size: 1em;
        margin-bottom: 2em;
    }

    /* KG nodes */
    .kg-node {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85em;
        cursor: default;
    }
    .kg-document { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .kg-person { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .kg-policy { background: rgba(139, 92, 246, 0.2); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.3); }
    .kg-entity { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None


# ------------------------------------------------------------------
# API Helpers
# ------------------------------------------------------------------
def api_headers():
    """Return auth headers for API calls."""
    if st.session_state.jwt_token:
        return {"Authorization": f"Bearer {st.session_state.jwt_token}"}
    return {}


def api_get(path, params=None):
    """GET request to the API."""
    try:
        r = requests.get(f"{API_BASE}{path}", headers=api_headers(), params=params, timeout=30)
        return r
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the API server. Make sure it's running on port 8000.")
        return None


def api_post(path, json_data=None, files=None):
    """POST request to the API."""
    try:
        r = requests.post(f"{API_BASE}{path}", headers=api_headers(), json=json_data, files=files, timeout=60)
        return r
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the API server.")
        return None


def api_post_files(path, files):
    """POST request with file upload."""
    try:
        headers = {}
        if st.session_state.jwt_token:
            headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
        r = requests.post(f"{API_BASE}{path}", headers=headers, files=files, timeout=120)
        return r
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the API server.")
        return None


# ------------------------------------------------------------------
# Status / Category badge helpers
# ------------------------------------------------------------------
def status_badge(status_val):
    badge_class = {
        "ready": "badge-ready",
        "processing": "badge-processing",
        "uploaded": "badge-uploaded",
        "failed": "badge-failed",
    }.get(status_val, "badge-uploaded")
    return f'<span class="badge {badge_class}">{status_val}</span>'


def category_badge(category_val):
    if not category_val:
        return '<span class="badge">Uncategorized</span>'
    cat_lower = category_val.lower().replace(" ", "")
    badge_class = {
        "insurance": "badge-insurance",
        "banking": "badge-banking",
        "governmentid": "badge-government",
        "property": "badge-property",
        "tax": "badge-tax",
        "medical": "badge-medical",
    }.get(cat_lower, "badge")
    return f'<span class="badge {badge_class}">{category_val}</span>'


# ==================================================================
# LOGIN / REGISTER SCREEN
# ==================================================================
def show_auth_screen():
    """Login and registration form."""
    st.markdown('<div class="main-header">📄 Personal Document Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered document intelligence at your fingertips</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)

                if submitted and email and password:
                    res = api_post("/api/v1/auth/login", {"email": email, "password": password})
                    if res and res.status_code == 200:
                        data = res.json()
                        st.session_state.jwt_token = data["access_token"]
                        st.session_state.user_email = email
                        st.success("✅ Logged in successfully!")
                        st.rerun()
                    elif res:
                        st.error(f"❌ Login failed: {res.json().get('detail', 'Unknown error')}")

        with tab_register:
            with st.form("register_form"):
                reg_email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_pass")
                reg_role = st.selectbox("Role", ["user", "admin", "viewer"])
                reg_submitted = st.form_submit_button("Register", use_container_width=True)

                if reg_submitted and reg_email and reg_password:
                    res = api_post("/api/v1/auth/register", {
                        "email": reg_email,
                        "password": reg_password,
                        "role": reg_role,
                    })
                    if res and res.status_code in (200, 201):
                        st.success("✅ Account created! Please login.")
                    elif res:
                        st.error(f"❌ Registration failed: {res.json().get('detail', 'Unknown error')}")


# ==================================================================
# SIDEBAR NAVIGATION
# ==================================================================
def show_sidebar():
    """Sidebar with user info and navigation."""
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user_email}")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "📁 Documents",
                "💬 Chat Q&A",
                "🔍 Search",
                "📊 Insights & Alerts",
                "📅 Timeline",
                "🔗 Knowledge Graph",
                "⚙️ Compare",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.jwt_token = None
            st.session_state.user_email = None
            st.session_state.chat_history = []
            st.session_state.session_id = None
            st.rerun()

        return page


# ==================================================================
# DOCUMENTS PAGE
# ==================================================================
def show_documents_page():
    """Document upload & list view with category/status badges."""
    st.markdown('<div class="main-header">📁 My Documents</div>', unsafe_allow_html=True)

    # Upload section
    with st.expander("📤 Upload New Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, PNG, JPG, JPEG, TIFF)",
            type=["pdf", "png", "jpg", "jpeg", "tiff"],
        )
        if uploaded_file and st.button("Upload", use_container_width=True):
            with st.spinner("Uploading and processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = api_post_files("/api/v1/documents/upload", files=files)
                if res and res.status_code in (200, 201):
                    st.success(f"✅ Uploaded: {uploaded_file.name}")
                    st.rerun()
                elif res:
                    st.error(f"❌ Upload failed: {res.json().get('detail', 'Unknown error')}")

    st.divider()

    # Document list
    res = api_get("/api/v1/documents")
    if res and res.status_code == 200:
        docs = res.json()
        doc_list = docs if isinstance(docs, list) else docs.get("documents", [])

        if not doc_list:
            st.info("No documents uploaded yet. Use the uploader above to get started! 🚀")
            return

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(doc_list)}</div>
                <div class="metric-label">Total Documents</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            ready_count = sum(1 for d in doc_list if d.get("status") == "ready")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ready_count}</div>
                <div class="metric-label">Ready</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            categories = set(d.get("category") for d in doc_list if d.get("category"))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(categories)}</div>
                <div class="metric-label">Categories</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            processing = sum(1 for d in doc_list if d.get("status") == "processing")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{processing}</div>
                <div class="metric-label">Processing</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Document cards
        for doc in doc_list:
            with st.container():
                st.markdown(f"""
                <div class="doc-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.1em;">📄 {doc.get('filename', 'Unknown')}</strong>
                            <span style="color: rgba(255,255,255,0.4); margin-left: 12px;">ID: {doc.get('id', '—')}</span>
                        </div>
                        <div>
                            {status_badge(doc.get('status', 'unknown'))}
                            {category_badge(doc.get('category'))}
                        </div>
                    </div>
                    <div style="margin-top: 10px; color: rgba(255,255,255,0.5); font-size: 0.85em;">
                        📝 Pages: {doc.get('page_count', '—')} &nbsp;|&nbsp;
                        📐 Size: {_format_size(doc.get('file_size', 0))} &nbsp;|&nbsp;
                        🕐 Uploaded: {_format_date(doc.get('created_at'))}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _format_size(size_bytes):
    """Format file size for display."""
    if not size_bytes:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_date(date_str):
    """Format datetime string for display."""
    if not date_str:
        return "—"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return str(date_str)[:19]


# ==================================================================
# CHAT Q&A PAGE
# ==================================================================
def show_chat_page():
    """Chat-style Q&A hitting /api/v1/query."""
    st.markdown('<div class="main-header">💬 Ask Your Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your documents using natural language</div>', unsafe_allow_html=True)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                sources = msg.get("sources", [])
                st.markdown(f'<div class="chat-assistant">🤖 {content}</div>', unsafe_allow_html=True)
                if sources:
                    source_text = " | ".join(
                        f"📄 {s.get('document', 'Unknown')} (p.{s.get('page', '?')})" for s in sources
                    )
                    st.caption(f"Sources: {source_text}")

    # Input area
    query = st.chat_input("Ask anything about your documents...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            payload = {"query": query}
            if st.session_state.session_id:
                payload["session_id"] = st.session_state.session_id

            res = api_post("/api/v1/query", payload)
            if res and res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "I couldn't find an answer.")
                sources = data.get("sources", [])
                st.session_state.session_id = data.get("session_id", st.session_state.session_id)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            elif res:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {res.json().get('detail', 'Request failed')}",
                })
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.session_id = None
            st.rerun()


# ==================================================================
# SEARCH PAGE
# ==================================================================
def show_search_page():
    """Search interface with multiple search modes."""
    st.markdown('<div class="main-header">🔍 Search Documents</div>', unsafe_allow_html=True)

    search_mode = st.selectbox("Search Mode", ["Hybrid Search", "Semantic Search", "Keyword Search", "Entity Search"])
    query = st.text_input("Search query", placeholder="Find documents about...")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category filter", [
            None, "Insurance", "Banking", "Government ID", "Property",
            "Tax", "Medical", "Education", "Employment", "Legal", "Financial", "Utility",
        ])
    with col2:
        top_k = st.slider("Results", 3, 20, 5)

    if st.button("🔍 Search", use_container_width=True) and query:
        endpoint_map = {
            "Hybrid Search": "/api/v1/hybrid-search",
            "Semantic Search": "/api/v1/semantic-search",
            "Keyword Search": "/api/v1/search",
            "Entity Search": "/api/v1/entity-search",
        }
        endpoint = endpoint_map[search_mode]

        payload = {"query": query, "top_k": top_k}
        if category:
            payload["category"] = category

        with st.spinner("Searching..."):
            res = api_post(endpoint, payload)
            if res and res.status_code == 200:
                data = res.json()
                results = data.get("results", [])

                if not results:
                    st.info("No results found. Try a different query or search mode.")
                else:
                    st.success(f"Found {len(results)} results")
                    for i, r in enumerate(results, 1):
                        with st.expander(f"#{i} — {r.get('filename', 'Unknown')} (Score: {r.get('score', 0):.3f})"):
                            st.write(f"**Category:** {r.get('category', 'Unknown')}")
                            st.write(f"**Page:** {r.get('page', '—')}")
                            st.write(f"**Text preview:**")
                            st.text(r.get("text", r.get("chunk_text", "No preview available"))[:500])
            elif res:
                st.error(f"Search failed: {res.json().get('detail', 'Unknown error')}")


# ==================================================================
# INSIGHTS & ALERTS PAGE
# ==================================================================
def show_insights_alerts_page():
    """Alerts and insights dashboard."""
    st.markdown('<div class="main-header">📊 Insights & Alerts</div>', unsafe_allow_html=True)

    tab_alerts, tab_insights, tab_duplicates = st.tabs(["🔔 Alerts", "📈 Insights", "🔄 Duplicates"])

    with tab_alerts:
        warning_days = st.slider("Warning window (days)", 7, 365, 60)
        res = api_get("/api/v1/alerts", params={"warning_days": warning_days})
        if res and res.status_code == 200:
            data = res.json()

            # Summary
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Alerts", data.get("total_alerts", 0))
            with c2:
                st.metric("🔴 Critical", data.get("critical_count", 0))
            with c3:
                st.metric("🟡 Warning", data.get("warning_count", 0))
            with c4:
                st.metric("🔵 Info", data.get("info_count", 0))

            st.divider()

            alerts = data.get("alerts", [])
            if not alerts:
                st.info("🎉 No alerts — your documents look good!")
            for alert in alerts:
                severity = alert.get("severity", "info")
                css_class = f"alert-{severity}"
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "ℹ️")
                st.markdown(f"""
                <div class="{css_class}">
                    {icon} <strong>{alert.get('title', 'Alert')}</strong><br>
                    <span style="color: rgba(255,255,255,0.7);">{alert.get('description', '')}</span><br>
                    <span style="font-size: 0.8em; color: rgba(255,255,255,0.4);">
                        📄 {alert.get('filename', '')} &nbsp;|&nbsp; Type: {alert.get('alert_type', '')}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    with tab_insights:
        res = api_post("/api/v1/insights")
        if res and res.status_code == 200:
            data = res.json()

            # Category distribution
            distribution = data.get("category_distribution", {})
            if distribution:
                st.subheader("📊 Category Distribution")
                for cat, count in distribution.items():
                    st.progress(min(count / max(distribution.values(), default=1), 1.0), text=f"{cat}: {count}")

            # Key metrics
            metrics = data.get("metrics", {})
            if metrics:
                st.subheader("📈 Key Metrics")
                cols = st.columns(3)
                for i, (key, value) in enumerate(metrics.items()):
                    with cols[i % 3]:
                        st.metric(key.replace("_", " ").title(), value)
        elif res:
            st.warning("Could not load insights.")

    with tab_duplicates:
        if st.button("🔄 Scan for Duplicates", use_container_width=True):
            with st.spinner("Scanning..."):
                res = api_post("/api/v1/duplicates", {"threshold": 0.9})
                if res and res.status_code == 200:
                    data = res.json()
                    exact = data.get("exact_duplicates", [])
                    near = data.get("near_duplicates", [])

                    if not exact and not near:
                        st.success("✅ No duplicates found!")
                    else:
                        if exact:
                            st.subheader(f"🔗 Exact Duplicates ({len(exact)})")
                            for pair in exact:
                                st.write(f"• {pair.get('doc_a', '?')} ↔ {pair.get('doc_b', '?')}")
                        if near:
                            st.subheader(f"🔗 Near Duplicates ({len(near)})")
                            for pair in near:
                                st.write(f"• {pair.get('doc_a', '?')} ↔ {pair.get('doc_b', '?')} (similarity: {pair.get('similarity', 0):.2f})")


# ==================================================================
# TIMELINE PAGE
# ==================================================================
def show_timeline_page():
    """Timeline visualization of document dates."""
    st.markdown('<div class="main-header">📅 Document Timeline</div>', unsafe_allow_html=True)

    category = st.selectbox("Filter by category", [
        None, "Insurance", "Banking", "Government ID", "Property",
        "Tax", "Medical", "Education", "Employment",
    ])

    sort_order = st.radio("Sort order", ["asc", "desc"], horizontal=True)

    if st.button("📅 Generate Timeline", use_container_width=True):
        with st.spinner("Building timeline..."):
            payload = {"sort_order": sort_order}
            if category:
                payload["category"] = category

            res = api_post("/api/v1/timeline", payload)
            if res and res.status_code == 200:
                data = res.json()
                events = data.get("events", [])

                if not events:
                    st.info("No dated events found in your documents.")
                else:
                    st.success(f"Found {len(events)} events")
                    for event in events:
                        st.markdown(f"""
                        <div class="timeline-item">
                            <strong>{event.get('date', 'Unknown date')}</strong>
                            <span style="color: rgba(255,255,255,0.5);"> — {event.get('event_type', 'Event')}</span><br>
                            <span style="color: rgba(255,255,255,0.7);">{event.get('description', '')}</span><br>
                            <span style="font-size: 0.8em; color: rgba(255,255,255,0.4);">
                                📄 {event.get('filename', '')} &nbsp;|&nbsp; {event.get('category', '')}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
            elif res:
                st.error(f"Timeline failed: {res.json().get('detail', 'Unknown error')}")


# ==================================================================
# KNOWLEDGE GRAPH PAGE
# ==================================================================
def show_knowledge_graph_page():
    """Knowledge graph viewer with node/edge display."""
    st.markdown('<div class="main-header">🔗 Knowledge Graph</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        entity_type = st.selectbox("Filter by entity type", [
            None, "Person", "Policy", "TaxIdentifier", "BankAccount", "Property", "Document",
        ])
    with col2:
        search_query = st.text_input("Search nodes", placeholder="Search by name or value...")

    params = {}
    if entity_type:
        params["entity_type"] = entity_type
    if search_query:
        params["query"] = search_query

    res = api_get("/api/v1/graph", params=params)
    if res and res.status_code == 200:
        data = res.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Nodes", len(nodes))
        with c2:
            st.metric("Edges", len(edges))

        if not nodes:
            st.info("No graph data yet. Upload and process some documents first!")
            return

        st.divider()

        # Nodes display
        st.subheader("📌 Nodes")
        node_types = {}
        for node in nodes:
            ntype = node.get("type", "Unknown")
            if ntype not in node_types:
                node_types[ntype] = []
            node_types[ntype].append(node)

        for ntype, node_list in node_types.items():
            with st.expander(f"{ntype} ({len(node_list)})"):
                for node in node_list:
                    css = {
                        "Document": "kg-document",
                        "Person": "kg-person",
                        "Policy": "kg-policy",
                    }.get(ntype, "kg-entity")
                    st.markdown(
                        f'<span class="kg-node {css}">{node.get("label", node.get("id", "?"))}</span>',
                        unsafe_allow_html=True,
                    )

        # Edges display
        if edges:
            st.subheader("🔗 Relationships")
            for edge in edges[:50]:  # Limit displayed edges
                st.write(f"  {edge.get('source', '?')} —[{edge.get('relation', '?')}]→ {edge.get('target', '?')}")
    elif res:
        st.warning("Could not load knowledge graph data.")


# ==================================================================
# COMPARE PAGE
# ==================================================================
def show_compare_page():
    """Compare two documents side by side."""
    st.markdown('<div class="main-header">⚙️ Compare Documents</div>', unsafe_allow_html=True)

    # Load document list for selection
    res = api_get("/api/v1/documents")
    if not res or res.status_code != 200:
        st.warning("Could not load documents.")
        return

    docs = res.json()
    doc_list = docs if isinstance(docs, list) else docs.get("documents", [])

    if len(doc_list) < 2:
        st.info("You need at least 2 documents to compare.")
        return

    doc_options = {f"{d['filename']} (ID: {d['id']})": d["id"] for d in doc_list}

    col1, col2 = st.columns(2)
    with col1:
        doc1_label = st.selectbox("Document 1", list(doc_options.keys()))
    with col2:
        remaining = [k for k in doc_options.keys() if k != doc1_label]
        doc2_label = st.selectbox("Document 2", remaining)

    if st.button("🔍 Compare", use_container_width=True):
        doc_id_1 = doc_options[doc1_label]
        doc_id_2 = doc_options[doc2_label]

        with st.spinner("Comparing documents..."):
            res = api_post("/api/v1/compare", {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2})
            if res and res.status_code == 200:
                data = res.json()

                st.success("✅ Comparison complete")

                # Show structured diffs
                diffs = data.get("diffs", data.get("differences", []))
                if isinstance(diffs, list):
                    for diff in diffs:
                        if isinstance(diff, dict):
                            st.markdown(f"""
                            <div class="doc-card">
                                <strong>{diff.get('field', 'Field')}</strong><br>
                                <span style="color: #ef4444;">− {diff.get('doc1_value', diff.get('old', '—'))}</span><br>
                                <span style="color: #10b981;">+ {diff.get('doc2_value', diff.get('new', '—'))}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.write(diff)
                elif isinstance(diffs, str):
                    st.text(diffs)

                # Summary
                summary = data.get("summary", "")
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
            elif res:
                st.error(f"Comparison failed: {res.json().get('detail', 'Unknown error')}")


# ==================================================================
# MAIN APP ROUTING
# ==================================================================
def main():
    if not st.session_state.jwt_token:
        show_auth_screen()
        return

    page = show_sidebar()

    if page == "📁 Documents":
        show_documents_page()
    elif page == "💬 Chat Q&A":
        show_chat_page()
    elif page == "🔍 Search":
        show_search_page()
    elif page == "📊 Insights & Alerts":
        show_insights_alerts_page()
    elif page == "📅 Timeline":
        show_timeline_page()
    elif page == "🔗 Knowledge Graph":
        show_knowledge_graph_page()
    elif page == "⚙️ Compare":
        show_compare_page()


if __name__ == "__main__":
    main()
