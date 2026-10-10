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
import os
import requests
import streamlit as st
from datetime import datetime, date, timedelta

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_BASE = os.getenv("DOC_ASSISTANT_API_URL", "http://127.0.0.1:8001").rstrip("/")

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

saved_token = st.query_params.get("token") if hasattr(st, "query_params") else None
saved_email = st.query_params.get("email") if hasattr(st, "query_params") else None
if saved_token and not st.session_state.jwt_token:
    st.session_state.jwt_token = saved_token
    st.session_state.user_email = saved_email
if st.session_state.session_id:
    st.query_params["session_id"] = str(st.session_state.session_id)


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
        st.session_state.pop("api_last_error", None)
        return r
    except requests.RequestException as exc:
        st.session_state.api_last_error = str(exc)
        st.error(f"⚠️ Cannot reach the API server at {API_BASE}: {exc}")
        return None


def api_post(path, json_data=None, files=None):
    """POST request to the API."""
    try:
        r = requests.post(f"{API_BASE}{path}", headers=api_headers(), json=json_data, files=files, timeout=60)
        st.session_state.pop("api_last_error", None)
        return r
    except requests.RequestException as exc:
        st.session_state.api_last_error = str(exc)
        st.error(f"⚠️ Cannot reach the API server at {API_BASE}: {exc}")
        return None


def api_post_files(path, files):
    """POST request with file upload."""
    try:
        headers = {}
        if st.session_state.jwt_token:
            headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
        r = requests.post(f"{API_BASE}{path}", headers=headers, files=files, timeout=120)
        st.session_state.pop("api_last_error", None)
        return r
    except requests.RequestException as exc:
        st.session_state.api_last_error = str(exc)
        st.error(f"⚠️ Cannot reach the API server at {API_BASE}: {exc}")
        return None


def api_delete(path):
    """DELETE request to the API."""
    try:
        r = requests.delete(f"{API_BASE}{path}", headers=api_headers(), timeout=30)
        st.session_state.pop("api_last_error", None)
        return r
    except requests.RequestException as exc:
        st.session_state.api_last_error = str(exc)
        st.error(f"⚠️ Cannot reach the API server at {API_BASE}: {exc}")
        return None


def api_error_message(response, fallback="The request failed."):
    """Read a helpful API error when the server does not return JSON."""
    if response is None:
        detail = st.session_state.get("api_last_error")
        suffix = f" Details: {detail}" if detail else " Check that the backend is running and try again."
        return f"I couldn't reach the API at {API_BASE}.{suffix}"
    if response.status_code == 401:
        return "Your login has expired. Please log out and sign in again."
    try:
        detail = response.json().get("detail")
        return str(detail or fallback)
    except (ValueError, AttributeError):
        return f"{fallback} (HTTP {response.status_code})"


def extract_document_list(payload):
    """Normalize API document-list payloads across backend response shapes."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


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
                        st.query_params["token"] = data["access_token"]
                        st.query_params["email"] = email
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
            for key in ["token", "email", "session_id"]:
                if key in st.query_params:
                    del st.query_params[key]
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
        doc_list = extract_document_list(docs)

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
    """Chat interface with persistent conversations and document disambiguation."""
    st.markdown('<div class="main-header">💬 Ask Your Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions, explore your files, and continue previous conversations.</div>', unsafe_allow_html=True)

    health_response = api_get("/health")
    if not health_response or health_response.status_code != 200:
        st.warning("The document API is unavailable. Chat messages cannot be answered or saved until it reconnects.")

    saved_response = api_get("/api/v1/query/sessions")
    saved_sessions = saved_response.json().get("sessions", []) if saved_response and saved_response.status_code == 200 else []
    if saved_response and saved_response.status_code not in (200, 401):
        st.warning(api_error_message(saved_response, "Could not load saved conversations."))
    elif saved_response and saved_response.status_code == 401:
        st.warning(api_error_message(saved_response))
    labels = {
        f"{item.get('title', 'Chat')[:42]} · {item.get('message_count', 0)} messages · {item['session_id'][:6]}": item["session_id"]
        for item in saved_sessions
    }

    controls, conversation = st.columns([1, 3], gap="large")
    with controls:
        st.markdown("#### Your chats")
        if st.button("＋  New chat", key="chat_new", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.session_id = None
            if "session_id" in st.query_params:
                del st.query_params["session_id"]
            st.rerun()

        if labels:
            selected_label = st.selectbox(
                "Past conversations",
                list(labels),
                key="saved_chat_selector",
                label_visibility="collapsed",
            )
            selected_session_id = labels[selected_label]
            if st.button("Open chat", key="chat_open", use_container_width=True):
                result = api_get(f"/api/v1/query/session/{selected_session_id}")
                if result and result.status_code == 200:
                    stored = result.json()
                    st.session_state.session_id = stored["session_id"]
                    st.session_state.chat_history = stored.get("messages", [])
                    st.query_params["session_id"] = stored["session_id"]
                    st.rerun()
                elif result:
                    st.error("Could not open that chat.")

            active_id = st.session_state.session_id
            if active_id and active_id in labels.values():
                if st.button("Delete active chat", key="chat_delete", use_container_width=True):
                    result = api_delete(f"/api/v1/query/session/{active_id}")
                    if result and result.status_code == 200:
                        st.session_state.chat_history = []
                        st.session_state.session_id = None
                        st.rerun()
                    elif result:
                        st.error("Could not delete that chat.")
        else:
            if st.session_state.chat_history:
                st.caption("This conversation is currently open, but it is not saved on the server yet. Send a message while connected to save it.")
            else:
                st.caption("Your conversations will appear here and remain available after you sign in again.")

    with conversation:
        if not st.session_state.chat_history:
            st.markdown(
                "<div class='doc-card'><strong>What would you like to find?</strong><br>"
                "Ask about a document, compare files, or request a resume review.</div>",
                unsafe_allow_html=True,
            )

        for index, msg in enumerate(st.session_state.chat_history):
            role = msg.get("role", "assistant")
            chat_role = "user" if role == "user" else "assistant"
            with st.chat_message(chat_role, avatar="🧑" if chat_role == "user" else "🤖"):
                st.markdown(msg.get("content", ""))
                sources = msg.get("sources", []) or []
                if sources:
                    with st.expander(f"Sources · {len(sources)}"):
                        for source in sources:
                            filename = source.get("filename", "Document")
                            page_number = source.get("page_number", 1)
                            st.caption(f"📄 {filename} · page {page_number}")
                            if source.get("snippet"):
                                st.write(source["snippet"])

                document_options = msg.get("document_options", []) or []
                has_followup = any(
                    later.get("role") == "user"
                    for later in st.session_state.chat_history[index + 1:]
                )
                if role != "user" and document_options and not has_followup:
                    option_by_label = {
                        f"{option['filename']} · {option.get('page_count') or '?'} pages · ID {option['document_id']}": option
                        for option in document_options
                    }
                    choice = st.selectbox(
                        "Choose the resume to use",
                        list(option_by_label),
                        key=f"resume_choice_{st.session_state.session_id}_{index}",
                    )
                    if st.button("Answer using this resume", key=f"resume_use_{st.session_state.session_id}_{index}"):
                        prior_user = next(
                            (
                                previous for previous in reversed(st.session_state.chat_history[:index])
                                if previous.get("role") == "user"
                            ),
                            None,
                        )
                        question = prior_user.get("content", "Review this resume") if prior_user else "Review this resume"
                        option = option_by_label[choice]
                        with st.spinner(f"Reviewing {option['filename']}..."):
                            result = api_post(
                                "/api/v1/query",
                                {
                                    "query": question,
                                    "session_id": st.session_state.session_id,
                                    "document_id": option["document_id"],
                                },
                            )
                        if result and result.status_code == 200:
                            data = result.json()
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": f"Use {option['filename']} for my answer",
                            })
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": data.get("answer", "I couldn't complete that review."),
                                "sources": data.get("sources", []),
                                "document_options": data.get("document_options", []),
                            })
                            st.rerun()
                        elif result:
                            st.error(api_error_message(result, "Could not answer from that document."))
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": api_error_message(None),
                            })
                            st.rerun()

        query = st.chat_input("Ask about your documents...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.spinner("Searching your documents..."):
                payload = {"query": query}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id
                result = api_post("/api/v1/query", payload)

            if result and result.status_code == 200:
                data = result.json()
                st.session_state.session_id = data.get("session_id", st.session_state.session_id)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": data.get("answer", "I couldn't find an answer."),
                    "sources": data.get("sources", []),
                    "document_options": data.get("document_options", []),
                })
                st.query_params["session_id"] = st.session_state.session_id
            elif result:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": api_error_message(result),
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": api_error_message(None),
                })
            st.rerun()


# ==================================================================
# SEARCH PAGE
# ==================================================================
def show_search_page():
    """Search interface focused on actionable, inspectable document evidence."""
    st.markdown('<div class="main-header">🔍 Search Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find exact facts or passages, then inspect the source evidence and page before acting on it.</div>', unsafe_allow_html=True)

    with st.form("document_search_form"):
        col_query, col_mode = st.columns([3, 1])
        with col_query:
            query = st.text_input("Search terms", placeholder="Try a name, policy number, topic, amount, or phrase…")
        with col_mode:
            search_mode = st.selectbox("Search method", ["Hybrid", "Semantic", "Exact keywords", "Extracted entities"])

        col_category, col_entity, col_top_k = st.columns([2, 2, 1])
        with col_category:
            category = st.selectbox("Document category", ["All categories", "Insurance", "Banking", "Government ID", "Property", "Tax", "Medical", "Education", "Employment", "Legal", "Financial", "Utility", "Other"])
        with col_entity:
            entity_type = st.selectbox("Entity type", ["Any entity", "Policy Number", "PAN", "Aadhaar", "Bank Account Number", "IFSC", "Expiry Date", "Amount", "Person Name"])
        with col_top_k:
            top_k = st.selectbox("Max results", [5, 10, 20], index=1)
        submitted = st.form_submit_button("Search documents", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a name, keyword, or question to search for.")
            st.session_state.pop("search_result_state", None)
        else:
            endpoint_map = {
                "Hybrid": "/api/v1/hybrid-search",
                "Semantic": "/api/v1/semantic-search",
                "Exact keywords": "/api/v1/search",
                "Extracted entities": "/api/v1/entity-search",
            }
            endpoint = endpoint_map[search_mode]

            if search_mode == "Extracted entities":
                payload = {"entity_value": query, "top_k": top_k}
                if entity_type != "Any entity":
                    payload["entity_type"] = entity_type
            else:
                payload = {"query": query, "top_k": top_k}
                if category != "All categories":
                    payload["category"] = category

            with st.spinner("Searching your processed documents…"):
                res = api_post(endpoint, payload)
            if res and res.status_code == 200:
                data = res.json()
                st.session_state.search_result_state = {
                    "query": query,
                    "mode": search_mode,
                    "results": data.get("items", data.get("results", [])),
                    "total": data.get("total_results", len(data.get("items", data.get("results", [])))),
                }
            elif res:
                st.error(api_error_message(res, "Search failed."))

    result_state = st.session_state.get("search_result_state")
    if result_state:
        results = result_state["results"]
        st.divider()
        if not results:
            st.info(f'No matches for “{result_state["query"]}”. Try a shorter exact phrase, a related term, or another search method.')
        else:
            st.success(f'{len(results)} result(s) for “{result_state["query"]}” · {result_state["mode"]}')
            for i, result in enumerate(results, 1):
                filename = result.get("filename", "Unknown document")
                page = result.get("source_page", result.get("page_number", result.get("page", 1)))
                title = f"{i}. {filename} · page {page}"
                with st.expander(title, expanded=(i == 1)):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Category", result.get("category") or "Uncategorized")
                    c2.metric("Match type", result.get("method", result.get("entity_type", result_state["mode"])))
                    score = result.get("score", result.get("confidence"))
                    c3.metric("Relevance", f"{score:.0%}" if isinstance(score, (int, float)) and score <= 1 else (f"{score:.3f}" if isinstance(score, (int, float)) else "—"))
                    evidence = result.get("text", result.get("value", result.get("chunk_text", "")))
                    if evidence:
                        st.markdown("**Evidence from the document**")
                        st.code(str(evidence).strip(), language=None)
                    st.caption(f"Document ID {result.get('document_id', '—')} · source page {page}. Verify important details against the original document.")


# ==================================================================
# INSIGHTS & ALERTS PAGE
# ==================================================================
def show_insights_alerts_page():
    """Portfolio health dashboard with evidence-based alerts and explicit duplicate scans."""
    st.markdown('<div class="main-header">📊 Insights & Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review processing health, dates found in your files, and potential duplicate uploads.</div>', unsafe_allow_html=True)

    tab_alerts, tab_insights, tab_duplicates = st.tabs(["🔔 Alerts", "📈 Insights", "🔄 Duplicates"])

    with tab_alerts:
        filters = st.columns([1, 1, 2])
        with filters[0]:
            warning_days = st.selectbox("Expiry window", [30, 60, 90, 180, 365], index=2, format_func=lambda days: f"Next {days} days")
        with filters[1]:
            alert_type = st.selectbox("Alert type", ["All alerts", "expiry", "missing_document", "duplicate", "version_modified"])
        with filters[2]:
            if st.button("Refresh alerts", key="refresh_alerts"):
                st.session_state.pop("portfolio_alert_state", None)

        filter_key = (warning_days, alert_type)
        if st.session_state.get("portfolio_alert_state", {}).get("key") != filter_key:
            params = {"warning_days": warning_days}
            if alert_type != "All alerts":
                params["alert_type"] = alert_type
            response = api_get("/api/v1/alerts", params=params)
            if response and response.status_code == 200:
                st.session_state.portfolio_alert_state = {"key": filter_key, "data": response.json()}
            elif response:
                st.error(api_error_message(response, "Could not load alerts."))

        data = st.session_state.get("portfolio_alert_state", {}).get("data")
        if data:
            cols = st.columns(4)
            cols[0].metric("Needs attention", data.get("total_alerts", 0))
            cols[1].metric("Urgent", data.get("critical_count", 0))
            cols[2].metric("Upcoming", data.get("warning_count", 0))
            cols[3].metric("Portfolio notes", data.get("info_count", 0))
            st.caption("Alerts are generated from extracted document data. Confirm dates and recommendations against the original file.")
            alerts = data.get("alerts", [])
            if not alerts:
                st.success("No alerts match these filters.")
            for alert in alerts:
                severity = alert.get("severity", "info")
                title = alert.get("title", "Portfolio alert")
                description = alert.get("description", "")
                if severity == "critical":
                    st.error(f"**{title}**\n\n{description}")
                elif severity == "warning":
                    st.warning(f"**{title}**\n\n{description}")
                else:
                    st.info(f"**{title}**\n\n{description}")
                details = [f"Type: {alert.get('alert_type', '—')}"]
                if alert.get("filename"):
                    details.append(f"Document: {alert['filename']}")
                if alert.get("expiry_date"):
                    details.append(f"Detected date: {alert['expiry_date']}")
                if alert.get("days_remaining") is not None:
                    details.append(f"Days remaining: {alert['days_remaining']}")
                st.caption(" · ".join(details))

    with tab_insights:
        if st.button("Refresh portfolio", key="refresh_insights") or "portfolio_insight_state" not in st.session_state:
            response = api_post("/api/v1/insights")
            if response and response.status_code == 200:
                st.session_state.portfolio_insight_state = response.json()
            elif response:
                st.error(api_error_message(response, "Could not load portfolio insights."))
        data = st.session_state.get("portfolio_insight_state")
        if data:
            statuses = data.get("status_distribution", {})
            total_docs = data.get("total_documents", 0)
            ready = statuses.get("ready", 0)
            failed = statuses.get("failed", 0)
            uploaded = statuses.get("uploaded", 0) + statuses.get("processing", 0)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Documents", total_docs)
            metric_cols[1].metric("Ready for search", ready, help="Only ready documents are searchable in chat.")
            metric_cols[2].metric("Need processing", uploaded)
            metric_cols[3].metric("Processing errors", failed)
            st.caption(f"Portfolio total: {data.get('total_pages', 0)} known pages · {_format_size(data.get('total_storage_bytes', 0))} stored. Unprocessed documents may not have page counts yet.")

            left, right = st.columns(2)
            with left:
                st.subheader("Documents by category")
                distribution = data.get("category_distribution", {})
                if distribution:
                    st.bar_chart(distribution, horizontal=True)
                    denominator = sum(distribution.values()) or 1
                    st.caption(" · ".join(f"{name}: {count} ({count / denominator:.0%})" for name, count in sorted(distribution.items(), key=lambda pair: pair[1], reverse=True)))
                else:
                    st.info("Category counts will appear after you upload documents.")
            with right:
                st.subheader("Extracted information")
                entities = data.get("entity_counts", {})
                if entities:
                    st.dataframe(
                        [{"Field type": name, "Extracted values": count} for name, count in sorted(entities.items(), key=lambda pair: pair[1], reverse=True)],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No structured fields have been extracted yet.")

            st.subheader("Dates identified as expiry/validity dates")
            expiring_docs = data.get("expiring_documents", [])
            if expiring_docs:
                st.dataframe(
                    [{"Document": item.get("filename"), "Category": item.get("category") or "Uncategorized", "Extracted date": item.get("expiry_date"), "Evidence status": "Review source document"} for item in expiring_docs],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Dates are extracted from document text and are not independently verified.")
            else:
                st.success("No expiry/validity dates have been identified in your documents.")

    with tab_duplicates:
        st.caption("This scan can take longer on large collections because it compares document content. It only inspects your own processed files.")
        threshold = st.slider("Near-duplicate similarity", 0.70, 0.98, 0.90, 0.01, help="Higher values report only more similar documents.")
        if st.button("Scan processed documents", key="run_duplicate_scan", use_container_width=True):
            with st.spinner("Scanning..."):
                res = api_post("/api/v1/duplicates", {"threshold": threshold})
                if res and res.status_code == 200:
                    st.session_state.duplicate_result_state = {"threshold": threshold, "data": res.json()}
                elif res:
                    st.error(api_error_message(res, "Duplicate scan failed."))
        duplicate_state = st.session_state.get("duplicate_result_state")
        if duplicate_state:
            data = duplicate_state["data"]
            st.caption(f"Scanned {data.get('total_documents_scanned', 0)} ready documents · threshold {duplicate_state['threshold']:.0%}")
            exact = data.get("exact_duplicates", [])
            near = data.get("near_duplicates", [])
            summary_cols = st.columns(3)
            summary_cols[0].metric("Exact groups", data.get("exact_duplicate_groups", len(exact)))
            summary_cols[1].metric("Similar pairs", data.get("near_duplicate_pairs", len(near)))
            summary_cols[2].metric("Documents scanned", data.get("total_documents_scanned", 0))
            if not exact and not near:
                st.success("No duplicate candidates found at this threshold.")
            for group in exact:
                names = [doc.get("filename", "Unknown") for doc in group.get("documents", [])]
                with st.expander(f"Exact match · {len(names)} files · same file hash"):
                    for name in names:
                        st.write(f"• {name}")
            for pair in near:
                doc1 = pair.get("document_1", {}).get("filename", "Unknown")
                doc2 = pair.get("document_2", {}).get("filename", "Unknown")
                similarity = pair.get("similarity_score", 0)
                with st.expander(f"Similar content · {doc1} ↔ {doc2} · {similarity:.0%}"):
                    st.write("These files may contain overlapping text. Compare the originals before deleting either one.")


# ==================================================================
# TIMELINE PAGE
# ==================================================================
def show_timeline_page():
    """Review extracted dates as a filterable, source-attributed event timeline."""
    st.markdown('<div class="main-header">📅 Document Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track expiry dates, policy dates, and other dates found in processed documents. Each event includes its source file and page.</div>', unsafe_allow_html=True)

    with st.form("timeline_filters"):
        col_category, col_sort, col_dates = st.columns([2, 1, 2])
        with col_category:
            category = st.selectbox("Category", ["All categories", "Insurance", "Banking", "Government ID", "Property", "Tax", "Medical", "Education", "Employment", "Legal", "Financial", "Utility", "Other"])
        with col_sort:
            sort_order = st.selectbox("Order", ["asc", "desc"], format_func=lambda order: "Oldest first" if order == "asc" else "Newest first")
        with col_dates:
            include_range = st.checkbox("Limit to a date range")
            start_date = st.date_input("From", value=date.today() - timedelta(days=365), disabled=not include_range)
            end_date = st.date_input("Through", value=date.today() + timedelta(days=365), disabled=not include_range)
        submitted = st.form_submit_button("Load timeline", type="primary", use_container_width=True)

    if submitted:
        if include_range and start_date > end_date:
            st.error("The start date must be on or before the end date.")
        else:
            with st.spinner("Loading timeline events…"):
                payload = {"sort_order": sort_order}
                if category != "All categories":
                    payload["category"] = category
                res = api_post("/api/v1/timeline", payload)
            if res and res.status_code == 200:
                st.session_state.timeline_result_state = {
                    "data": res.json(),
                    "range_enabled": include_range,
                    "start": start_date.isoformat() if include_range else None,
                    "end": end_date.isoformat() if include_range else None,
                }
            elif res:
                st.error(api_error_message(res, "Timeline could not be loaded."))

    timeline_state = st.session_state.get("timeline_result_state")
    if timeline_state:
        all_events = timeline_state["data"].get("events", [])
        if timeline_state["range_enabled"]:
            start = timeline_state["start"]
            end = timeline_state["end"]
            all_events = [event for event in all_events if event.get("parsed_timestamp") and start <= event["parsed_timestamp"][:10] <= end]
        event_types = sorted({event.get("event_type", "Date") for event in all_events})
        selected_types = st.multiselect("Event types", event_types, default=event_types)
        visible_events = [event for event in all_events if event.get("event_type", "Date") in selected_types]
        parsed_events = [event for event in visible_events if event.get("parsed_timestamp")]
        unparsed_events = [event for event in visible_events if not event.get("parsed_timestamp")]
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Events shown", len(visible_events))
        kpi2.metric("Dates understood", len(parsed_events))
        kpi3.metric("Dates need review", len(unparsed_events))
        if parsed_events:
            st.dataframe([{
                "Date": event["date_string"],
                "Event": event["event_type"],
                "Document": event["filename"],
                "Category": event.get("category") or "Uncategorized",
                "Page": event.get("page", 1),
                "Confidence": event.get("confidence"),
            } for event in parsed_events], use_container_width=True, hide_index=True)
        else:
            st.info("No parsed events match these filters. Try another category or remove the date range.")
        if unparsed_events:
            with st.expander(f"Dates needing manual review ({len(unparsed_events)})"):
                st.caption("These date-like values could not be safely interpreted and are excluded from date-range filtering.")
                for event in unparsed_events:
                    st.write(f"• {event['date_string']} · {event['event_type']} · {event['filename']} · page {event.get('page', 1)}")


# ==================================================================
# KNOWLEDGE GRAPH PAGE
# ==================================================================
def show_knowledge_graph_page():
    """Explore how processed documents relate to people, identifiers, and organizations."""
    st.markdown('<div class="main-header">🔗 Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Relationships are inferred from extracted document fields. Use the source documents to verify identity matches.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        entity_type = st.selectbox("Filter by entity type", [
            None, "Person", "Policy", "TaxIdentifier", "BankAccount", "Property", "Organization", "Document",
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
            st.info("No relationships found for these filters. Process documents with identifiable people, organizations, policies, accounts, or properties to build the graph.")
            return

        st.divider()
        max_nodes = 80
        graph_nodes = sorted(nodes, key=lambda node: (node.get("node_type", ""), node.get("label", "")))[:max_nodes]
        visible_ids = {node["id"] for node in graph_nodes}
        graph_edges = [edge for edge in edges if edge.get("source") in visible_ids and edge.get("target") in visible_ids]
        if len(nodes) > max_nodes:
            st.info(f"Showing {max_nodes} of {len(nodes)} nodes to keep the graph readable. Refine the filters to focus on a specific person, policy, or document.")

        graphviz_ids = {node["id"]: f"n{index}" for index, node in enumerate(graph_nodes)}
        palette = {
            "Document": ("#dbeafe", "#2563eb", "note"),
            "Person": ("#dcfce7", "#16a34a", "ellipse"),
            "Policy": ("#ede9fe", "#7c3aed", "ellipse"),
            "BankAccount": ("#cffafe", "#0891b2", "ellipse"),
            "TaxIdentifier": ("#fef3c7", "#d97706", "ellipse"),
            "Property": ("#fce7f3", "#db2777", "ellipse"),
            "Organization": ("#e0e7ff", "#4f46e5", "ellipse"),
        }
        dot_lines = ["digraph document_graph {", "rankdir=LR;", "graph [bgcolor=\"transparent\", pad=\"0.2\", nodesep=\"0.35\", ranksep=\"0.65\"];", "node [style=\"filled,rounded\", fontname=\"Arial\", fontsize=10, margin=\"0.12,0.08\"];", "edge [fontname=\"Arial\", fontsize=9, color=\"#94a3b8\", arrowsize=0.7];"]
        for node in graph_nodes:
            node_type = node.get("node_type", "Unknown")
            fill, stroke, shape = palette.get(node_type, ("#f1f5f9", "#64748b", "ellipse"))
            label = f"{node.get('label', node['id'])}\\n[{node_type}]"
            dot_lines.append(f"{graphviz_ids[node['id']]} [label={json.dumps(label)}, shape={shape}, fillcolor={json.dumps(fill)}, color={json.dumps(stroke)}];")
        for edge in graph_edges[:160]:
            source_id = graphviz_ids[edge["source"]]
            target_id = graphviz_ids[edge["target"]]
            relation = edge.get("relation", "related to").replace("_", " ")
            dot_lines.append(f"{source_id} -> {target_id} [label={json.dumps(relation)}];")
        dot_lines.append("}")
        st.subheader("Relationship map")
        st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

        st.subheader("Connected records")
        node_by_id = {node["id"]: node for node in nodes}
        if edges:
            relationship_rows = []
            for edge in graph_edges:
                source = node_by_id.get(edge["source"], {})
                target = node_by_id.get(edge["target"], {})
                relationship_rows.append({
                    "From": source.get("label", edge["source"]),
                    "Relationship": edge.get("relation", "related to").replace("_", " "),
                    "To": target.get("label", edge["target"]),
                })
            st.dataframe(relationship_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Nodes were found, but no relationship edges matched this selection.")

        with st.expander("Node details and provenance"):
            for node in graph_nodes:
                properties = node.get("properties", {})
                st.markdown(f"**{node.get('label', node.get('id'))}** · {node.get('node_type', 'Unknown')}")
                if properties:
                    st.json(properties)
    elif res:
        st.warning("Could not load knowledge graph data.")


# ==================================================================
# COMPARE PAGE
# ==================================================================
def show_compare_page():
    """Compare processed documents with a clear, side-by-side evidence report."""
    st.markdown('<div class="main-header">⚙️ Compare Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare extracted fields across two processed documents. Differences are review candidates—not proof that either document is correct.</div>', unsafe_allow_html=True)

    res = api_get("/api/v1/documents", params={"page_size": 100, "status": "ready"})
    if not res or res.status_code != 200:
        if res:
            st.error(api_error_message(res, "Could not load processed documents."))
        return

    docs = res.json()
    doc_list = extract_document_list(docs)

    ready_docs = [document for document in doc_list if document.get("status") == "ready"]
    if len(ready_docs) < 2:
        st.info("Comparison requires at least two processed (ready) documents. Upload and process another document first.")
        return

    doc_options = {
        f"{doc['filename']} · {doc.get('category') or 'Uncategorized'} · ID {doc['id']}": doc
        for doc in ready_docs
    }

    col1, col2 = st.columns(2)
    with col1:
        doc1_label = st.selectbox("First document", list(doc_options.keys()), key="compare_doc1")
    with col2:
        remaining = [k for k in doc_options.keys() if k != doc1_label]
        doc2_label = st.selectbox("Second document", remaining, key="compare_doc2")

    doc1 = doc_options[doc1_label]
    doc2 = doc_options[doc2_label]
    if doc1.get("category") and doc2.get("category") and doc1["category"] != doc2["category"]:
        st.warning(f"These documents have different categories ({doc1['category']} vs. {doc2['category']}). Compare the results carefully; different document types may not share comparable fields.")

    if st.button("🔍 Compare", use_container_width=True):
        doc_id_1 = doc1["id"]
        doc_id_2 = doc2["id"]
        with st.spinner("Comparing documents..."):
            res = api_post("/api/v1/compare", {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2})
            if res and res.status_code == 200:
                st.session_state.document_comparison_state = {
                    "key": (doc_id_1, doc_id_2),
                    "data": res.json(),
                }
            elif res:
                st.error(api_error_message(res, "Comparison failed."))

    compare_state = st.session_state.get("document_comparison_state")
    if compare_state and compare_state["key"] == (doc1["id"], doc2["id"]):
        data = compare_state["data"]
        st.success(data.get("diff_summary", "Comparison complete."))
        differences = data.get("differences", [])
        if differences:
            st.subheader(f"Fields to review · {len(differences)}")
            first_name = data.get("document_1", {}).get("filename", doc1["filename"])
            second_name = data.get("document_2", {}).get("filename", doc2["filename"])
            key1 = f"{first_name} (ID {doc1['id']})"
            key2 = f"{second_name} (ID {doc2['id']})"
            rows = []
            for diff in differences:
                value1 = diff.get(key1, "—")
                value2 = diff.get(key2, "—")
                rows.append({
                    "Extracted field": diff.get("entity", "Unknown field"),
                    first_name: ", ".join(map(str, value1)) if isinstance(value1, list) else value1,
                    second_name: ", ".join(map(str, value2)) if isinstance(value2, list) else value2,
                    f"{first_name} source pages": ", ".join(map(str, diff.get("document_1_pages", []))) or "—",
                    f"{second_name} source pages": ", ".join(map(str, diff.get("document_2_pages", []))) or "—",
                    f"{first_name} extraction confidence": diff.get("document_1_confidence"),
                    f"{second_name} extraction confidence": diff.get("document_2_confidence"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        elif data.get("identical_entities"):
            st.success("No differences were detected in the structured fields extracted from these files.")
        else:
            st.warning("No comparable structured fields were extracted from one or both files. Check their processing status or compare more similar document types.")

        identical = data.get("identical_entities", [])
        if identical:
            with st.expander(f"Matching extracted fields · {len(identical)}"):
                st.dataframe(
                    [{
                        "Field": item.get("entity"),
                        "Matching value": item.get("value"),
                        f"{data.get('document_1', {}).get('filename', 'Document 1')} source pages": ", ".join(map(str, item.get("document_1_pages", []))) or "—",
                        f"{data.get('document_2', {}).get('filename', 'Document 2')} source pages": ", ".join(map(str, item.get("document_2_pages", []))) or "—",
                    } for item in identical],
                    use_container_width=True,
                    hide_index=True,
                )
        st.caption("This report compares OCR-extracted structured fields only; wording, clauses, and unrecognized text are not compared. Check the source pages before making a decision.")


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
