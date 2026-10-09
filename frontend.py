import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Backend Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .status-badge-ok {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .sample-btn {
        margin-bottom: 6px;
    }
    .source-box {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper to check backend health ---
def check_backend_status():
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your **Autonomous MirAI Student Policy Advisor**. I can help you with attendance marks, medical leave procedures, club & society approvals, and code of conduct guidelines strictly based on the official 2026 handbook.",
            "sources": []
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --- Sidebar ---
with st.sidebar:
    st.title("🎓 MirAI Advisor")
    st.caption("Autonomous Generative Policy Assistant")

    st.markdown("---")
    st.subheader("⚡ System Connection")
    backend_ok, health_data = check_backend_status()

    if backend_ok:
        st.markdown('<span class="status-badge-ok">● Backend Connected</span>', unsafe_allow_html=True)
        chunks_count = health_data.get("vector_store_chunks", 0)
        st.write(f"Indexed Chunks: `{chunks_count}`")
    else:
        st.markdown('<span class="status-badge-warn">● Standalone / Direct RAG Mode</span>', unsafe_allow_html=True)
        st.caption("FastAPI backend not detected on port 8000. Fallback direct execution enabled.")

    st.markdown("---")
    st.subheader("📄 Upload Policy Document")
    uploaded_file = st.file_uploader("Upload Policy PDF to Index", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Processing and indexing PDF..."):
                if backend_ok:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        r = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
                        if r.status_code == 200:
                            data = r.json()
                            st.success(f"{data.get('message', 'Indexed successfully')} ({data.get('chunks_added', 0)} chunks)")
                        else:
                            st.error(f"Ingestion failed: {r.text}")
                    except Exception as err:
                        st.error(f"Upload error: {err}")
                else:
                    # Direct local ingestion fallback
                    try:
                        from backend import process_pdf, get_vectorstore
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        chunks = process_pdf(tmp_path)
                        vs = get_vectorstore()
                        vs.add_documents(chunks)
                        os.remove(tmp_path)
                        st.success(f"Indexed {len(chunks)} chunks directly into local ChromaDB!")
                    except Exception as err:
                        st.error(f"Direct ingestion failed: {err}")

    st.markdown("---")
    st.subheader("🧪 Certification Test Queries")
    st.caption("Click any query to test against official handbook:")

    test_queries = [
        ("Precision: 72% Attendance", "I have 72% attendance. How many attendance marks will I get?"),
        ("Multi-Hop: Ratnam Medical Leave", "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"),
        ("Process: Cybersecurity Society", "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"),
        ("Negative Constraint: Smoking Fine", "How much is the fine for smoking a cigarette on campus?")
    ]

    for label, query_text in test_queries:
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.pending_query = query_text

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Area ---
st.markdown('<div class="main-header">Autonomous MirAI Student Policy Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strict retrieval-grounded policy advisor for Mirai School of Technology. Zero hallucinations guaranteed.</div>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 View {len(msg['sources'])} Cited Handbook Excerpts"):
                for idx, src in enumerate(msg["sources"], 1):
                    page_str = f"Page {src['page']}" if src.get("page") else "Handbook Excerpt"
                    st.markdown(f"**{page_str}**")
                    st.markdown(f'<div class="source-box">{src["content"]}</div>', unsafe_allow_html=True)


# --- Query Processing Function ---
def execute_query(user_query: str):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving handbook policies and synthesizing response..."):
            answer = ""
            sources = []

            # 1. Try FastAPI backend
            backend_success = False
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": user_query},
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    backend_success = True
                else:
                    st.warning(f"Backend returned status {resp.status_code}. Using direct fallback.")
            except requests.exceptions.RequestException:
                # Backend timeout or connection failure handled gracefully
                backend_success = False

            # 2. Graceful direct fallback if backend not running or timed out
            if not backend_success:
                try:
                    import asyncio
                    from backend import chat_endpoint, ChatRequest
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    res = loop.run_until_complete(chat_endpoint(ChatRequest(question=user_query)))
                    answer = res.answer
                    sources = [{"page": s.page, "content": s.content} for s in res.sources]
                except Exception as ex:
                    answer = f"⚠️ An error occurred while retrieving policy information: {str(ex)}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander(f"📚 View {len(sources)} Cited Handbook Excerpts"):
                    for idx, src in enumerate(sources, 1):
                        page_str = f"Page {src['page']}" if src.get("page") else "Handbook Excerpt"
                        st.markdown(f"**{page_str}**")
                        st.markdown(f'<div class="source-box">{src["content"]}</div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })


# Check for button click trigger
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    execute_query(q)
    st.rerun()

# Standard chat input
if prompt := st.chat_input("Ask any policy question (e.g., attendance marks, medical leave, club proposals)..."):
    execute_query(prompt)
