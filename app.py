import os
import uuid
import base64
import streamlit as st

# ── 0. Page config — must be the very first Streamlit call ───────────────────
st.set_page_config(page_title="Prompt-based Agent", page_icon="🎯")

# ── 1. Load configuration before importing the agent ────────────────────────

def _bootstrap_env() -> None:
    """Populate os.environ from Streamlit secrets or .env, whichever is present."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


_bootstrap_env()

# ── 2. Import agent (env vars must be set first) ─────────────────────────────

from PromptBasedAgent import graph  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def make_thread_id(seed: str) -> str:
    """Return a deterministic UUID5 from *seed* (session key)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def file_to_base64(uploaded_file) -> tuple[str, str]:
    """Convert an uploaded file to (base64_string, mime_type)."""
    mime_type = uploaded_file.type or "image/jpeg"
    b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
    return b64, mime_type


def build_lc_content(text: str, image_b64: str | None, mime_type: str | None) -> str | list:
    """Build LangChain message content — multimodal list if image present, plain str otherwise."""
    if image_b64:
        return [
            {"type": "text", "text": text or "Describe this image."},
            {"type": "image", "base64": image_b64, "mime_type": mime_type},
        ]
    return text


def run_graph(messages: list, thread_id: str) -> str:
    """Invoke the LangGraph agent and return the last AI message content."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": messages}, config=config)
    last = result["messages"][-1]
    if hasattr(last, "content"):
        return last.content
    return str(last.get("content", last))


# ── 4. Streamlit UI ───────────────────────────────────────────────────────────

st.title("Prompt Based Agent")

# ── Session seed ──────────────────────────────────────────────────────────────
if "session_seed" not in st.session_state:
    st.session_state.session_seed = str(uuid.uuid4())

thread_id = make_thread_id(st.session_state.session_seed)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Thread ID: `{thread_id}`")
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.pending_image_b64 = None
        st.session_state.pending_image_mime = None
        st.session_state.pending_image_preview = None
        st.rerun()

# ── Conversation state ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Pending image state (attached before sending)
if "pending_image_b64" not in st.session_state:
    st.session_state.pending_image_b64 = None
if "pending_image_mime" not in st.session_state:
    st.session_state.pending_image_mime = None
if "pending_image_preview" not in st.session_state:
    st.session_state.pending_image_preview = None

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("image_b64"):
            st.image(
                base64.b64decode(msg["image_b64"]),
                caption="Attached image",
                use_container_width=True,
            )
        st.markdown(msg["content"] or "*(image only)*")

# ── Image attachment controls ─────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📎 Upload image",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        key="file_uploader",
        label_visibility="collapsed",
    )

with col2:
    camera_photo = st.camera_input("📷 Take a photo", key="camera_input", label_visibility="collapsed")

# Process whichever image source was used (upload takes priority)
new_image_source = uploaded_file or camera_photo
if new_image_source:
    b64, mime = file_to_base64(new_image_source)
    st.session_state.pending_image_b64 = b64
    st.session_state.pending_image_mime = mime
    st.session_state.pending_image_preview = new_image_source

# Show preview of pending image
if st.session_state.pending_image_preview is not None:
    st.info("🖼️ Image attached — it will be sent with your next message.")
    preview_col, clear_col = st.columns([4, 1])
    with preview_col:
        st.image(st.session_state.pending_image_preview, width=200)
    with clear_col:
        if st.button("✕ Remove"):
            st.session_state.pending_image_b64 = None
            st.session_state.pending_image_mime = None
            st.session_state.pending_image_preview = None
            st.rerun()

# ── New user input ────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input or (st.session_state.pending_image_b64 and user_input is not None):
    text = user_input or ""
    image_b64 = st.session_state.pending_image_b64
    image_mime = st.session_state.pending_image_mime

    # Show and store user message
    with st.chat_message("user"):
        if image_b64:
            st.image(base64.b64decode(image_b64), caption="Attached image", use_container_width=True)
        if text:
            st.markdown(text)

    st.session_state.chat_history.append({
        "role": "user",
        "content": text,
        "image_b64": image_b64,
        "image_mime": image_mime,
    })

    # Clear pending image after attaching
    st.session_state.pending_image_b64 = None
    st.session_state.pending_image_mime = None
    st.session_state.pending_image_preview = None

    # Build LangChain message list for the graph
    lc_messages = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            content = build_lc_content(m["content"], m.get("image_b64"), m.get("image_mime"))
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = run_graph(lc_messages, thread_id)
            except Exception as exc:
                response = f"⚠️ Error: {exc}"
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
