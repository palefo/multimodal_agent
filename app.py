import os
import uuid
import base64
import streamlit as st

# ── 0. Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prompt-based Agent", page_icon="🎯")

# ── 1. Bootstrap env ──────────────────────────────────────────────────────────
def _bootstrap_env() -> None:
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

# ── 2. Import agent ───────────────────────────────────────────────────────────
from PromptBasedRagMultimodalAgent import graph  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

# ── 3. Helpers ────────────────────────────────────────────────────────────────
def make_thread_id(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def file_to_base64(uploaded_file) -> tuple[str, str]:
    mime_type = uploaded_file.type or "image/jpeg"
    b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
    return b64, mime_type


def build_lc_content(text: str, image_b64: str | None, mime_type: str | None) -> str | list:
    if image_b64:
        return [
            {"type": "text", "text": text or "Describe this image."},
            {"type": "image", "base64": image_b64, "mime_type": mime_type},
        ]
    return text


def run_graph(messages: list, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": messages}, config=config)
    last = result["messages"][-1]
    if hasattr(last, "content"):
        return last.content
    return str(last.get("content", last))


def render_image(b64: str, width: int = 280) -> None:
    st.image(base64.b64decode(b64), width=width)


# ── 4. Session state ──────────────────────────────────────────────────────────
if "session_seed" not in st.session_state:
    st.session_state.session_seed = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_b64" not in st.session_state:
    st.session_state.pending_b64 = None
if "pending_mime" not in st.session_state:
    st.session_state.pending_mime = None
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

thread_id = make_thread_id(st.session_state.session_seed)

# ── 5. Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Thread ID: `{thread_id}`")
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.pending_b64 = None
        st.session_state.pending_mime = None
        st.session_state.show_camera = False
        st.rerun()

    st.divider()
    st.subheader("📎 Attach Image")

    # File upload
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        key="uploader",
    )
    if uploaded:
        b64, mime = file_to_base64(uploaded)
        st.session_state.pending_b64 = b64
        st.session_state.pending_mime = mime

    # Camera toggle
    toggle_label = "📷 Close camera" if st.session_state.show_camera else "📷 Take a photo"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.show_camera = not st.session_state.show_camera
        st.rerun()

    if st.session_state.show_camera:
        camera_snap = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera_snap:
            b64, mime = file_to_base64(camera_snap)
            st.session_state.pending_b64 = b64
            st.session_state.pending_mime = mime
            st.session_state.show_camera = False
            st.rerun()

    # Pending image preview
    if st.session_state.pending_b64:
        st.divider()
        st.caption("📌 Attached — will send with next message")
        render_image(st.session_state.pending_b64, width=220)
        if st.button("✕ Remove image", use_container_width=True):
            st.session_state.pending_b64 = None
            st.session_state.pending_mime = None
            st.rerun()

# ── 6. Title ──────────────────────────────────────────────────────────────────
st.title("Prompt Based Agent")

# ── 7. Chat history ───────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("image_b64"):
            render_image(msg["image_b64"])
        if msg.get("content"):
            st.markdown(msg["content"])

# ── 8. Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input or (st.session_state.pending_b64 and user_input is not None):
    text = user_input or ""
    image_b64 = st.session_state.pending_b64
    image_mime = st.session_state.pending_mime

    # Show user turn
    with st.chat_message("user"):
        if image_b64:
            render_image(image_b64)
        if text:
            st.markdown(text)

    # Persist to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": text,
        "image_b64": image_b64,
        "image_mime": image_mime,
    })

    # Clear pending image
    st.session_state.pending_b64 = None
    st.session_state.pending_mime = None

    # Build LangChain messages
    lc_messages = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            content = build_lc_content(m["content"], m.get("image_b64"), m.get("image_mime"))
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = run_graph(lc_messages, thread_id)
            except Exception as exc:
                response = f"⚠️ Error: {exc}"
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})