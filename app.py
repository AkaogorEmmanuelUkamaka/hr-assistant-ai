import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
st.write("Secrets loaded:", "google" in st.secrets)
st.set_page_config(
    page_title="HR Assistant AI",
    page_icon="🏢",
    layout="wide"
)

import tempfile
import os
import hashlib
import json
import threading

from googleapiclient.discovery import build
from google.oauth2 import service_account

import speech_recognition as sr

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# ------------------------
# CONFIG
# ------------------------
GOOGLE_DRIVE_FOLDER_ID = "1j5btciU2XzsdVuwjBwp-rjg7RFuxhslG"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
METADATA_FILE = "doc_metadata.json"

# ------------------------
# VOICE ENGINE
# ------------------------
import threading
import pyttsx3
import time  # keep this for later streaming effect

def speak_text(text):
    def worker():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS Error:", e)

    threading.Thread(target=worker, daemon=True).start()
def voice_to_text():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")

            audio = recognizer.listen(source, timeout=5)

        return recognizer.recognize_google(audio)

    except:
        return None

def hands_free_mode():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Hands-free mode ON... Speak anytime")
        while st.session_state.hands_free:
            try:
                audio = recognizer.listen(source, timeout=5)
                st.session_state.voice_input = recognizer.recognize_google(audio)
                break
            except:
                continue

# ------------------------
# HASHING + METADATA
# ------------------------

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def get_last_sync_time():
    metadata = load_metadata()
    return metadata.get("_last_updated", "Never")


def save_metadata(metadata):
    metadata["_last_updated"] = str(__import__("datetime").datetime.now())

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# ------------------------
# GOOGLE DRIVE
# ------------------------
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["google"],
        scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)
def list_files():
    service = get_drive_service()
    results = service.files().list(
        q=f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    return results.get("files", [])

def download_file(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    return request.execute()

def sync_google_drive_threaded():
    threading.Thread(target=sync_google_drive).start()

def sync_google_drive():
    global vectorstore
    drive_files = list_files()
    existing_metadata = load_metadata()
    docs, new_metadata, changes_detected = [], {}, False

    for file in drive_files:
        file_bytes = download_file(file["id"])
        file_hash = get_file_hash(file_bytes)
        new_metadata[file["name"]] = file_hash

        if file["name"] not in existing_metadata or existing_metadata[file["name"]] != file_hash:
            changes_detected = True

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            path = tmp.name

        if file["name"].endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file["name"].endswith(".txt"):
            loader = TextLoader(path)
        else:
            loader = Docx2txtLoader(path)

        docs.extend(loader.load())

    if changes_detected:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        save_metadata(new_metadata)
        st.success("📚 Google Drive synced & Knowledge Base updated ✅")

# ------------------------
# PAGE CONFIG + STYLE
# ------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#020617); color:white;}
.block-container {max-width:820px; margin:auto;}
.subtitle {text-align:center; color:#94a3b8;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(255,255,255,0.04); padding:12px; border-radius:12px; text-align:center; margin-bottom:10px; font-size:14px; color:#94a3b8;">
Internal HR Knowledge Assistant • Voice Enabled AI Mode
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🗨️ HR AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions via text or voice</div>", unsafe_allow_html=True)

# ------------------------
# SESSION STATE
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "category_stats" not in st.session_state:
    st.session_state.category_stats = {
        "Leave": 0,
        "Benefits": 0,
        "Attendance": 0,
        "Other": 0
    }

if "voice_input" not in st.session_state:
    st.session_state.voice_input = None

if "hands_free" not in st.session_state:
    st.session_state.hands_free = False

# ------------------------
# CLASSIFIER
# ------------------------
def classify_question(query):
    q = query.lower()
    if any(x in q for x in ["leave","vacation","absence","holiday","sick"]): return "Leave"
    if any(x in q for x in ["insurance","salary","benefit","bonus"]): return "Benefits"
    if any(x in q for x in ["attendance","late","clock","shift","time"]): return "Attendance"
    return "Other"

# ------------------------
# MODELS (CACHED)
# ------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_answer(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generate_answer


embeddings = load_embeddings()
llm = load_llm()
# ------------------------
# VECTORSTORE
# ------------------------
@st.cache_resource
def load_vectorstore():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

vectorstore = load_vectorstore()

# ------------------------
# AUTO SYNC GOOGLE DRIVE
# ------------------------
if "google" in st.secrets:
    if "sync_started" not in st.session_state:
        st.session_state.sync_started = True
        threading.Thread(target=sync_google_drive, daemon=True).start()

# ------------------------
# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:

    st.markdown("## 🏢 HR Assistant")

    mode = st.radio("Access mode", ["Employee", "Admin"])
    st.markdown("---")

    # ===== SYSTEM STATUS BADGES =====
    if os.path.exists("faiss_index"):
        st.success("🟢 Knowledge Base: Ready")
    else:
        st.warning("🟡 Knowledge Base: Not Built")

    if "google" in st.secrets:
        st.success("🟢 Google Drive Sync: Connected")
    else:
        st.info("⚪ Google Drive Sync: Not Connected")

    # ===== DOCUMENT STATUS BADGE =====
    if os.path.exists("faiss_index"):
        try:
            doc_count = len(vectorstore.docstore._dict)
        except:
            doc_count = "Unknown"

        last_sync = get_last_sync_time()
        st.success(f"📄 Documents Indexed: {doc_count}")
        st.caption(f"Last Updated: {last_sync}")
    else:
        st.info("📄 Documents Indexed: 0")

    st.markdown("---")

    # ===== CLEAR CHAT HISTORY (ALL USERS) =====
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Chat history cleared ✅")

    st.markdown("---")

    # ===== EMPLOYEE MODE =====
    if mode == "Employee":
        st.info("👤 Employee Mode: Ask questions and view policies only.")

        if os.path.exists("faiss_index"):
            st.success("📚 Knowledge Base Ready")
        else:
            st.warning("📂 Knowledge Base not ready")

    # ===== ADMIN MODE =====
    elif mode == "Admin":
        st.info("🛠️ Admin Mode: Manage documents and sync knowledge base.")

        # Google Drive sync button
        if "google" in st.secrets:
            if st.button("🔄 Sync Google Drive"):
                sync_google_drive_threaded()

        # Upload documents
        uploaded_files = st.file_uploader(
            "Upload HR docs",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            existing_metadata = load_metadata()
            new_metadata, docs, changes_detected = {}, [], False

            for file in uploaded_files:
                file_bytes = file.read()
                file_hash = get_file_hash(file_bytes)
                new_metadata[file.name] = file_hash

                if file.name not in existing_metadata or existing_metadata[file.name] != file_hash:
                    changes_detected = True

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file_bytes)
                    path = tmp.name

                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(path)
                else:
                    loader = Docx2txtLoader(path)

                docs.extend(loader.load())

            if changes_detected:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local("faiss_index")
                save_metadata(new_metadata)
                st.success("Knowledge base updated ✅")

    # ===== FOOTER VERSION BADGE =====
    st.markdown("---")
    st.caption("HR Assistant v1.1 • Voice + Google Drive Sync • Built with Streamlit, FAISS & LangChain")
   
# ------------------------
# PROMPT TEMPLATE
# ------------------------
prompt_template = """
You are an HR assistant.
Conversation:
{history}
Context:
{context}
Question:
{question}
Answer clearly:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context","question","history"])

# ------------------------
# ------------------------
# QA ENGINE
# ------------------------
def get_answer(query):
    if vectorstore is None:
        return "Please upload documents first.", []

    docs_scores = vectorstore.similarity_search_with_score(query, k=6)
    filtered = [d for d, s in docs_scores if s < 1.2]

    if not filtered:
        return "I couldn’t find this information in the company policy. You may wish to contact HR.", []

    context = "\n\n".join(d.page_content for d in filtered)

    history = "\n".join(
        f"User: {item['user']}\nAI: {item['ai']}"
        for item in st.session_state.chat_history[-5:]
    )

    final_prompt = prompt.format(
        context=context,
        question=query,
        history=history
    )

    answer = llm(final_prompt)

    # STREAMING EFFECT
    placeholder = st.empty()
    typed_text = ""

    words = answer.split()

    for i, word in enumerate(words):
        typed_text += word + " "
        placeholder.markdown(typed_text + "▌")
        time.sleep(0.03)

    placeholder.markdown(typed_text.strip())

    # MEMORY UPDATE
    st.session_state.chat_history.append({
        "user": query,
        "ai": answer
    })

    return answer, filtered


# ------------------------
# CHAT UI (KEEP SEPARATE)
# ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

col1, col2, col3 = st.columns([6,1,1])
with col2:
    if st.button("🎤"):
        spoken = voice_to_text()
        if spoken: st.session_state.voice_input = spoken
with col3:
    if st.button("🔁"):
        st.session_state.hands_free = not st.session_state.hands_free
        if st.session_state.hands_free: threading.Thread(target=hands_free_mode, daemon=True).start()

query = st.chat_input("Ask HR anything...")
if st.session_state.voice_input:
    query = st.session_state.voice_input
    st.session_state.voice_input = None
    st.session_state.input_mode = "voice"
else:
    st.session_state.input_mode = "text"

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤖 Thinking...")

        answer, sources = get_answer(query)

        placeholder.markdown(answer)

        # 🔊 Speak ONLY if voice input
        if st.session_state.get("input_mode") == "voice":
            speak_text(answer)

        # 📚 Show sources safely
        if sources:
            with st.expander("Sources"):
                for d in sources:
                    st.write(d.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------------
# ANALYTICS
# ------------------------
if vectorstore:
    col1, col2, col3 = st.columns(3)
    col1.metric("Docs", len(vectorstore.docstore._dict))
    col2.metric("Model", "MiniLM")
    col3.metric("Questions", st.session_state.question_count)
    with st.expander("Analytics"): st.bar_chart(st.session_state.category_stats)

# ------------------------
# FOOTER
# ------------------------
st.markdown("""
<br><br>
<div style="text-align:center">
<a href="https://www.linkedin.com/in/emmanuelakaogor" target="_blank"
style="text-decoration:none; font-size:16px; color:white; background:#0077b5; padding:10px 18px; border-radius:8px; display:inline-block;">
🔗 Connect with the developer
</a>
</div>
""", unsafe_allow_html=True)
