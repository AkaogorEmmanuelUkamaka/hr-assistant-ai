import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pandas as pd
from collections import Counter
from datetime import datetime

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

import time  

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
# ANALYTICS STORAGE
# ------------------------
ANALYTICS_FILE = "analytics.json"


def load_analytics():
    if os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, "r") as f:
            return json.load(f)

    return {
        "total_questions": 0,
        "daily_usage": {},
        "categories": {
            "Leave": 0,
            "Benefits": 0,
            "Attendance": 0,
            "Other": 0
        },
        "questions": []
    }


def save_analytics(data):
    with open(ANALYTICS_FILE, "w") as f:
        json.dump(data, f, indent=2)

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

        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file['name']}: {e}")

    if changes_detected and docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        #ONLY SAVE (NO session_state here)
        vstore = FAISS.from_documents(chunks, embeddings)
        vstore.save_local("faiss_index")

        save_metadata(new_metadata)

        print("✅ FAISS index saved to disk")

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
Internal HR Knowledge Assistant
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🗨️ HR AI Assistant</h1>", unsafe_allow_html=True)
st.caption("🚀 Coming in v2: Voice interaction, Slack integration, and advanced analytics")
st.markdown(
    "<div class='subtitle'>This assistant is designed for company HR-related queries only. Responses are based strictly on available internal documents.</div>",
    unsafe_allow_html=True
)

with st.expander("🚀 Roadmap"):
    st.markdown("""
    - 🎤 Voice interaction (speech-to-text + audio responses)
    - 💬 Slack / Teams integration
    - 🔐 Employee authentication
    - 📊 Advanced HR analytics dashboard
    """)
# ------------------------
# SESSION STATE
# ------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []    
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
# ------------------------
# VECTORSTORE (FIXED)
# ------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ✅ Load from disk if exists
if os.path.exists("faiss_index"):
    try:
        st.session_state.vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading index: {e}")


# ------------------------
# AUTO SYNC GOOGLE DRIVE
# ------------------------
if "google" in st.secrets:
    if "sync_started" not in st.session_state:
        st.session_state.sync_started = True
        threading.Thread(target=sync_google_drive, daemon=True).start()

# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:

    st.markdown("## 🏢 HR Assistant")

    mode = st.radio("Access mode", ["Employee", "Admin"])
    st.markdown("---")

    # ===== SYSTEM STATUS =====
    if os.path.exists("faiss_index"):
        st.success("🟢 Knowledge Base: Ready")
    else:
        st.warning("🟡 Knowledge Base: Not Built")

    if "google" in st.secrets:
        st.success("🟢 Google Drive Sync: Connected")
    else:
        st.info("⚪ Google Drive Sync: Not Connected")

    # ===== DOCUMENT COUNT =====
    if os.path.exists("faiss_index") and st.session_state.vectorstore:
        try:
            doc_count = len(st.session_state.vectorstore.docstore._dict)
        except:
            doc_count = "Unknown"
        st.success(f"📄 Documents Indexed: {doc_count}")
    else:
        st.info("📄 Documents Indexed: 0")

    st.markdown("---")

    # ===== CLEAR CHAT =====
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Chat history cleared ✅")

    st.markdown("---")

    # =========================
    # MODES
    # =========================

    if mode == "Employee":
        st.info("👤 Employee Mode: Ask questions only.")

    elif mode == "Admin":
        st.info("🛠️ Admin Mode: Manage documents.")

        # 🔄 Sync button
        if "google" in st.secrets:
            if st.button("🔄 Sync Google Drive"):
                sync_google_drive_threaded()

        # 📤 Upload (NOW CORRECTLY INSIDE ADMIN + SIDEBAR)
        uploaded_files = st.file_uploader(
            "Upload HR docs",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )

        # ✅ Show uploaded files
        if st.session_state.uploaded_file_names:
            st.markdown("### 📂 Uploaded Documents")
            for name in st.session_state.uploaded_file_names:
                st.write(f"📄 {name}")

        # ✅ Process uploads
        if uploaded_files:
            existing_metadata = load_metadata()
            new_metadata, docs, changes_detected = {}, [], False

            for file in uploaded_files:
                file_bytes = file.read()

                if file.name not in st.session_state.uploaded_file_names:
                    st.session_state.uploaded_file_names.append(file.name)

                file_hash = get_file_hash(file_bytes)
                new_metadata[file.name] = file_hash

                if file.name not in existing_metadata or existing_metadata.get(file.name) != file_hash:
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

                try:
                    docs.extend(loader.load())
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")

            if changes_detected and docs:
                with st.spinner("🧠 Absorbing document into HR brain..."):
                    time.sleep(1)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )

                    chunks = splitter.split_documents(docs)

                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                    else:
                        new_vectorstore = FAISS.from_documents(chunks, embeddings)
                        st.session_state.vectorstore.merge_from(new_vectorstore)

                    st.session_state.vectorstore.save_local("faiss_index")

                    save_metadata(new_metadata)

                    st.success("📚 Document absorbed into knowledge base")
                    st.balloons()

                    st.rerun()

    # ===== FOOTER =====
    st.markdown("---")

    st.caption("HR Assistant v1.1 • Google Drive Sync • Built with Streamlit, FAISS & LangChain")
   
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
    if st.session_state.vectorstore is None:
        return "Please upload documents first.", []

    docs_scores = st.session_state.vectorstore.similarity_search_with_score(query, k=6)
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

query = st.chat_input("Ask HR anything...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    # -------------------------
    # Session tracking
    # -------------------------
    st.session_state.question_count += 1

    category = classify_question(query)
    st.session_state.category_stats[category] += 1

    # -------------------------
    # Persistent tracking
    # -------------------------
    analytics = load_analytics()

    analytics["total_questions"] += 1
    analytics["categories"][category] += 1

    today = datetime.now().strftime("%Y-%m-%d")

    if today not in analytics["daily_usage"]:
        analytics["daily_usage"][today] = 0

    analytics["daily_usage"][today] += 1

    analytics["questions"].append(query)

    save_analytics(analytics)

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤖 Thinking...")

        answer, sources = get_answer(query)

        placeholder.markdown(answer)

        # 📚 Show sources safely
        if sources:
            with st.expander("Sources"):
                for d in sources:
                    st.write(d.page_content[:300])

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
# ------------------------
# ------------------------
# EXECUTIVE DASHBOARD
# ------------------------
if st.session_state.vectorstore and mode == "Admin":

    analytics = load_analytics()

    st.markdown("---")
    st.header("📊 Executive HR Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        len(st.session_state.vectorstore.docstore._dict)
    )

    col2.metric(
        "Session Questions",
        st.session_state.question_count
    )

    col3.metric(
        "All-Time Questions",
        analytics["total_questions"]
    )

    # --------------------
    # DAILY TREND
    # --------------------
    st.subheader("📈 Daily Usage Trend")

    trend_df = pd.DataFrame(
        list(analytics["daily_usage"].items()),
        columns=["Date", "Questions"]
    )

    if not trend_df.empty:
        trend_df["Date"] = pd.to_datetime(trend_df["Date"])
        trend_df = trend_df.set_index("Date")
        st.line_chart(trend_df)
    else:
        st.info("No usage data yet.")

    # --------------------
    # CATEGORY BREAKDOWN
    # --------------------
    st.subheader("📊 Search Categories")

    cat_df = pd.DataFrame(
        list(analytics["categories"].items()),
        columns=["Category", "Count"]
    ).set_index("Category")

    st.bar_chart(cat_df)

    # --------------------
    # TOP SEARCHES
    # --------------------
    st.subheader("🔥 Top Questions")

    top_q = Counter(
        analytics["questions"]
    ).most_common(5)

    if top_q:
        for q, count in top_q:
            st.write(f"• {q} ({count}x)")
    else:
        st.info("No questions yet.")

    # --------------------
    # AI INSIGHT
    # --------------------
    st.subheader("🧠 AI Insight")

    top_cat = max(
        analytics["categories"],
        key=analytics["categories"].get
    )

    st.success(
        f"Employees most frequently ask about **{top_cat}**."
    )

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