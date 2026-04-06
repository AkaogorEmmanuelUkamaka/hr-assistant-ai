print("Building vector database...")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDFs
files = [
    "data/hr_policy.pdf",
    "data/leave_policy.pdf",
    "data/faq.pdf"
]

documents = []
for file in files:
    print(f"Loading {file}...")
    loader = PyPDFLoader(file)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector DB
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save
vectorstore.save_local("faiss_index")

print("✅ Done! Vector DB saved.")