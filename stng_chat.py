import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import chat
from PIL import Image

# ----------- RAG Database Loader (no recompute) -----------
class StarTrekRAGDatabase:
    def __init__(self, chunk_path, embed_path, chroma_path, model_name="all-MiniLM-L6-v2"):
        # Load precomputed chunks and embeddings
        with open(chunk_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(embed_path, "rb") as f:
            self.embeddings = pickle.load(f)
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection("tng_docs")
        
    def semantic_search(self, query, top_n=5):
        query_emb = self.model.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_n)
        return results['documents'][0]

class OllamaLLM:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
    def ask(self, question, context):
        messages = [
            {"role": "system", "content": "You are a Star Trek TNG computer. Respond as a Federation AI. Reference episodes and scenes as possible."},
            {"role": "user", "content": f"Episode context:\n{context}\n\nQuestion: {question}\nAnswer as TNG Computer:"}
        ]
        response = chat(model=self.model_name, messages=messages)
        return response.message.content if hasattr(response, "message") else response

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Star Trek TNG RAG Chatbot", page_icon="🖖", layout="centered")
st.markdown("<h1 style='color:#39c6fa'>🖖 Star Trek TNG RAG Chat</h1>", unsafe_allow_html=True)
try:
    banner = Image.open("tng.jpg") 
    st.image(banner, use_column_width=True, caption="Star Trek: The Next Generation")
except Exception:
    st.markdown("")

st.markdown("""
Ask about TNG episodes, characters, philosophy, or classic scenes. You'll get real context from The Next Generation transcripts!
""")

# Database & LLM only loaded once
ragdb = StarTrekRAGDatabase(
    chunk_path="stng_chunks.pkl",
    embed_path="stng_embeddings.pkl",
    chroma_path="stng_chroma_db"
)

llm_agent = OllamaLLM(model_name="llama3.2:1b")  

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Enter your question for Picard, Data, or any TNG episode:", "")

if user_question:
    st.session_state.chat_history.append(("user", user_question))
    retrieved = ragdb.semantic_search(user_question, top_n=5)
    context = "\n".join(retrieved)
    answer = llm_agent.ask(user_question, context)
    st.session_state.chat_history.append(("bot", answer))

# Trek chat bubbles (color, LCARS style)
def render_msg(role, msg):
    colors = {"user": "#fae84b", "bot": "#39c6fa"}
    bgs = {"user": "#222", "bot": "#101640"}
    name = "You" if role == "user" else "TNG Computer"
    icon = "🧑‍🚀" if role == "user" else "🖖"
    st.markdown(
        f"<div style='color:{colors[role]};background:{bgs[role]};padding:8px;margin:2px;border-radius:8px;font-family:sans-serif'><b>{icon} {name}:</b> {msg}</div>", 
        unsafe_allow_html=True
    )

for role, msg in st.session_state.chat_history:
    render_msg(role, msg)

st.caption("Powered by Streamlit, ChromaDB, and Ollama · All context is drawn from authentic Star Trek TNG episodes.")

