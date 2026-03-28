import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class HybridRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.texts = [d.page_content for d in docs]

        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 📁 Folder to store FAISS index
        self.index_path = "data/vectorstore"

        # ✅ LOAD OR CREATE FAISS
        if os.path.exists(self.index_path):
            print("⚡ Loading existing FAISS index...")
            self.db = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS loaded successfully")

        else:
            print("🚀 Creating FAISS index (first time, may take time)...")

            # Show progress while embedding
            for _ in tqdm(range(len(docs)), desc="Embedding documents"):
                pass  # just for visual progress

            self.db = FAISS.from_documents(docs, self.embeddings)

            # Save for next runs
            self.db.save_local(self.index_path)
            print("💾 FAISS index saved")

        # 🔎 BM25
        print("🔄 Building BM25 index...")
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        print("✅ Retriever Ready")

    def retrieve(self, query, k=3):
        vec_docs = self.db.similarity_search(query, k=k)

        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        bm_docs = [self.docs[i] for i in top_idx]

        return vec_docs + bm_docs