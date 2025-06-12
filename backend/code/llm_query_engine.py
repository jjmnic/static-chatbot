# llm_query_engine.py
import re
import pandas as pd
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer

class LLMQueryEngine:
    def __init__(self, model_path, df: pd.DataFrame, embeddings, vector_store, text_columns, state_filter=None):
        self.llm = GPT4All(model_path)
        self.df = df
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_columns = text_columns
        self.state_filter = state_filter

    def ask(self, question: str, top_k=5):
        # 1. Garbage query check — early return
        if not re.search(r'\b[a-zA-Z]{3,}\b', question):
            return "The query seems invalid or nonsensical. Please ask a meaningful question."

        # 2. Filter by state if needed — early return if no data
        if self.state_filter:
            filtered_df = self.df[self.df['State'].str.lower() == self.state_filter.lower()].reset_index(drop=True)
            if filtered_df.empty:
                return f"Sorry, no data is available for the state: {self.state_filter}."
        else:
            filtered_df = self.df

        # 3. Embedding & Search
        query_embedding = self.embedder.encode([question])
        distances, indices = self.vector_store.search(query_embedding, top_k=top_k)

        relevant_rows = filtered_df.iloc[indices[0]]
        context = relevant_rows[self.text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()

        prompt = f"""
You are an intelligent assistant helping answer questions about government water supply schemes.

Data:
{context}

Question: {question}

Answer in simple English. If it is a count, give the count. If a sum, give the sum. If a summary, write clearly.
"""
        response = self.llm.generate(prompt, max_tokens=256, temp=0.7)
        return response