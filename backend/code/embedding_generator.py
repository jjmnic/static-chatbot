# embedding_generator.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, df: pd.DataFrame, text_columns: list) -> np.ndarray:
        combined_text = df[text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
        embeddings = self.model.encode(combined_text.tolist(), show_progress_bar=True)
        return embeddings
