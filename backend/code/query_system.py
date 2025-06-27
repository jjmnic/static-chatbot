# query_system.py
import pandas as pd
import os
import numpy as np
import faiss
from data_preprocessing import load_data, preprocess_data
from intelligent_query_engine import IntelligentQueryEngine # Corrected import
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore

class QuerySystem:
    def __init__(self, data_path, gpt4all_model_path, output_dir):
        self.output_dir = output_dir
        print(data_path)
        # Load and preprocess
        df = load_data(data_path)
        self.df_preprocess = preprocess_data(df)
        
        # Convert all non-numeric columns to lowercase
        # Apply this after data preprocessing, as some might become numeric.
        for col in self.df_preprocess.select_dtypes(include=['object']).columns:
            self.df_preprocess[col] = self.df_preprocess[col].astype(str).str.lower()
        
        # Save preprocessed data
        preprocessed_path = os.path.join(self.output_dir, 'preprocessed.csv')
        self.df_preprocess.to_csv(preprocessed_path, index=False)
        print(f"Saved preprocessed data at {preprocessed_path}")

        # Infer text columns
        self.text_columns = self._infer_text_columns()

        # Generate embeddings
        self.embedding_generator = EmbeddingGenerator()
        self.embeddings = self.embedding_generator.create_embeddings(self.df_preprocess, self.text_columns)

        # Save embeddings
        embeddings_path = os.path.join(self.output_dir, 'embeddings.npy')
        np.save(embeddings_path, self.embeddings)
        print(f"Saved embeddings at {embeddings_path}")

        # Vector store
        self.vector_store = VectorStore(dim=self.embeddings.shape[1])
        self.vector_store.add_embeddings(self.embeddings)

        # Save FAISS index
        faiss_index_path = os.path.join(self.output_dir, 'faiss_index.index')
        faiss.write_index(self.vector_store.index, faiss_index_path)
        print(f"Saved FAISS index at {faiss_index_path}")
        
        # Pass vector_store and embedder to IntelligentQueryEngine
        self.query_engine = IntelligentQueryEngine(
            llama_api_url="http://10.197.112.27:10022/docs#/default/rag_inference_rag_inference_post",  # Replace with real URL
            df=self.df_preprocess,
            text_columns=self.text_columns, 
            state_column_name='state name', # Assuming 'state name' is the correct column
            district_column_name='district', # Add your district column name here if applicable
            vector_store=self.vector_store, # Pass the vector store
            embedder=self.embedding_generator.model # Pass the SentenceTransformer model
        )

    def _infer_text_columns(self):
        """
        Select object/text columns, except 'State' which is used separately for filtering.
        """
        object_columns = self.df_preprocess.select_dtypes(include=['object']).columns.tolist()
        # Exclude both 'state name' and 'district' (or 'zila') from general text columns
        object_columns = [col for col in object_columns if col.lower() not in ['state name', 'district', 'zila']]
        return object_columns

    def set_state_filter(self, state_name: str):
        self.query_engine.set_state_filter(state_name)

    def set_district_filter(self, district_name: str): # New method for district filtering
        self.query_engine.set_district_filter(district_name)

    def ask(self, question: str):
        return self.query_engine.ask(question)