import re
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from torch.nn.functional import cosine_similarity
import json

class IntelligentQueryEngine:
    def __init__(self, llama_api_url, df: pd.DataFrame, text_columns, state_filter=None, state_column_name='state name'):
        self.llama_api_url = llama_api_url
        self.df = df
        self.text_columns = text_columns
        self.state_filter = state_filter
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.state_column_name = state_column_name

    def set_state_filter(self, state_name: str):
        self.state_filter = state_name

    def ask(self, question: str):
        df = self.df

        if self.state_filter:
            state_column_name = self.state_column_name.lower()
            df_columns_clean = df.columns.str.strip().str.lower()
            if state_column_name not in df_columns_clean:
                raise ValueError(f"State column '{self.state_column_name}' not found!")
            matching_col = next((col for col in df.columns if col.strip().lower() == state_column_name), None)
            if matching_col:
                df = df[df[matching_col].str.lower() == self.state_filter.lower()]
            else:
                raise ValueError("State column not found after searching.")

        df = df.reset_index(drop=True)
        question = question.strip()

        question_embedding = self.embedder.encode([question], convert_to_tensor=True)

        intents = {
            "count_schemes": ["How many schemes are there?", "Tell me the total schemes"],
            "group_by_category": ["What are the scheme categories?", "Categorize the schemes"],
            "groupby_by_village_scheme_category": ["SVS or MVS schemes", "Details about BVS schemes"],
            "group_by_status": ["What is the completion status?", "How many completed?"],
            "group_by_sanction_year": ["Sanctioned per year", "Sanctioned year-wise info"],
            "sum_expenditure": ["Total money spent", "How much cost after 2019"],
            "summarize_by_water_scheme_type": ["Summarize schemes by type"],
            "summarize_by_water_scheme_source": ["Scheme by source"],
            "summarize_by_water_scheme_funded": ["Who funds the schemes?", "Funding"],
            "get_scheme_details_by_id": ["Tell me about scheme ID", "Scheme info for ID"],
            "get_scheme_details_by_name": ["Tell me about scheme name", "Give details of scheme"],
        }

        all_examples, intent_map = [], []
        for intent, examples in intents.items():
            all_examples.extend(examples)
            intent_map.extend([intent] * len(examples))

        example_embeddings = self.embedder.encode(all_examples, convert_to_tensor=True)
        similarities = cosine_similarity(question_embedding, example_embeddings).squeeze(0)
        best_match_idx = similarities.argmax().item()
        best_intent = intent_map[best_match_idx]
        best_score = similarities[best_match_idx].item()

        if best_score < 0.5:
            return self.llm_query(df, question)

        intent_functions = {
            "count_schemes": self.count_schemes,
            "group_by_category": self.group_by_category,
            "groupby_by_village_scheme_category": lambda df: self.groupby_by_village_scheme_category(df, question),
            "group_by_status": self.group_by_status,
            "group_by_sanction_year": self.group_by_sanction_year,
            "sum_expenditure": lambda df: self.sum_expenditure(df, question),
            "summarize_by_water_scheme_type": self.summarize_by_water_scheme_type,
            "summarize_by_water_scheme_source": self.summarize_by_water_scheme_source,
            "summarize_by_water_scheme_funded": self.summarize_by_water_scheme_funded,
            "get_scheme_details_by_id": lambda df: self.get_scheme_details_by_id(df, question),
            "get_scheme_details_by_name": lambda df: self.get_scheme_details_by_name(df, question),
            "group_by_scheme_type": self.group_by_scheme_type
        }

        return intent_functions.get(best_intent, self.llm_query)(df)

    def llm_query(self, df, question):
        valid_text_cols = [col for col in self.text_columns if col in df.columns]

        if not valid_text_cols:
            return "No valid columns available for generating context."

        context = df[valid_text_cols].head(5).fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
        prompt = f"""You are a helpful assistant analyzing water supply schemes.

Data Sample:
{context}

User Question:
{question}

Answer in simple, clear English based on the data and user needs.
"""
        try:
            response = requests.post(self.llama_api_url, json={
                "text-input": prompt,
                "max-tokens": 4096
            })
            if response.status_code == 200:
                return response.json().get("output", "No answer generated.")
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"LLAMA API call failed: {str(e)}"

    def count_schemes(self, df):
        return f"Total number of schemes: {len(df)}"

    def group_by_category(self, df):
        col = self.get_matching_column(df, ['Category'])
        return df[col].value_counts().to_string() if col else "Category column not found."

    def group_by_scheme_type(self, df):
        col = self.get_matching_column(df, ['Type of scheme'])
        return df[col].value_counts().to_string() if col else "Scheme Type column not found."

    def groupby_by_village_scheme_category(self, df, question):
        col = self.get_matching_column(df, ['Village Scheme Type'])
        if not col:
            return "Village Scheme Type column not found."
        for keyword in ['svs', 'mvs', 'bvs']:
            if keyword in question.lower():
                df = df[df[col].str.lower() == keyword]
        return df[col].value_counts().to_string()

    def group_by_status(self, df):
        col = self.get_matching_column(df, ['Status Of Completion', 'Physical Work Status'])
        return df[col].value_counts().to_string() if col else "Status column not found."

    def group_by_sanction_year(self, df):
        if 'Sanction Year' not in df.columns:
            return "Sanction Year column not found."
        counts = df['Sanction Year'].value_counts().sort_index()
        return "\n".join(f"• {k}: {v} schemes" for k, v in counts.items())

    def sum_expenditure(self, df, question):
        if "central" in question.lower():
            col = self.get_matching_column(df, ['Total central expenditure (in lakhs)'])
        elif "after 2019" in question.lower():
            col = self.get_matching_column(df, ['Total expenditure (in lakhs) on or after 2019-20'])
        else:
            col = self.get_matching_column(df, ['Total expenditure (in lakhs)'])
        if not col:
            return "Expenditure column not found."
        return f"Sum of {col}: {pd.to_numeric(df[col], errors='coerce').sum():.2f} lakhs"

    def summarize_by_water_scheme_type(self, df):
        col = self.get_matching_column(df, ['Type of Scheme'])
        return df[col].value_counts().to_string() if col else "Type column not found."

    def summarize_by_water_scheme_source(self, df):
        col = self.get_matching_column(df, ['Source of Scheme'])
        return df[col].value_counts().to_string() if col else "Source column not found."

    def summarize_by_water_scheme_funded(self, df):
        col = self.get_matching_column(df, ['Main Schemes Funded From'])
        return df[col].value_counts().to_string() if col else "Funding column not found."

    def get_scheme_details_by_id(self, df, question):
        match = re.search(r'\b\d+\b', question)
        if not match:
            return "No valid ID found."
        scheme_id = int(match.group(0))
        if 'SchemeId' not in df.columns:
            return "SchemeId column not found."
        results = df[df['SchemeId'] == scheme_id]
        return json.dumps(results.to_dict(orient='records'), indent=4) if not results.empty else "Scheme ID not found."

    def get_scheme_details_by_name(self, df, question):
        matches = re.findall(r'(["\'])(.*?)\1', question)
        if not matches:
            return "No scheme name found."
        scheme_names = [m[1].strip() for m in matches]
        col = self.get_matching_column(df, ['Scheme Name', 'SchemeTitle', 'Name'])
        if not col:
            return "Scheme name column not found."
        results = df[df[col].isin(scheme_names)]
        return json.dumps(results.to_dict(orient='records'), indent=4) if not results.empty else "No matching schemes found."

    def get_matching_column(self, df, possible_names):
        df_cols_clean = df.columns.str.strip().str.lower()
        for name in possible_names:
            if name.lower() in df_cols_clean.tolist():
                return next((col for col in df.columns if col.strip().lower() == name.lower()), None)
        return None
