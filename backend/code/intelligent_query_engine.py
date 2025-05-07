import re
import pandas as pd
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

class IntelligentQueryEngine:
    def __init__(self, model_path, df: pd.DataFrame, text_columns, state_filter=None, state_column_name='state name'):
        self.llm = GPT4All(model_path)
        self.df = df
        self.text_columns = text_columns
        self.state_filter = state_filter
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.state_column_name = state_column_name

    def set_state_filter(self, state_name: str):
        self.state_filter = state_name

    def ask(self, question: str):
        df = self.df

        # Apply state filter if needed
        if self.state_filter:
            df_columns_clean = df.columns.str.strip().str.lower()
            if self.state_column_name.lower() not in df_columns_clean.tolist():
                raise ValueError(f"State column '{self.state_column_name}' not found in data!")

            matching_col = next((col for col in df.columns if col.strip().lower() == self.state_column_name.strip().lower()), None)
            if matching_col:
                df = df[df[matching_col].str.lower() == self.state_filter.lower()]
            else:
                raise ValueError("State column not found after searching.")

        df = df.reset_index(drop=True)
        question = question.strip()

        # Compute embeddings
        question_embedding = self.embedder.encode([question], convert_to_tensor=True)

        # Define intents with example layman questions
        intents = {
            "count_schemes": ["How many schemes are there?", "Tell me the total schemes", "What's the scheme count?"],
            "group_by_category": ["What are the scheme types?", "Categorize the schemes", "Show me scheme categories"],
            "groupby_by_village_scheme_category": ["Tell me about SVS or MVS schemes", "Details about BVS schemes"],
            "group_by_status": ["What is the completion status?", "How many completed?", "Status of the schemes"],
            "group_by_sanction_year": ["How many schemes were sanctioned per year?", "Sanctioned year-wise info"],
            "sum_expenditure": ["Total money spent?", "What’s the overall expenditure?", "How much cost after 2019"],
            "summarize_by_water_scheme_type": ["Summarize schemes by type", "Types of schemes summary"],
            "summarize_by_water_scheme_source": ["Summarize based on water source", "Scheme by source"],
            "summarize_by_water_scheme_funded": ["Who funds the schemes?", "Funding-based scheme summary",'funding'],
            "get_scheme_details_by_id": ["Tell me about scheme ID 123", "Give scheme details"]
        }

        # Compute embeddings for all intent examples
        all_examples = []
        intent_map = []
        for intent, examples in intents.items():
            all_examples.extend(examples)
            intent_map.extend([intent] * len(examples))

        example_embeddings = self.embedder.encode(all_examples, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = cosine_similarity(question_embedding, example_embeddings).squeeze(0)
        best_match_idx = similarities.argmax().item()
        best_intent = intent_map[best_match_idx]
        best_score = similarities[best_match_idx].item()

        # Threshold to detect if intent is strong enough
        if best_score < 0.5:
            return self.llm_query(df, question)

        # Map to function calls
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
        }

        # Execute the matched function
        
        return intent_functions[best_intent](df)
    def count_schemes(self, df):
        return f"Total number of schemes: {len(df)}"

    def group_by_category(self, df):
        category_col = self.get_matching_column(df, ['Category'])
        if not category_col:
            return "Category column is not available in the data."
        result = df[category_col].value_counts().to_string()
        return f"Schemes grouped by Category:\n{result}"

    def groupby_by_village_scheme_category(self, df, question):
        category_col = self.get_matching_column(df, ['Category'])
        if not category_col:
            return "Category column is not available in the data."
        if 'svs' in question.lower():
            data_df = df[df[category_col].str.lower() == 'svs']
        elif 'mvs' in question.lower():
            data_df = df[df[category_col].str.lower() == 'mvs']
        elif 'bvs' in question.lower():
            data_df = df[df[category_col].str.lower() == 'bvs']
        else:
            data_df = df
        result = data_df[category_col].value_counts().to_string()
        return f"Schemes grouped by Category:\n{result}"

    def group_by_status(self, df):
        status_col = self.get_matching_column(df, ['Status Of Completion', 'Physical Work Status'])
        if not status_col:
            return "Status column is not available in the data."
        result = df[status_col].value_counts().to_string()
        return f"Schemes grouped by Status:\n{result}"

    def group_by_sanction_year(self, df):
        date_col = self.get_matching_column(df, ['SLSSC/ DWSSM meeting date (dd/mm/yyyy)', 'Sanction Date'])
        if not date_col:
            return "Sanction year column is not available."
        df['Sanction Year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
        result = df['Sanction Year'].value_counts().sort_index().to_string()
        return f"Schemes grouped by Sanction Year:\n{result}"

    def sum_expenditure(self, df, question):
        if "central" in question.lower():
            col = self.get_matching_column(df, ['Total central expenditure (in lakhs)'])
        elif "after 2019" in question.lower():
            col = self.get_matching_column(df, ['Total expenditure (in lakhs) on or after 2019-20'])
        else:
            col = self.get_matching_column(df, ['Total expenditure (in lakhs)'])

        if not col:
            return "Relevant expenditure column not available."

        total = pd.to_numeric(df[col], errors='coerce').sum()
        return f"Sum of {col}: {total:.2f} lakhs"

    def summarize_by_water_scheme_type(self, df):
        col = self.get_matching_column(df, ['Type of Scheme'])
        if not col:
            return "Type of Scheme column not found."
        return f"Schemes by Type of Water:\n{df[col].value_counts().to_string()}"

    def summarize_by_water_scheme_source(self, df):
        col = self.get_matching_column(df, ['Source of Scheme'])
        if not col:
            return "Source of Water column not found."
        return f"Schemes by source of Water:\n{df[col].value_counts().to_string()}"

    def summarize_by_water_scheme_funded(self, df):
        col = self.get_matching_column(df, ['Main Schemes Funded From'])
        if not col:
            return "Funding column not found."
        return f"Schemes by funding:\n{df[col].value_counts().to_string()}"

    def get_scheme_details_by_id(self, df, question):
        match = re.search(r'\b([A-Za-z0-9\-_]+)\b', question)
        if not match:
            return "No valid Scheme ID found in the question."

        scheme_id = match.group(1)
        col = self.get_matching_column(df, ['SchemeId', 'Scheme Code', 'ID', 'schemeid'])
        if not col:
            return "Scheme ID column not found in the data."

        row = df[df[col].astype(str).str.strip().str.lower() == scheme_id.lower()]
        if row.empty:
            return f"No scheme found with ID: {scheme_id}"

        return f"Details for Scheme ID '{scheme_id}':\n\n{row.iloc[0].dropna().to_string()}"

    def get_matching_column(self, df, possible_names):
        df_cols_clean = df.columns.str.strip().str.lower()
        for name in possible_names:
            if name.lower() in df_cols_clean.tolist():
                return next((col for col in df.columns if col.strip().lower() == name.lower()), None)
        return None

    def llm_query(self, df, question):
        available_cols = df.columns.tolist()
        valid_text_cols = [col for col in self.text_columns if col in available_cols]

        if not valid_text_cols:
            return "Sorry, no valid columns available for generating context."

        context = df[valid_text_cols].head(5).fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
        prompt = f"""
You are a helpful assistant analyzing water supply schemes.

Data Sample:
{context}

User Question:
{question}

Answer in simple, clear English based on the data and user needs.
"""
        response = self.llm.generate(prompt)
        return response

   
