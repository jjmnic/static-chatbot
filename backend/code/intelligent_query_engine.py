import re
import pandas as pd
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from torch.nn.functional import cosine_similarity
import json

class IntelligentQueryEngine:
    def __init__(self, model_path, df: pd.DataFrame, text_columns, state_filter=None, state_column_name='state name'):
        self.llm = GPT4All(model_path)
        self.df = df
        self.text_columns = text_columns
        self.state_filter = state_filter
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.state_column_name = state_column_name

    def set_state_filter(self, state_name: str):
        """Set the state filter to limit queries to a specific state."""
        self.state_filter = state_name

    def ask(self, question: str):
        """Processes the user's query and provides a relevant response."""
        df = self.df

        # Apply state filter if needed
        if self.state_filter:
            # Normalize column names to lowercase for comparison
            state_column_name = self.state_column_name.lower()
            df_columns_clean = df.columns.str.strip().str.lower()

            # Check if the state column exists in the DataFrame
            if state_column_name not in df_columns_clean:
                raise ValueError(f"State column '{self.state_column_name}' not found in the data!")

            # Get the actual column name to apply filter
            matching_col = next((col for col in df.columns if col.strip().lower() == state_column_name), None)

            if matching_col:
                df = df[df[matching_col].str.lower() == self.state_filter.lower()]
            else:
                raise ValueError("State column not found after searching.")

        df = df.reset_index(drop=True)
        question = question.strip()

        # Compute embeddings for the question
        question_embedding = self.embedder.encode([question], convert_to_tensor=True)

        # Define intents with example layman questions
        intents = {
            "count_schemes": ["How many schemes are there?", "Tell me the total schemes", "What's the scheme count?"],
            "group_by_category": ["What are the scheme categories?", "Categorize the schemes", "Show me scheme categories"],
            "groupby_by_village_scheme_category": ["Tell me about SVS or MVS schemes", "Details about BVS schemes","How many svs schemes are there",'how many mvs schemes are there'],
            "group_by_status": ["What is the completion status?", "How many completed?", "Status of the schemes"],
            "group_by_sanction_year": ["How many schemes were sanctioned per year?", "Sanctioned year-wise info"],
            "sum_expenditure": ["Total money spent?", "What is the overall expenditure?", "How much cost after 2019","total expenditure"],
            "summarize_by_water_scheme_type": ["Summarize schemes by type", "Types of schemes summary"],
            "summarize_by_water_scheme_source": ["Summarize based on water source", "Scheme by source"],
            "summarize_by_water_scheme_funded": ["Who funds the schemes?", "Funding-based scheme summary",'funding'],
            "get_scheme_details_by_id": ["Tell me about scheme ID ", "Give scheme details", "Scheme info for ID"],
            "get_scheme_details_by_name": ["Tell me about scheme name ", "Give details for scheme name","Give me the details of scheme","What are the details of the scheme called"]
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
            "get_scheme_details_by_name": lambda df: self.get_scheme_details_by_name(df, question),
            "group_by_scheme_type": self.group_by_scheme_type
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

    def group_by_scheme_type(self, df):
        category_col = self.get_matching_column(df, ['Type of scheme'])
        if not category_col:
            return "Category column is not available in the data."
        result = df[category_col].value_counts().to_string()
        return f"Schemes grouped by Scheme Type:\n{result}"

    def groupby_by_village_scheme_category(self, df, question):
        category_col = self.get_matching_column(df, ['Village Scheme Type'])
        if not category_col:
            return "Village Scheme Type column is not available in the data."
        if 'svs' in question.lower():
            data_df = df[df[category_col].str.lower() == 'svs']
        elif 'mvs' in question.lower():
            data_df = df[df[category_col].str.lower() == 'mvs']
        elif 'bvs' in question.lower():
            data_df = df[df[category_col].str.lower() == 'bvs']
        else:
            data_df = df
        result = data_df[category_col].value_counts().to_string()
        return f"Schemes grouped by Village Scheme Category:\n{result}"

    def group_by_status(self, df):
        status_col = self.get_matching_column(df, ['Status Of Completion', 'Physical Work Status'])
        if not status_col:
            return "Status column is not available in the data."
        result = df[status_col].value_counts().to_string()
        return f"Schemes grouped by Status:\n{result}"

    def group_by_sanction_year(self, df):
        # Assuming 'Sanction Year' is already available in the dataframe
        year_counts = df['Sanction Year'].value_counts().sort_index()

        if year_counts.empty:
            return "No valid sanction year data found."

        formatted_result = "Schemes sanctioned per year:\n\n"
        
        for year, count in year_counts.items():
            # Handle non-integer years (such as ranges like '2007-2008' or invalid data)
            try:
                # Check if the year is a range (like '2007-2008')
                if isinstance(year, str) and '-' in year:
                    formatted_result += f"  • {year}: {count} schemes \n"
                else:
                    formatted_result += f"  • {int(year)}: {count} schemes\n"
            except ValueError:
                # If it's not a valid integer year, handle gracefully
                formatted_result += f"  • Invalid Year '{year}': {count} schemes\n"

        return formatted_result



    def sum_expenditure(self, df, question):
        if "central" in question.lower() or "spent by government" in question.lower():
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
        match = re.search(r'\b\d+\b', question)

        if match:
            scheme_id = int(match.group(0))  # Convert extracted ID to integer
            
            # Filter the DataFrame
            filtered_df = df[df['SchemeId'] == scheme_id]
            
            # Convert the result to a dictionary
            scheme_details = filtered_df.to_dict(orient='records', indent=4)

            print(scheme_details)

        return f"Details for Scheme ID '{match}':\n\n{scheme_details}"

    def get_scheme_details_by_name(self, df, question):
        """
        Extract scheme names from quoted text in the question and return matching rows from DataFrame.
        """
        # Extract text within matching quotes (single or double)
        pattern = r'(["\'])(.*?)\1'
        matches = re.findall(pattern, question)

        if not matches:
            return "No valid scheme name found in the question."

        # Extract scheme names (second group from match tuples)
        scheme_names = [m[1].strip() for m in matches]
        print(f"Extracted scheme names: {scheme_names}")

        # Try to identify the correct column name
        col = self.get_matching_column(df, ['Scheme Name', 'SchemeTitle', 'Name'])
        if not col:
            return "Scheme name column not found in the data."

        # Search for all matching rows
        results = df[df[col].isin(scheme_names)]
        if results.empty:
            return f"No scheme(s) found matching: {', '.join(scheme_names)}"

        return f"Details for Scheme(s):\n{json.dumps(results.to_dict(orient='records'), indent=4)}"    

    def get_matching_column(self, df, possible_names):
        """Finds the matching column from a list of possible names."""
        df_cols_clean = df.columns.str.strip().str.lower()
        for name in possible_names:
            if name.lower() in df_cols_clean.tolist():
                return next((col for col in df.columns if col.strip().lower() == name.lower()), None)
        return None

    def llm_query(self, df, question):
        """Queries the LLM if the intent does not match."""
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
