import re
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz # For fuzzy string matching
from spellchecker import SpellChecker # For spell correction
from torch.nn.functional import cosine_similarity
import json
import numpy as np # Import numpy for array operations

class IntelligentQueryEngine:
    
    def __init__(self, llama_api_url, df: pd.DataFrame, text_columns, state_filter=None, state_column_name='state name', district_column_name='district', vector_store=None, embedder=None):
        self.llama_api_url = llama_api_url
        self.df = df
        self.text_columns = text_columns
        self.state_filter = state_filter
        self.district_filter = None # Initialize district filter
        self.embedder = embedder if embedder is not None else SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = vector_store
        self.state_column_name = state_column_name.lower()
        self.district_column_name = district_column_name.lower() # New attribute
        self.spell = SpellChecker() # Initialize spell checker

        # Robustly pre-load unique states and districts for faster lookup
        # Check for state column existence
        state_col_for_init = self.get_matching_column(self.df, ['State Name'])
        if state_col_for_init:
            self.unique_states = self.df[state_col_for_init].dropna().str.lower().unique().tolist()
        else:
            self.unique_states = []
            print(f"Warning: State column '{state_column_name}' or 'State Name' not found during initialization.")

        # Check for district column existence
        district_col_for_init = self.get_matching_column(self.df, ['District', 'Zila'])
        if district_col_for_init:
            self.unique_districts = self.df[district_col_for_init].dropna().str.lower().unique().tolist()
        else:
            self.unique_districts = []
            print(f"Warning: District column (e.g., 'District', 'Zila') not found during initialization.")


    def set_state_filter(self, state_name: str):
        self.state_filter = state_name
        self.district_filter = None # Clear district filter if state is set

    def set_district_filter(self, district_name: str): # New method for district filtering
        self.district_filter = district_name
        self.state_filter = None # Clear state filter if district is set

    def ask(self, question: str):
        original_question = question.strip()

        # 1. Spell Correction
        words = original_question.split()
        corrected_words = [self.spell.correction(word) if self.spell.correction(word) is not None else word for word in words]
        question = " ".join(corrected_words)
        print(f"Original question: {original_question}")
        print(f"Corrected question: {question}")

        df_filtered = self.df.copy() # Work on a copy of the dataframe
        question_lower = question.lower()

        # 2. Location Extraction and Filtering (Prioritize District)
        detected_location_name = None

        # Try to detect district first
        district_col_name = self.get_matching_column(df_filtered, ['District', 'Zila'])
        if district_col_name: # Only proceed if a district column is found
            best_district_match = None
            highest_district_score = 0
            for district in self.unique_districts:
                # Use fuzz.partial_ratio for sub-string matching, as questions might contain full sentences
                score = fuzz.partial_ratio(question_lower, district) 
                if district in question_lower: # Direct match for precision
                    best_district_match = district
                    highest_district_score = 100
                    break
                elif score > highest_district_score and score > 75: # Fuzzy match threshold
                    highest_district_score = score
                    best_district_match = district

            if best_district_match:
                self.set_district_filter(best_district_match)
                detected_location_name = best_district_match
                print(f"Detected district: {best_district_match}")
                # Remove the detected location from the question for intent recognition
                # Use re.sub to remove all occurrences of the detected location
                question_lower = re.sub(r'\b' + re.escape(best_district_match) + r'\b', '', question_lower).strip()
                
                # If district found, filter the DataFrame
                df_filtered = df_filtered[df_filtered[district_col_name].str.lower() == self.district_filter.lower()].reset_index(drop=True)
                if df_filtered.empty:
                    return f"Sorry, no data is available for the district: {self.district_filter}."
        
        # If no district detected or no district column, try to detect state
        if not detected_location_name:
            state_col_name = self.get_matching_column(df_filtered, ['State Name'])
            if state_col_name: # Only proceed if a state column is found
                best_state_match = None
                highest_state_score = 0
                for state in self.unique_states:
                    # Use fuzz.partial_ratio for sub-string matching
                    score = fuzz.partial_ratio(question_lower, state) 
                    if state in question_lower: # Direct match for precision
                        best_state_match = state
                        highest_state_score = 100
                        break
                    elif score > highest_state_score and score > 75: # Fuzzy match threshold
                        highest_state_score = score
                        best_state_match = state
                
                if best_state_match:
                    self.set_state_filter(best_state_match)
                    detected_location_name = best_state_match
                    print(f"Detected state: {best_state_match}")
                    # Remove the detected location from the question for intent recognition
                    question_lower = re.sub(r'\b' + re.escape(best_state_match) + r'\b', '', question_lower).strip()
                    
                    # If state found, filter the DataFrame
                    df_filtered = df_filtered[df_filtered[state_col_name].str.lower() == self.state_filter.lower()].reset_index(drop=True)
                    if df_filtered.empty:
                        return f"Sorry, no data is available for the state: {self.state_filter}."

        # If no location was detected in the question itself, but filters are pre-set
        if not detected_location_name and self.district_filter:
            district_col_name = self.get_matching_column(df_filtered, ['District', 'Zila'])
            if district_col_name:
                df_filtered = df_filtered[df_filtered[district_col_name].str.lower() == self.district_filter.lower()].reset_index(drop=True)
                if df_filtered.empty:
                    return f"Sorry, no data is available for the district: {self.district_filter}."
            else:
                return "District column not found in data for filtering."
        elif not detected_location_name and self.state_filter:
            state_col_name = self.get_matching_column(df_filtered, ['State Name'])
            if state_col_name:
                df_filtered = df_filtered[df_filtered[state_col_name].str.lower() == self.state_filter.lower()].reset_index(drop=True)
                if df_filtered.empty:
                    return f"Sorry, no data is available for the state: {self.state_filter}."
            else:
                return "State column not found in data for filtering."

        # If after all filtering, the dataframe is empty, return early
        if df_filtered.empty:
            return "No data available after applying filters."

        # 3. Intent Recognition
        intents = {
            "count_schemes": [
                "How many schemes are there?", "Tell me the total schemes",
                "Number of schemes", "Total schemes count", "How many schemes?", "Count of schemes",
                "schemes count", "total number of schemes", "give me the count of all schemes" # Added more examples
            ],
            "group_by_category": [
                "What are the scheme categories?", "Categorize the schemes",
                "Scheme categories", "List categories", "Categories of schemes", "breakdown by category", "show schemes by type" # Added more examples
            ],
            "groupby_by_village_scheme_category": [
                "SVS or MVS schemes", "Details about BVS schemes",
                "SVS schemes", "MVS schemes", "BVS schemes", "Village scheme types",
                "Types of village schemes", "Village scheme category breakdown", "what kind of village schemes are there?" # Added more examples
            ],
            "group_by_status": [
                "What is the completion status?", "How many completed?",
                "Completion status", "Scheme status", "Status of schemes", "show me the status of completion", "  " # Added more examples
            ],
            "group_by_sanction_year": [
                "Sanctioned per year", "Sanctioned year-wise info",
                "Schemes by sanction year", "Sanction year breakdown", "Year of sanction", "when were schemes approved?", "sanction year report" # Added more examples
            ],
            "sum_expenditure": [
                "Total money spent", "How much cost after 2019",
                "Expenditure summary", "Total spending", "Cost breakdown", "Sum of expenditure",
                "Amount spent", "Central expenditure", "Expenditure after 2019", "how much has been spent?", "total funds utilized" # Added more examples
            ],
            "summarize_by_water_scheme_type": [
                "Summarize schemes by type", "Types of water schemes", "Water scheme types", "what water schemes are there?", "list water scheme types" # Added more examples
            ],
            "summarize_by_water_scheme_source": [
                "Scheme by source", "Water scheme sources", "Sources of schemes", "where do schemes get water from?", "water sources for schemes" # Added more examples
            ],
            "summarize_by_water_scheme_funded": [
                "Who funds the schemes?", "Funding", "Scheme funding", "Funding sources", "scheme funding details", "who provides funds for schemes?" # Added more examples
            ],
            "get_scheme_details_by_id": [
                "Tell me about scheme ID", "Scheme info for ID", "Details for ID",
                "Scheme ID details", "Info on scheme ID", "find scheme by ID", "what is scheme X?" # Added more examples
            ],
            "get_scheme_details_by_name": [
                "Tell me about scheme name", "Give details of scheme",
                "Scheme info for name", "Details of scheme", "Info on scheme", "what about scheme [name]?", "describe [scheme name]" # Added more examples
            ],
            "group_by_scheme_type": [
                "Group by scheme type", "Scheme types", "What are the scheme types?", "classify schemes by type" # Added more examples
            ]
        }

        all_examples, intent_map = [], []
        for intent, examples in intents.items():
            all_examples.extend(examples)
            intent_map.extend([intent] * len(examples))

        example_embeddings = self.embedder.encode(all_examples, convert_to_tensor=True)
        # Use the question_lower that might have had location removed
        question_embedding = self.embedder.encode([question_lower], convert_to_tensor=True)
        similarities = cosine_similarity(question_embedding, example_embeddings).squeeze(0)
        best_match_idx = similarities.argmax().item()
        best_intent = intent_map[best_match_idx]
        best_score = similarities[best_match_idx].item()

        print(f"Best intent: {best_intent}, Score: {best_score:.2f}")

        if best_score < 0.5: # Adjusted threshold if needed
            return self.llm_query(df_filtered, original_question) # Pass original question to LLM

        intent_functions = {
            "count_schemes": self.count_schemes,
            "group_by_category": self.group_by_category,
            "groupby_by_village_scheme_category": lambda df: self.groupby_by_village_scheme_category(df, original_question), # Pass original question
            "group_by_status": self.group_by_status,
            "group_by_sanction_year": self.group_by_sanction_year,
            "sum_expenditure": lambda df: self.sum_expenditure(df, original_question), # Pass original question
            "summarize_by_water_scheme_type": self.summarize_by_water_scheme_type,
            "summarize_by_water_scheme_source": self.summarize_by_water_scheme_source,
            "summarize_by_water_scheme_funded": self.summarize_by_water_scheme_funded,
            "get_scheme_details_by_id": lambda df: self.get_scheme_details_by_id(df, original_question), # Pass original question
            "get_scheme_details_by_name": lambda df: self.get_scheme_details_by_name(df, original_question), # Pass original question
            "group_by_scheme_type": self.group_by_scheme_type
        }

        # Call the identified intent function with the filtered DataFrame
        return intent_functions.get(best_intent, lambda df: self.llm_query(df, original_question))(df_filtered)

    def llm_query(self, df, question, top_k_context=5):
        if self.vector_store is None or self.embedder is None:
            print("Warning: Vector store or embedder not provided. Falling back to head(5) for context.")
            valid_text_cols = [col for col in self.text_columns if col in df.columns]
            if not valid_text_cols:
                return "No valid columns available for generating context."
            context_df = df[valid_text_cols].head(top_k_context)
            
        else:
            # Generate embeddings for the current filtered DataFrame
            # Concatenate all text columns for embedding
            df_text_for_embedding = df[self.text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
            if not df_text_for_embedding:
                return "No text data available in the filtered DataFrame for generating context."
            
            # Create a temporary FAISS index for the *filtered* DataFrame embeddings
            # This is crucial because the main vector_store contains embeddings for the *entire* dataset.
            # We need to search only within the subset of data relevant to the current filters.
            
            # Generate embeddings for the filtered dataframe's text columns
            filtered_df_embeddings = self.embedder.encode(df_text_for_embedding, convert_to_tensor=False)
            
            # Create a temporary in-memory FAISS index for filtered data
            # Ensure the dimension matches the embeddings
            dim = filtered_df_embeddings.shape[1]
            temp_index = faiss.IndexFlatL2(dim)
            temp_index.add(filtered_df_embeddings.astype('float32')) # FAISS expects float32
            
            # Embed the user's question
            query_embedding = self.embedder.encode([question], convert_to_tensor=False).astype('float32')
            
            # Search the temporary index
            distances, indices = temp_index.search(query_embedding, min(top_k_context, len(df_filtered)))
            
            # Retrieve the relevant rows from the original filtered DataFrame based on these indices
            relevant_rows = df_filtered.iloc[indices[0]]
            context_df = relevant_rows[self.text_columns].fillna('')

        # Convert context_df to a list of strings for the prompt
        context = context_df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
        
        prompt = f"""You are an intelligent, helpful assistant analyzing water supply schemes.
Your goal is to provide concise, accurate, and relevant answers to user questions based *only* on the provided data.
If the information is not explicitly available in the data, state that you don't have the information.

Data Context (relevant scheme details):
{context}

User Question:
{question}

Answer:
"""
        try:
            response = requests.post(self.llama_api_url, json={
                "text-input": prompt,
                "max-tokens": 4096,
                "temperature": 0.2 # Lower temperature for more deterministic/factual answers
            })
            if response.status_code == 200:
                return response.json().get("output", "No answer generated by LLM.")
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
        # Filter based on keywords in the original question
        df_filtered_by_keyword = df.copy()
        found_keyword_filter = False
        for keyword in ['svs', 'mvs', 'bvs']:
            if keyword in question.lower():
                df_filtered_by_keyword = df_filtered_by_keyword[df_filtered_by_keyword[col].str.lower() == keyword]
                found_keyword_filter = True
                break # Apply only the first matching keyword
        
        if not found_keyword_filter: # If no specific keyword, return general counts
            return df[col].value_counts().to_string()
        elif df_filtered_by_keyword.empty:
            return f"No schemes found for the specified village scheme type."
        else:
            return df_filtered_by_keyword[col].value_counts().to_string()


    def group_by_status(self, df):
        col = self.get_matching_column(df, ['Status Of Completion', 'Physical Work Status'])
        return df[col].value_counts().to_string() if col else "Status column not found."

    def group_by_sanction_year(self, df):
        # Convert to numeric, coercing errors, then drop NaNs for meaningful count
        sanction_year_col = self.get_matching_column(df, ['Sanction Year'])
        if not sanction_year_col:
            return "Sanction Year column not found."
        
        # Ensure 'Sanction Year' is numeric, handling non-numeric entries
        df[sanction_year_col] = pd.to_numeric(df[sanction_year_col], errors='coerce')
        
        # Drop rows where 'Sanction Year' became NaN after coercion for counts
        df_cleaned = df.dropna(subset=[sanction_year_col])
        
        if df_cleaned.empty:
            return "No valid sanction year data available."
            
        # Convert to int after dropping NaNs and before value_counts, as year should be integer
        counts = df_cleaned[sanction_year_col].astype(int).value_counts().sort_index()
        return "\n".join(f"• {k}: {v} schemes" for k, v in counts.items())

    def sum_expenditure(self, df, question):
        col = None
        if "central" in question.lower():
            col = self.get_matching_column(df, ['Total central expenditure (in lakhs)'])
        elif "after 2019" in question.lower() or "2019-20" in question.lower():
            col = self.get_matching_column(df, ['Total expenditure (in lakhs) on or after 2019-20'])
        else:
            col = self.get_matching_column(df, ['Total expenditure (in lakhs)'])
        if not col:
            return "Expenditure column not found."
        # Ensure the column is numeric before summing, coerce errors to NaN
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
        id_col = self.get_matching_column(df, ['SchemeId', 'Scheme ID']) # Added 'Scheme ID' as a possible name
        if not id_col:
            return "Scheme ID column not found."
        
        # Ensure the ID column is numeric for comparison
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
        
        results = df[df[id_col] == scheme_id]
        return json.dumps(results.to_dict(orient='records'), indent=4) if not results.empty else "Scheme ID not found."

    def get_scheme_details_by_name(self, df, question):
        # Improved regex to capture names that might not be in quotes and are part of the general query
        # This will try to extract anything that looks like a scheme name, even if not explicitly quoted.
        # It's a heuristic and might need fine-tuning based on typical user queries.
        matches = re.findall(r'(?:scheme\s*name is|scheme\s*called|about scheme|scheme|details\s*of|name\s*of)?\s*["\']?([^"\']+)["\']?', question, re.IGNORECASE)
        scheme_names = [m.strip().lower() for m in matches if m.strip() and len(m.strip()) > 2] # Filter out short matches
        
        if not scheme_names:
            # Fallback: simple keyword extraction for very short or direct queries
            keywords = [word for word in question.lower().split() if len(word) > 2 and not word.isdigit() and word not in ["scheme", "details", "info", "about", "tell", "me", "give", "of"]]
            if keywords:
                # This fallback might be too broad; specific examples are better.
                # Consider adding common scheme name patterns to intents for better structured recognition.
                scheme_names.append(" ".join(keywords))

        if not scheme_names:
            return "No scheme name found in the query."

        col = self.get_matching_column(df, ['Scheme Name', 'SchemeTitle', 'Name'])
        if not col:
            return "Scheme name column not found in data."
        
        # Use fuzzy matching for scheme names
        results = pd.DataFrame()
        for query_name in scheme_names:
            # Find best matching scheme name in the column
            best_match_scheme = None
            highest_score = 0
            for actual_name in df[col].dropna().unique():
                score = fuzz.ratio(query_name, actual_name.lower())
                # Using fuzz.partial_ratio for cases where the user query might be a subset of the actual scheme name
                partial_score = fuzz.partial_ratio(query_name, actual_name.lower())
                
                # Prioritize exact ratio, but consider partial ratio if it's very high
                current_score = max(score, partial_score)

                if current_score > highest_score and current_score > 75: # Threshold for a good match
                    highest_score = current_score
                    best_match_scheme = actual_name
            
            if best_match_scheme:
                # Use .loc to avoid SettingWithCopyWarning and ensure consistent indexing
                matched_rows = df[df[col].str.lower() == best_match_scheme.lower()]
                results = pd.concat([results, matched_rows])
        
        return json.dumps(results.to_dict(orient='records'), indent=4) if not results.empty else f"No matching schemes found for: {', '.join(scheme_names)}."


    def get_matching_column(self, df, possible_names):
        df_cols_clean = df.columns.str.strip().str.lower()
        for name in possible_names:
            # First, check for an exact case-insensitive match
            if name.lower() in df_cols_clean.tolist():
                return next((col for col in df.columns if col.strip().lower() == name.lower()), None)
            
            # If no exact match, try fuzzy matching with a high threshold
            for col_name in df_cols_clean:
                if fuzz.ratio(name.lower(), col_name) > 90: # High threshold for close match
                    return next((c for c in df.columns if c.strip().lower() == col_name), None)
        return None