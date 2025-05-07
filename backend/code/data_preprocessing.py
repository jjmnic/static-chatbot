import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# ----- Data Processing Functions -----

def load_data(file_path):
    """
    Load data from CSV file with improved error handling and format detection
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
   
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try to infer the delimiter by reading the first few lines
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(5000)  # Read first 5000 characters
    
    # Check for common delimiters in the sample
    delimiters = [',', '\t', ';', '|']
    delimiter_counts = {d: sample.count(d) for d in delimiters}
    likely_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    
    print(f"Detected delimiter: '{likely_delimiter}' (counts in sample: {delimiter_counts})")
    
    try:
        # First attempt with the detected delimiter
        df = pd.read_csv(file_path, delimiter=likely_delimiter, encoding='utf-8')
        
        # If we got very few columns but many rows, the delimiter might be wrong
        if len(df.columns) == 1 and df.shape[0] > 10:
            print("Single column detected, trying with pandas' csv sniffer...")
            df = pd.read_csv(file_path, encoding='utf-8', engine='python')
    
    except Exception as e:
        print(f"Error with detected delimiter, falling back to pandas default: {str(e)}")
        # Fallback to pandas default behavior which tries to infer the dialect
        try:
            df = pd.read_csv(file_path, encoding='utf-8', engine='python')
        except Exception as inner_e:
            # Last resort: try excel format
            print(f"CSV reading failed, trying Excel format: {str(inner_e)}")
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except Exception as xl_e:
                raise ValueError(f"Failed to load data file with all attempts: {str(xl_e)}")
    
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Column names: {df.columns.tolist()}")
    
    return df

def preprocess_data(df):
    """
    Preprocesses the water scheme dataset:
    - Renames columns
    - Converts date columns to datetime
    - Converts numeric columns to numbers
    - Drops fully blank rows
    - Fills missing values appropriately
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Strip whitespace from headers
    df.columns = df.columns.str.strip()

    # Save column names for debugging
    print("Original column names before renaming:")
    print(df.columns.tolist())

    # Rename specific columns
    rename_dict = {
        "Work not awarded/ Ongoing/ Financially completed": "Status Of Completion",
        "New scheme/ Retrofit/ Augmentation": "Type of Scheme",
        "SVS/ MVS/ Bulk Water Schemes": "Category",
        "Ground/ Surface water/ Bulk Water Based/ Other": "Source of Scheme",
        "NRDWP/ State and Others/ JJM-PWS/ JJM-Non-PWS": "Main Schemes Funded From",
        "Physically completed/ Ongoing but physically not completed/ Work order not issued": "Physical Work Status"
    }
    
    # Check which columns from the rename dict exist in the dataframe
    existing_cols = [col for col in rename_dict.keys() if col in df.columns]
    print(f"Found {len(existing_cols)} out of {len(rename_dict)} columns to rename")
    
    # Only rename columns that exist
    rename_dict_filtered = {k: v for k, v in rename_dict.items() if k in df.columns}
    df.rename(columns=rename_dict_filtered, inplace=True)
    
    print("Column names after renaming:")
    print(df.columns.tolist())

    # Drop fully blank rows
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Date columns to convert
    date_columns = [
        'SLSSC/ DWSSM meeting date (dd/mm/yyyy)',
        'Work order date (dd/mm/yyyy)',
        'Physical completion date',
        'Tentative completion date',
        'Last FHTC reported Month',
        'Last Expenditure reported Month',
    ]

    # Process only date columns that exist in the dataframe
    existing_date_cols = [col for col in date_columns if col in df.columns]
    print(f"Processing {len(existing_date_cols)} date columns")
    
    for col in existing_date_cols:
        print(f"Converting '{col}' to datetime")
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
        # Fill NaT with 'Not Provided' for date columns
        df[col] = df[col].apply(lambda x: x.strftime('%d-%m-%Y') if pd.notnull(x) else 'Not Provided')

    # Numeric columns to convert
    numeric_columns = [
        'Estimated cost (in lakhs) as per work order',
        'Derived estimated cost  (in lakhs)',
        'Total_inadmissible_cost  (in lakhs)',
        'Derived estimated cost after removing inadmisible cost loaded on JJM  (in lakhs)',
        'Total expenditure (in lakhs)',
        'Total central expenditure (in lakhs)',
        'Total expenditure (in lakhs) on or after 2019-20',
        'Total central expenditure (in lakhs) on or after 2019-20',
        'FHTCS planned',
        'FHTCS provided',
        'In-village infrastructure cost (in lakhs) (After financial authentication)',
        'Central share cost (in lakhs) (After financial authentication)',
        'Community contribution (in lakhs) (After financial authentication)',
        'Physical completion progress (In percentage)'
    ]
    print(df.dtypes)
    # Process only numeric columns that exist in the dataframe
    existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
    print(f"Processing {len(existing_numeric_cols)} numeric columns")
    
    for col in existing_numeric_cols:
        print(f"Converting '{col}' to numeric")
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill NaN in numeric columns with 0
        df[col].fillna(0, inplace=True)

    # # Fill NaN values for object columns with empty string
    # for col in df.columns:
    #     if df[col].dtype == 'object':  # For text columns (non-numeric)
    #         missing_count = df[col].isna().sum()
    #         if missing_count > 0:
    #             print(f"Filling {missing_count} missing values in column '{col}' with empty string")
    #             df[col].fillna('Not Specified', inplace=True)

    # Create a dictionary for remaining columns that need specific NA values
    # Note: Check that these columns actually exist and use correct capitalization
    fillna_dict = {}
    
    # Check and add columns to fillna_dict only if they exist and still have NAs
    # For text columns
    text_fill_values = {
        'Type of Scheme': 'Not Specified',       # Corrected capitalization
        'Source of Scheme': 'Not Specified',
        'Category': 'Not Specified',
        'Main Schemes Funded From': 'Not Specified',
        'Physical Work Status': 'Not Specified',
        'Status Of Completion': 'Not Specified',   # Added this from rename dict
        'Un-verified status':'Not Specified',
        'Last FHTC reported Year':'Not Specified',
        'Last Expenditure reported Year':'Not Specified',
        
    }
    
    for col, fill_val in text_fill_values.items():
        if col in df.columns and df[col].isna().any():
            fillna_dict[col] = fill_val
            print(f"Will fill NA values in '{col}' with '{fill_val}'")
    
    # Now fill the NA values if we have any columns to fill
    if fillna_dict:
        df.fillna(fillna_dict, inplace=True)
        print(f"Filled NA values in {len(fillna_dict)} columns with specified values")
    
    # Verify that all NAs have been filled
    na_counts = df.isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    
    if len(cols_with_na) > 0:
        print("\nWarning: These columns still have NA values:")
        print(cols_with_na)
    else:
        print("\nAll NA values have been successfully filled!")

    # Apply lowercase conversion to all string fields (optional)
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    print("Data Preprocessing done successfully!")
    return df