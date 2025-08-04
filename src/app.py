import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from io import BytesIO
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utility import *
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Initialize LLM model and tokenizer
# def load_llm_model():
#     """Load the Mistral model and tokenizer from Hugging Face"""
#     model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
#     # Pass token to both tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name, 
#         trust_remote_code=True,
#         token=hf_token  # Make sure hf_token is defined
#     )
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto" if torch.cuda.is_available() else None,
#         trust_remote_code=True,
#         token=hf_token  # Add token here too!
#     )
    
#     return tokenizer, model
hf_token = "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"
@st.cache_resource
def load_llm_model():
    
    """Load the Mistral-7B-Instruct model and tokenizer from Hugging Face"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
        trust_remote_code=True,
        token=hf_token)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
         model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        token=hf_token
    )
    return tokenizer, model

# LLM Query Processing Functions
def is_query_vague(query, tokenizer, model):
    """Use LLM to determine if a query is vague or needs clarification"""
    prompt = f"""[INST] You are a helpful assistant that determines if a user query about data analysis is clear or vague.
Respond with only "CLEAR" or "VAGUE" followed by a brief explanation.

A query is CLEAR if it has specific filtering criteria, analysis type, or clear intent.
A query is VAGUE if it lacks specifics about what data to look at or what analysis to perform.

Query: "{query}" [/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response.split("[/INST]")[-1].strip()
    
    return "VAGUE" in response_text.upper(), response_text

def ask_for_clarification(query, tokenizer, model):
    """Generate a clarification question for vague queries"""
    prompt = f"""[INST] You are a helpful assistant for a diamond inventory analysis system. 
The user has asked a vague question and you need to ask for clarification.
Ask a specific question to understand what they want to analyze.

The user asked: "{query}"

Generate a clarification question to understand their specific needs regarding:
- Which time period (month/year/quarter)
- Which product attributes (shape, color, bucket)
- What type of analysis (pricing, inventory gaps, trends, etc.) [/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

def process_data_query(query, df, tokenizer, model):
    """Process user query against the data using LLM"""
    # Get data info for context
    columns = df.columns.tolist()
    shape_keys = df['Shape key'].unique().tolist() if 'Shape key' in df.columns else []
    color_keys = df['Color Key'].unique().tolist() if 'Color Key' in df.columns else []
    buckets = df['Buckets'].unique().tolist() if 'Buckets' in df.columns else []
    months = df['Month'].unique().tolist() if 'Month' in df.columns else []
    years = df['Year'].unique().tolist() if 'Year' in df.columns else []
    
    # Create a summary of the data
    data_summary = f"""
    Columns: {', '.join(columns[:20])}...
    Shape Keys: {', '.join(map(str, shape_keys[:10]))}
    Color Keys: {', '.join(map(str, color_keys[:10]))}
    Buckets: {', '.join(map(str, buckets[:10]))}
    Months: {', '.join(map(str, months[:10]))}
    Years: {', '.join(map(str, years[:10]))}
    Total Records: {len(df)}
    """
    
    prompt = f"""[INST] You are a data analysis assistant for a diamond inventory system. 
You help users filter and analyze data based on their queries.
Generate Python code that will filter or analyze the dataframe based on the user's request.
The dataframe is called 'df' and is already loaded.

Available data context:
{data_summary}

Important columns for analysis:
- Shape key: Diamond shape categories
- Color Key: Color classifications
- Buckets: Weight/size categories
- Month, Year, Quarter: Time dimensions
- Max Buying Price, Min Selling Price: Pricing data
- Max Qty, Min Qty: Inventory limits
- Weight, Product Id: Product details

Return only executable Python code that:
1. Filters the dataframe if needed
2. Performs the requested analysis
3. Returns results as 'result' variable (either a dataframe or a summary dict)

User Query: {query} [/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    code_response = response.split("[/INST]")[-1].strip()
    
    # Extract Python code from response
    code_match = re.search(r'```python\n(.*?)\n```', code_response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # If no code blocks, assume entire response is code
        code = code_response
    
    return code

def execute_llm_code(code, df):
    """Safely execute the LLM-generated code"""
    # Create a safe execution environment
    safe_globals = {
        'df': df,
        'pd': pd,
        'np': np,
        'datetime': datetime,
    }
    
    safe_locals = {}
    
    try:
        exec(code, safe_globals, safe_locals)
        result = safe_locals.get('result', None)
        if result is None:
            # Try to get any DataFrame variable
            for var_name, var_value in safe_locals.items():
                if isinstance(var_value, pd.DataFrame):
                    result = var_value
                    break
        return result, None
    except Exception as e:
        return None, str(e)

def generate_analysis_summary(query, result, tokenizer, model):
    """Generate a natural language summary of the analysis results"""
    if isinstance(result, pd.DataFrame):
        result_summary = f"DataFrame with {len(result)} rows and {len(result.columns)} columns"
        if len(result) > 0:
            result_summary += f"\nFirst few rows:\n{result.head().to_string()}"
    else:
        result_summary = str(result)
    
    prompt = f"""[INST] You are a helpful assistant that summarizes data analysis results in clear, business-friendly language.

User Query: {query}

Analysis Result:
{result_summary}

Provide a clear, concise summary of the findings. [/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# File history management functions (keeping existing functions)
def get_history_file_path():
    """Get the path for the history JSON file"""
    history_dir = Path("history")
    history_dir.mkdir(exist_ok=True)
    return history_dir / "upload_history.json"

def load_upload_history() -> List[Dict]:
    """Load upload history from JSON file"""
    history_file = get_history_file_path()
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading history: {e}")
            return []
    return []

def save_upload_history(history: List[Dict]):
    """Save upload history to JSON file"""
    history_file = get_history_file_path()
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")

def add_to_upload_history(filename: str, file_size: int = None):
    """Add a new file to upload history, maintaining max 10 entries"""
    history = load_upload_history()
    
    # Create new entry
    new_entry = {
        "filename": filename,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": file_size,
        "status": "Processed"
    }
    
    # Add to beginning of list
    history.insert(0, new_entry)
    
    # Keep only last 10 entries
    history = history[:10]
    
    # Save updated history
    save_upload_history(history)
    
    return history

def display_upload_history():
    """Display the upload history in the sidebar"""
    history = load_upload_history()
    
    if history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Recent Upload History")
        st.sidebar.markdown("*Last 10 uploaded files*")
        
        for idx, entry in enumerate(history, 1):
            with st.sidebar.expander(f"{idx}. {entry['filename']}", expanded=False):
                st.write(f"**Uploaded:** {entry['upload_time']}")
                if entry.get('file_size'):
                    size_mb = entry['file_size'] / (1024 * 1024)
                    st.write(f"**Size:** {size_mb:.2f} MB")
                st.write(f"**Status:** {entry['status']}")
    else:
        st.sidebar.markdown("---")
        st.sidebar.info("No upload history available")

# [Keeping all existing functions from load_data through create_summary_charts - not repeating them here for brevity]
# Include all functions from load_data to create_summary_charts from the original code

def load_data(file):
    # Handle different input types
    if isinstance(file, str):
        # String file path (for database files)
        file_type = file.split('.')[-1]
        if file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'pkl':
            df = pd.read_pickle(f"src/{file}")
            return df
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file, sheet_name=None)
            df_dict = {}
            for sheet_name, df_ in df.items():
                df_dict[sheet_name] = df_
            # st.info(df.keys())
            return df_dict
            
            
            
    else:
        # File object from Streamlit uploader
        if hasattr(file, 'name'):
            file_type = file.name.split('.')[-1]
        else:
            file_type = 'xlsx'  # Default assumption for uploaded files
        
        if file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file, sheet_name=None)
            df_dict = {}
            for sheet_name, df_ in df.items():
                df_dict[sheet_name] = df_
            # st.info(df.keys())
            return df_dict
        elif file_type == 'pkl':
            df = pd.read_pickle(f"src/{file}")
            return df
        elif file_type == 'csv':
            return pd.read_csv(file)


def save_data(df):
    df.to_pickle('src/kunmings.pkl')
def create_color_key(df,color_map):
    df['Color Key'] = df.Color.map(lambda x: color_map[x] if x in color_map else '')
    return df
def create_bucket(df,stock_bucket=stock_bucket):
    """
    df : Monthly Stock Data Sheet
    stock_bucket : Dictionary containing bucket ranges
    """
    for key , values in stock_bucket.items():
        lower_bound , upper_bound = values
        index = df[(df['Weight']>=lower_bound) & (df['Weight']<upper_bound)].index.tolist()
        df.loc[index,'Buckets'] = key
    return df

def calculate_avg(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Avg Cost Total'] = df['Weight'] * df['Average\nCost\n(USD)']
    return df

def create_date_join(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Month'] = pd.to_datetime('today').month_name()
    df['Year'] = pd.to_datetime('today').year
    df['Join'] = df['Month'].astype(str) + '-' + df['Year'].map(lambda x: x-2000).astype(str)
    return df
def concatenate_first_two_rows(df):
    result = {}
    for col in df.columns:
        value1 = str(df.iloc[0][col])
        value2 = str(df.iloc[1][col])
        result[col] = f"{value1}_{value2}"
    return result
def populate_max_qty(df,MONTHLY_STOCK_DATA):
    """
    df : Max Qty Sheet
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _MAX_QTY_ = []
    MONTHLY_STOCK_DATA['Max Qty'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _MAX_QTY_.append(value)
    MONTHLY_STOCK_DATA['Max Qty'] = _MAX_QTY_
    MONTHLY_STOCK_DATA['Max Qty']=MONTHLY_STOCK_DATA['Max Qty'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA

def populate_min_qty(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Min Qty Sheet
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _MIN_QTY_ = []
    MONTHLY_STOCK_DATA['Min Qty'] = None
    for _, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _MIN_QTY_.append(value)
    MONTHLY_STOCK_DATA['Min Qty'] = _MIN_QTY_
    MONTHLY_STOCK_DATA['Min Qty']=MONTHLY_STOCK_DATA['Min Qty'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA
def populate_selling_prices(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Max Prices Sheet 
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,1:]).values())
    columns = ['Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _SELLING_PRICE_ = []
    MONTHLY_STOCK_DATA['Min Selling Price'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _SELLING_PRICE_.append(value)
    MONTHLY_STOCK_DATA['Min Selling Price'] = _SELLING_PRICE_
    MONTHLY_STOCK_DATA['Min Selling Price']=MONTHLY_STOCK_DATA['Min Selling Price'].map(lambda x: x[0] if (isinstance(x,list) and len(x)>0) else 0)
    MONTHLY_STOCK_DATA['Min Selling Price'] = MONTHLY_STOCK_DATA['Max Buying Price'] * MONTHLY_STOCK_DATA['Min Selling Price'] 
    return MONTHLY_STOCK_DATA
def populate_buying_prices(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Max Prices Sheet 
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _BUYING_PRICE_ = []
    MONTHLY_STOCK_DATA['Max Buying Price'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _BUYING_PRICE_.append(value)
    MONTHLY_STOCK_DATA['Max Buying Price'] = _BUYING_PRICE_
    MONTHLY_STOCK_DATA['Max Buying Price']=MONTHLY_STOCK_DATA['Max Buying Price'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA
def calculate_buying_price_avg(df):
    df['Buying Price Avg'] = df['Max Buying Price'] * df['Weight']
    return df

def get_quarter(month):
    Quarter_Month_Map = {
    'Q1': ['January', 'February', 'March'],
    'Q2': ['April', 'May', 'June'],
    'Q3': ['July', 'August', 'September'],
    'Q4': ['October', 'November', 'December']
    }
    year = pd.to_datetime('today').year
    yr = year - 2000

    if month in Quarter_Month_Map['Q1']:
        return f'Q1-{yr}'
    elif month in Quarter_Month_Map['Q2']:
        return f'Q2-{yr}'
    elif month in Quarter_Month_Map['Q3']:
        return f'Q3-{yr}'
    elif month in Quarter_Month_Map['Q4']:
        return f'Q4-{yr}'
    else:
        return None
def populate_quarter(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Quarter'] = df['Month'].apply(get_quarter)
    return df
def create_shape_key(x):
    if x.__contains__(r'HEART'):
        return 'Other'
    elif x.__contains__(r'CUSHION'):
        return 'Cushion'
    elif x.__contains__(r'OVAL'):
        return 'Oval'
    elif x.__contains__(r'PEAR'):
        return 'Pear'
    elif x.__contains__(r'CUT-CORNERED'):
        return 'Radiant'
    elif x.__contains__(r'MODIFIED RECTANGULAR'):
        return 'Cushion'
    elif x.__contains__(r'MODIFIED SQUARE'):
        return 'Cushion'
    
    elif x.__contains__(r'MARQUISE MODIFIED'):
        return 'Other'
    elif x.__contains__(r'ROUND_CORNERED'):
        return 'Cushion'
    elif x.__contains__(r'EMERALD'):
        return 'Other'
    else:
        return 'Other'
def poplutate_monthly_stock_sheet(file):
    """
    df_stock : Monthly Stock Data Sheet
    df_buying : Buying Max Prices Sheet
    df_min_qty : Buying Min Qty Sheet
    df_max_qty : Max Qty Sheet
    """
    df = load_data(file)
    
    df_stock = df['Monthly Stock Data']
    df_stock.rename(columns={'avg': 'Avg Cost Total'}, inplace=True)
    df_buying = df['Buying Max Prices']
    df_min_qty = df['MIN Data']
    df_max_qty = df['MAX Data']
    df_min_sp = df['Min Selling Price']
    if df_stock.empty or df_buying.empty or df_min_qty.empty or df_max_qty.empty:
        raise ValueError("One or more dataframes are empty. Please check the input files.")
    df_stock = create_date_join(df_stock)
    df_stock = populate_quarter(df_stock)
    df_stock = calculate_avg(df_stock)
    df_stock = create_bucket(df_stock)
    df_stock = create_color_key(df_stock, color_map)
    df_stock['Shape key'] = df_stock['Shape'].apply(create_shape_key)
    df_stock = populate_max_qty(df_max_qty, df_stock)
    df_stock = populate_min_qty(df_min_qty, df_stock)
    df_stock = populate_buying_prices(df_buying, df_stock)
    df_stock = calculate_buying_price_avg(df_stock)
    df_stock = populate_selling_prices(df_min_sp,df_stock)
    df_stock.fillna(0,inplace=True)
    return df_stock
def calculate_qoq_variance_percentage(current_quarter_price, previous_quarter_price):
    """
    Calculate quarter-on-quarter variance percentage of price.
    
    Args:
        current_quarter_price (float): Price for the current quarter
        previous_quarter_price (float): Price for the previous quarter
    
    Returns:
        float: Variance percentage (positive for increase, negative for decrease)
        
    Raises:
        ValueError: If previous quarter price is zero or negative
        TypeError: If inputs are not numeric
    """
    # Input validation
    if not isinstance(current_quarter_price, (int, float)) or not isinstance(previous_quarter_price, (int, float)):
        raise TypeError("Both prices must be numeric values")
    
    if previous_quarter_price <= 0:
        variance_percentage = 0.00001
        # raise ValueError("Previous quarter price must be positive (cannot be zero or negative)")
    
    # Calculate variance percentage
    if previous_quarter_price !=0:
        variance_percentage = ((current_quarter_price - previous_quarter_price) / previous_quarter_price) * 100
    else:
        variance_percentage = ((current_quarter_price - previous_quarter_price) / (previous_quarter_price+current_quarter_price)) * 100
    return round(variance_percentage, 2)


def calculate_qoq_variance_series(price_data):
    """
    Calculate quarter-on-quarter variance for a series of quarterly prices.
    
    Args:
        price_data (list): List of quarterly prices in chronological order
    
    Returns:
        list: List of QoQ variance percentages (starts from Q2 since Q1 has no previous quarter)
    """
    if len(price_data) < 2:
        raise ValueError("Need at least 2 quarters of data to calculate variance")
    
    variances = []
    for i in range(1, len(price_data)):
        variance = calculate_qoq_variance_percentage(price_data[i], price_data[i-1])
        variances.append(variance)
    
    return variances
def monthly_variance(df,col):
    analysis=df.groupby(['Month','Year'],as_index=False)[col].sum()
    analysis['Num_Month'] = analysis['Month'].map(month_map)
    analysis.sort_values(by=['Year','Num_Month'],inplace=True)
    analysis['Monthly_change']=analysis[col].pct_change().fillna(0).round(2)*100
    analysis['qaurter_change']=[0]+calculate_qoq_variance_series(analysis[col].tolist())
    return analysis
def gap_analysis(max_qty,min_qty,stock_in_hand):
    """
    max_qty : Maximum Quantity
    min_qty : Minimum Quantity
    stock_in_hand : Stock in Hand
    """
    if stock_in_hand > max_qty:
        excess_qty = stock_in_hand - max_qty
        return excess_qty
    elif stock_in_hand < min_qty:
        deficit_qty = stock_in_hand - min_qty
        return deficit_qty
    else:
        return 0

def get_filtered_data(FILTER_MONTH,FILTE_YEAR,FILTER_SHAPE,FILTER_COLOR,FILTER_BUCKET):
    """
    file : Monthly Stock Data Sheet
    FILTER_MONTH : Month to filter
    FILTE_YEAR : Year to filter
    FILTER_SHAPE : Shape Key to filter
    FILTER_COLOR : Color Key to filter
    FILTER_BUCKET : Buckets to filter
    FILTER_MONTHLY_VAR_COL : Column to calculate monthly variance
    PARENT_DF : Parent DataFrame to concatenate with the monthly stock data
    """
    master_df = load_data('kunmings.pkl')
    if (type(FILTE_YEAR)==str) & (str(FILTE_YEAR).isnumeric()):
        FILTE_YEAR = int(FILTE_YEAR)
    #     FILTE_YEAR = int(FILTE_YEAR)
    #     filter_data=master_df[(master_df['Month'] == FILTER_MONTH) | \
    #                                   (master_df['Year'] == FILTE_YEAR) | \
    #                                     (master_df['Shape key'] == FILTER_SHAPE) |\
    #                                     (master_df['Color Key'] == FILTER_COLOR) |\
    #                                     (master_df['Buckets'] == FILTER_BUCKET)]
    # else:
    #     filter_data=master_df[(master_df['Month'] == FILTER_MONTH) | \
                                      
    #                                     (master_df['Shape key'] == FILTER_SHAPE) |\
    #                                     (master_df['Color Key'] == FILTER_COLOR) |\
    #                                     (master_df['Buckets'] == FILTER_BUCKET)]
    filter_data=master_df[(master_df['Month'] == FILTER_MONTH) & \
                                      (master_df['Year'] == FILTE_YEAR) & \
                                        (master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    max_qty = filter_data['Max Qty'].max()
    min_qty = filter_data['Min Qty'].min()
    stock_in_hand = filter_data.shape[0]
    gap_analysis_op = gap_analysis(max_qty, min_qty, stock_in_hand)
    _filter_ = master_df[(master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    try:
        max_buying_price = filter_data['Max Buying Price'].max()
        current_avg_cost = (sum(filter_data['Avg Cost Total'])/(filter_data['Weight'].sum() if filter_data['Weight'].sum() != 0 else 1))*.9
        min_selling_price = filter_data['Min Selling Price'].min()
        # avg_value = _filter_[FILTER_MONTHLY_VAR_COL].mean()
        # MOM_Variance = (sum((filter_data[FILTER_MONTHLY_VAR_COL] - avg_value)/ avg_value )/filter_data.shape[0]) * 100
        # var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
        # MOM_Percent_Change = var_analysis[(var_analysis['Month'] == FILTER_MONTH) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
        # MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == FILTER_MONTH) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
        # if MOM_Percent_Change == np.inf:
        #     MOM_Percent_Change = 0
        # if MOM_QoQ_Percent_Change == np.inf:
        #     MOM_QoQ_Percent_Change = 0
        return [filter_data,int(max_buying_price),int(current_avg_cost), gap_analysis_op,min_selling_price]
    except:
        return [pd.DataFrame(columns=master_df.columns.tolist()),f"There is {filter_data.shape[0]} rows after filter",f"There is {filter_data.shape[0]} rows after filter",gap_analysis_op,0]
def get_summary_metrics(filter_data,Filter_Month,FILTER_SHAPE,FILTE_YEAR,FILTER_COLOR,FILTER_BUCKET,FILTER_MONTHLY_VAR_COL):
    FILTE_YEAR = int(FILTE_YEAR)
    master_df = load_data('kunmings.pkl')
    _filter_ = master_df[(master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    Prev_Month_Name = None
    for Month_Name, Month_Num in month_map.items():
        prev_month_num = month_map[Filter_Month]-1
        if prev_month_num == Month_Num:
            Prev_Month_Name = Month_Name
    
    Prev_filter_data=master_df[(master_df['Month'] == Prev_Month_Name) & \
                                      (master_df['Year'] == FILTE_YEAR) & \
                                        (master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    try:
        if FILTER_MONTHLY_VAR_COL == 'Current Average Cost':
            FILTER_MONTHLY_VAR_COL='Buying Price Avg'
            avg_value = Prev_filter_data[FILTER_MONTHLY_VAR_COL].mean()
            current_avg_cost = (sum(filter_data['Avg Cost Total'])/(filter_data['Weight'].sum() if filter_data['Weight'].sum() != 0 else 1))*.9
            prev_current_avg_cost = (sum(Prev_filter_data['Avg Cost Total'])/(Prev_filter_data['Weight'].sum() if Prev_filter_data['Weight'].sum() != 0 else 1))*.9
            MOM_Variance = ((current_avg_cost-prev_current_avg_cost)/prev_current_avg_cost)* 100
            var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
            MOM_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
            MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
            if MOM_Percent_Change == np.inf or pd.isna(MOM_Percent_Change) :
                MOM_Percent_Change = 0
            if MOM_QoQ_Percent_Change == np.inf or pd.isna(MOM_QoQ_Percent_Change):
                MOM_QoQ_Percent_Change = 0
            if pd.isna(MOM_Variance) or MOM_Variance == np.inf :
                MOM_Variance = 0
            return [MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change]
        elif FILTER_MONTHLY_VAR_COL == 'Max Buying Price':
            avg_value = _filter_[FILTER_MONTHLY_VAR_COL].mean()
            MOM_Variance = (sum((filter_data[FILTER_MONTHLY_VAR_COL] - avg_value)/ avg_value )/filter_data.shape[0]) * 100
            var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
            MOM_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
            MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
            if MOM_Percent_Change == np.inf or pd.isna(MOM_Percent_Change) :
                MOM_Percent_Change = 0
            if MOM_QoQ_Percent_Change == np.inf or pd.isna(MOM_QoQ_Percent_Change):
                MOM_QoQ_Percent_Change = 0
            if pd.isna(MOM_Variance) or MOM_Variance == np.inf :
                MOM_Variance = 0
            return [MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change]
        elif FILTER_MONTHLY_VAR_COL == 'Min Selling Price':
            avg_value = _filter_[FILTER_MONTHLY_VAR_COL].mean()
            MOM_Variance = (sum((filter_data[FILTER_MONTHLY_VAR_COL] - avg_value)/ avg_value )/filter_data.shape[0]) * 100
            var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
            MOM_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
            MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
            if MOM_Percent_Change == np.inf or pd.isna(MOM_Percent_Change) :
                MOM_Percent_Change = 0
            if MOM_QoQ_Percent_Change == np.inf or pd.isna(MOM_QoQ_Percent_Change):
                MOM_QoQ_Percent_Change = 0
            if pd.isna(MOM_Variance) or MOM_Variance == np.inf :
                MOM_Variance = 0
            return [MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change]
        else:
            return [0,0,0]
            
    except:
        return [0,0,0]
        
    
def get_gap_summary_table(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket):
    """
    Generate GAP summary table for all combinations of filter values
    """
    gap_summary = []
    
    # Get unique values for each filter
    months = [selected_month] if selected_month != "None" else list(master_df['Month'].unique())
    years = [selected_year] if selected_year != "None" else list(master_df['Year'].unique())
    shapes = [selected_shape] if selected_shape != "None" else list(master_df['Shape key'].unique())
    colors = [selected_color] if selected_color != "None" else list(master_df['Color Key'].unique())
    buckets = [selected_bucket] if selected_bucket != "None" else list(master_df['Buckets'].unique())
    
    # Generate all combinations
    for month in months:
        for year in years:
            for shape in shapes:
                for color in colors:
                    for bucket in buckets:
                        # Filter data for current combination
                        filtered_data = master_df[
                            (master_df['Month'] == month) & 
                            (master_df['Year'] == year) & 
                            (master_df['Shape key'] == shape) & 
                            (master_df['Color Key'] == color) & 
                            (master_df['Buckets'] == bucket)
                        ]
                        
                        if not filtered_data.empty:
                            max_qty = int(filtered_data['Max Qty'].max())
                            min_qty = int(filtered_data['Min Qty'].min())
                            stock_in_hand = filtered_data.shape[0]
                            gap_value = gap_analysis(max_qty, min_qty, stock_in_hand)
                            
                            gap_summary.append({
                                'Month': month,
                                'Year': year,
                                'Shape': shape,
                                'Color': color,
                                'Bucket': bucket,
                                'Max Qty': max_qty,
                                'Min Qty': min_qty,
                                'Stock in Hand': stock_in_hand,
                                'GAP Value': int(gap_value),
                                'Status': 'Excess' if gap_value > 0 else 'Need' if gap_value < 0 else 'Adequate'
                            })
    
    return pd.DataFrame(gap_summary).sort_values(by=['Shape','Color','Bucket'])

def get_final_data(file,PARENT_DF = 'kunmings.pkl'):
    df = poplutate_monthly_stock_sheet(file)
    parent_df = load_data(PARENT_DF)
    master_df = pd.concat([df, parent_df], ignore_index=True,axis=0)
    master_df = master_df.drop_duplicates(subset='Product Id')
    save_data(master_df)
    return master_df
def sort_months(months):
    """
    Sort months supporting both full names and abbreviations.
    
    Args:
        months: List of month names (full names or abbreviations)
    
    Returns:
        List of months sorted in chronological order
    """
    import calendar
    
    # Create mapping for both full names and abbreviations
    month_mapping = {}
    
    for i in range(1, 13):
        full_name = calendar.month_name[i]
        abbr_name = calendar.month_abbr[i]
        month_mapping[full_name] = i
        month_mapping[abbr_name] = i
        month_mapping[full_name.lower()] = i
        month_mapping[abbr_name.lower()] = i
    
    # Sort based on month order
    sorted_months = sorted(months, key=lambda month: month_mapping.get(month, 13))
    
    return sorted_months
def create_trend_visualization(master_df, selected_shape=None, selected_color=None, selected_bucket=None, selected_variance_column=None):
    """
    Create trend line visualizations for MOM Variance and MOM QoQ Percent Change
    
    Args:
        master_df: Master dataframe containing all data
        selected_shape: Selected shape filter
        selected_color: Selected color filter  
        selected_bucket: Selected bucket filter
        selected_variance_column: Column to calculate variance for
    
    Returns:
        plotly figure object
    """
    
    # Filter data based on selections
    if (selected_shape!=None and selected_color!=None and selected_bucket!=None) or (selected_shape!="None" and selected_color!="None" and selected_bucket!="None"):
        filtered_df = master_df[
            (master_df['Shape key'] == selected_shape) & 
            (master_df['Color Key'] == selected_color) & 
            (master_df['Buckets'] == selected_bucket)
        ]
    else:
        filtered_df = master_df
    
    if filtered_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Prepare variance column
    variance_col = selected_variance_column
    if variance_col == 'Current Average Cost':
        variance_col = 'Buying Price Avg'
    elif variance_col == 'None' or variance_col == None:
        variance_col = 'Max Buying Price' # Default column
    # monthly_variance
    # Calculate monthly variance data
    try:
        var_analysis = monthly_variance(filtered_df, variance_col)
        
        # Create date column for proper sorting
        var_analysis['Date']='01'+'-'+var_analysis['Num_Month'].astype(str)+'-'+var_analysis['Year'].astype(str)
        var_analysis['Date'] = pd.to_datetime(var_analysis['Date'], format='%d-%m-%Y')
        var_analysis = var_analysis.sort_values('Date')
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Variance Trend', 'Quarter-over-Quarter Change'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Add Monthly Variance line
        fig.add_trace(
            go.Scatter(
                x=var_analysis['Date'],
                y=var_analysis['Monthly_change'],
                mode='lines+markers',
                name='Monthly Change %',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, color='#1f77b4'),
                hovertemplate='<b>%{x|%b %Y}</b><br>' +
                             'Monthly Change: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add QoQ Change line
        fig.add_trace(
            go.Scatter(
                x=var_analysis['Date'],
                y=var_analysis['qaurter_change'],
                mode='lines+markers',
                name='QoQ Change %',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8, color='#ff7f0e'),
                hovertemplate='<b>%{x|%b %Y}</b><br>' +
                             'QoQ Change: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Trend Analysis - {selected_shape} | {selected_color} | {selected_bucket}",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update subplot title colors to orange
        fig.update_annotations(font=dict(color='black', size=16))
        
        # Update x-axis
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Monthly Change (%)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="QoQ Change (%)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        return fig
        
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

def create_summary_charts(master_df, selected_shape, selected_color, selected_bucket):
    """
    Create summary charts showing overall trends across all months/years
    
    Args:
        master_df: Master dataframe
        selected_shape: Selected shape filter
        selected_color: Selected color filter
        selected_bucket: Selected bucket filter
    
    Returns:
        plotly figure object
    """
    
    # Filter data
    if (selected_shape!=None and selected_color!=None and selected_bucket!=None) or (selected_shape!="None" and selected_color!="None" and selected_bucket!="None"):
        filtered_df = master_df[
            (master_df['Shape key'] == selected_shape) & 
            (master_df['Color Key'] == selected_color) & 
            (master_df['Buckets'] == selected_bucket)
        ]
    else:
        filtered_df = master_df
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Group by month and year to get summary statistics
    summary_data = filtered_df.groupby(['Month', 'Year']).agg({
        'Avg Cost Total': 'mean',
        'Max Buying Price': 'mean',
        'Weight': 'sum',
        'Product Id': 'count'
    }).reset_index()
    
    
    # Create date column for proper sorting
    summary_data['Num_Month'] = summary_data['Month'].map(month_map)
    summary_data['Date']='01'+'-'+summary_data['Num_Month'].astype(str)+'-'+summary_data['Year'].astype(str)
    summary_data['Date'] = pd.to_datetime(summary_data['Date'], format='%d-%m-%Y')
    summary_data = summary_data.sort_values('Date')
    
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Cost Trend', 'Max Buying Price Trend', 
                       'Total Weight', 'Product Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Average Cost Trend
    fig.add_trace(
        go.Scatter(
            x=summary_data['Date'],
            y=summary_data['Avg Cost Total'],
            mode='lines+markers',
            name='Avg Cost',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Max Buying Price Trend
    fig.add_trace(
        go.Scatter(
            x=summary_data['Date'],
            y=summary_data['Max Buying Price'],
            mode='lines+markers',
            name='Max Buying Price',
            line=dict(color='#A23B72', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    # Total Weight
    fig.add_trace(
        go.Scatter(
            x=summary_data['Date'],
            y=summary_data['Weight'],
            mode='lines+markers',
            name='Total Weight',
            line=dict(color='#F18F01', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    # Product Count
    fig.add_trace(
        go.Scatter(
            x=summary_data['Date'],
            y=summary_data['Product Id'],
            mode='lines+markers',
            name='Product Count',
            line=dict(color='#C73E1D', width=2),
            marker=dict(size=6)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Summary Analytics - {selected_shape} | {selected_color} | {selected_bucket}",
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update subplot title colors to orange
    fig.update_annotations(font=dict(color='black', size=16))
    
    # Update all x-axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
    
    return fig

def main():
    st.set_page_config(page_title="Yellow Diamond Dashboard", layout="wide")
    st.title("Yellow Diamond Dashboard")
    st.markdown("Upload Excel files to process multiple sheets and filter data.")
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'master_df' not in st.session_state:
        st.session_state.master_df = pd.DataFrame()
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = load_upload_history()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'awaiting_clarification' not in st.session_state:
        st.session_state.awaiting_clarification = False
    if 'original_query' not in st.session_state:
        st.session_state.original_query = ""
        
    # Load LLM model
    with st.spinner("Loading AI Model..."):
        tokenizer, model = load_llm_model()
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with multiple sheets"
    )
    
    # Display upload history
    display_upload_history()
    
    # Main content area
    if uploaded_file is not None and not st.session_state.data_processed:
        with st.spinner("Processing Excel file..."):
            try:
                # Get file size
                file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else None
                
                # Process the file
                st.subheader("🗄️ Master Database")
                st.session_state.master_df = get_final_data(uploaded_file)
                st.session_state.data_processed = True
                
                # Add to upload history after successful processing
                st.session_state.upload_history = add_to_upload_history(
                    filename=uploaded_file.name,
                    file_size=file_size
                )
                
                # Show success message
                st.success(f"✅ Successfully processed: {uploaded_file.name}")
                
                # Force sidebar refresh to show updated history
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Still add to history but mark as failed
                history = load_upload_history()
                new_entry = {
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_size": uploaded_file.size if hasattr(uploaded_file, 'size') else None,
                    "status": "Failed"
                }
                history.insert(0, new_entry)
                history = history[:10]
                save_upload_history(history)
                st.session_state.upload_history = history
    
    # Try to load existing data if available
    if st.session_state.master_df.empty:
        try:
            st.session_state.master_df = load_data('kunmings.pkl')
            st.session_state.data_processed = True
        except:
            pass
    
    if not st.session_state.master_df.empty:
        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["📊 Manual Filtering & Analysis", "🤖 AI Query Assistant"])
        
        with tab1:
            # Existing filtering functionality
            Month,Year,Shape,Color,Bucket,Variance_Column = st.columns(6)
            with Month:
                categories = ["None"]+sort_months(list(st.session_state.master_df['Month'].unique()))
                selected_month = st.selectbox("Filter by Month", categories)
            with Year:
                years = ["None"]+sorted(list(st.session_state.master_df['Year'].unique()))
                selected_year = st.selectbox("Filter by Year", years)
            with Shape:
                shapes = ["None"]+list(st.session_state.master_df['Shape key'].unique())
                selected_shape = st.selectbox("Filter by Shape", shapes)
            with Color:
                colors = ["None"]+['WXYZ','FLY','FY','FIY','FVY']
                selected_color = st.selectbox("Filter by Color", colors)
            with Bucket:
                buckets = ["None"]+list(stock_bucket.keys())
                selected_bucket = st.selectbox("Filter by Bucket", buckets)
            with Variance_Column:
                variance_columns = ["None"]+['Current Average Cost','Max Buying Price','Min Selling Price']
                selected_variance_column = st.selectbox("Select Variance Column", variance_columns)
            
            # Apply filters and show results (existing code)
            filtered_df = st.session_state.master_df.copy()
            if ((selected_month != "None") & (selected_year != "None") & (selected_shape != "None") & (selected_color != "None") & (selected_bucket != "None")) :
                filter_data,max_buying_price,current_avg_cost,gap_output,min_selling_price = get_filtered_data(selected_month,\
                                                                                                                            selected_year,\
                                                                                                                            selected_shape,\
                                                                                                                            selected_color,\
                                                                                                                            selected_bucket)
                MOM_Variance,MOM_Percent_Change,MOM_QoQ_Percent_Change = get_summary_metrics(filter_data,selected_month,selected_shape,selected_year,\
                                                                                            selected_color,\
                                                                                            selected_bucket,\
                                                                                            selected_variance_column)
                # Display summary metrics
                st.subheader("📊 Summary Metrics")
                mbp,cac,mom_var,mom_perc,qoq_perc,GAP,msp = st.columns(7)
                if type(max_buying_price)!= str:
                    with GAP:
                        st.metric("Gap Analysis",value=gap_output,help=f"{'Excess' if gap_output>0 else 'Need' if gap_output < 0 else 'Enough'}")
                    with mbp:
                        st.metric("Max Buying Price", f"${max_buying_price:,.2f}")
                    with msp:
                        st.metric("Min Selling Price",f"${min_selling_price:,.2f}")
                    with cac:
                        st.metric("Current Avg Cost", f"${current_avg_cost:,.2f}",help="90% of Sum of (Average Cost Total) / Weight ")
                    with mom_var:
                        st.metric("MOM Variance ", f"{MOM_Variance:,.2f}%")
                    with mom_perc:
                        st.metric("MOM Percent Change", f"{MOM_Percent_Change:.2f}%")
                    with qoq_perc:
                        st.metric("MOM QoQ Percent Change", f"{MOM_QoQ_Percent_Change:.2f}%")
                    
                    
                else:
                    with GAP:
                        st.metric("Gap Analysis",value=gap_output,help=f"{'Excess' if gap_output>0 else 'Need' if gap_output < 0 else 'Enough'}")
                    with mbp:
                        st.metric("Max Buying Price", f"0")
                    with cac:
                        st.metric("Current Avg Cost", f"0")
                    with mom_var:
                        st.metric("MOM Variance ", f"0")
                    with mom_perc:
                        st.metric("MOM Percent Change", f"0")
                    with qoq_perc:
                        st.metric("MOM QoQ Percent Change", f"0")
                        
                    st.subheader("No Data Present for This Filter")
                # Add visualization section
                st.subheader("📈 Trend Analysis")
                
                # Create tabs for different visualizations
                st.markdown("""
                <style>
                    /* Style all tab labels */
                    .stTabs [data-baseweb="tab-list"] {
                        gap: 24px;
                    }
                    
                    .stTabs [data-baseweb="tab-list"] button {
                        height: 50px;
                        padding-left: 20px;
                        padding-right: 20px;
                    }
                    
                    /* Inactive tabs - VIOLET */
                    .stTabs [data-baseweb="tab-list"] button p {
                        color: #8B00FF;  /* Violet for inactive tabs */
                        font-size: 18px;
                    }
                    
                    /* Active tab - RED */
                    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
                        color: #FF0000;  /* Red for active tab */
                        font-weight: bold;
                    }
                    
                    /* Hover effect */
                    .stTabs [data-baseweb="tab-list"] button:hover p {
                        color: #FF0000;
                        transition: color 0.3s;
                    }
                    
                    /* Tab underline/highlight - RED */
                    .stTabs [data-baseweb="tab-highlight"] {
                        background-color: #FF0000;
                        height: 3px;
                    }
                    
                    /* Tab panels background (optional) */
                    .stTabs [data-baseweb="tab-panel"] {
                        padding-top: 20px;
                    }
                </style>
                """, unsafe_allow_html=True)
                viz_tab1, viz_tab2 = st.tabs(["📊 Variance Trends", "📈 Summary Analytics"])
                with viz_tab1:
                    if selected_variance_column != "None":
                        trend_fig = create_trend_visualization(
                            st.session_state.master_df, 
                            selected_shape, 
                            selected_color, 
                            selected_bucket, 
                            selected_variance_column
                        )
                        st.plotly_chart(trend_fig, use_container_width=True)
                    else:
                        st.info("Please select a variance column to view trend analysis.")
                
                with viz_tab2:
                    summary_fig = create_summary_charts(
                        st.session_state.master_df, 
                        selected_shape, 
                        selected_color, 
                        selected_bucket
                    )
                    st.plotly_chart(summary_fig, use_container_width=True)
                
                st.subheader("📊 Data Table")
                st.dataframe(
                    filter_data,
                    use_container_width=True,
                    hide_index=True
                        )
                # Download processed data
                st.subheader("💾 Download Filtered Data")
                # filter_data['Avg Cost Total'] = filter_data['avg']
                csv = filter_data.loc[:,['Product Id','Shape key','Color Key','Avg Cost Total','Min Qty','Max Qty','Buying Price Avg','Max Buying Price']].to_csv(index=False)
                st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
                )
                st.subheader("💾 Download Master Data")
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                label="Download Master Data as CSV",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
                )
            
            # GAP Summary Table - Show for all combinations
            st.subheader("📋 GAP Summary")
            gap_summary_df = get_gap_summary_table(
                st.session_state.master_df, 
                selected_month, 
                selected_year, 
                selected_shape, 
                selected_color, 
                selected_bucket
            )
            
            if not gap_summary_df.empty:
                # Apply styling to highlight negative GAP values in red
                def highlight_negative_gap(row):
                    if row['GAP Value'] < 0:
                        return ['background-color: #ffebee; color: #c62828'] * len(row)
                    else:
                        return [''] * len(row)
                def highlight_shape_gap(row):
                    if row['GAP Value'] < 0:
                        return ['background-color: #ffebee; color: #c62828'] * len(row)
                    else:
                        if row['Shape']=='Cushion':
                            return ['background-color: #baffc9; color: #c62828'] * len(row)
                        elif row['Shape']=='Oval':
                            return ['background-color: #bae1ff; color: #c62828'] * len(row)
                        elif row['Shape']=='Pear':
                            return ['background-color: #ffb3ba; color: #c62828'] * len(row)
                        elif row['Shape']=='Radiant':
                            return ['background-color: #ffdfba; color: #c62828'] * len(row)
                        elif row['Shape']=='Other':
                            return ['background-color: #ffffba; color: #c62828'] * len(row)
                        else:
                            return [''] * len(row)
                styled_df = gap_summary_df.style.apply(highlight_shape_gap, axis=1)
                # styled_df = gap_summary_df.style.apply(highlight_negative_gap, axis=1)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download GAP Summary
                st.subheader("💾 Download GAP Summary")
                gap_summary_df_cols = ['Month','Year','Shape','Color','Bucket','GAP Value']
                gap_csv = gap_summary_df.loc[:,gap_summary_df_cols].to_csv(index=False)
                gap_csv_excess = gap_summary_df[gap_summary_df['Status']=='Excess'].loc[:,gap_summary_df_cols].to_csv(index=False)
                gap_csv_need = gap_summary_df[gap_summary_df['Status']=='Need'].loc[:,gap_summary_df_cols].to_csv(index=False)
                st.download_button(
                    label="Download GAP Summary as CSV",
                    data=gap_csv,
                    file_name=f"gap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download GAP Excess Summary as CSV",
                    data=gap_csv_excess,
                    file_name=f"gap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download GAP Need Summary as CSV",
                    data=gap_csv_need,
                    file_name=f"gap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data available for GAP analysis with current filters.")
                
            if not ((selected_month != "None") & (selected_year != "None") & (selected_shape != "None") & (selected_color != "None") & (selected_bucket != "None")):
                st.info("Please select all filter values except 'Select Variance Column' to view detailed metrics.")
        
        with tab2:
            st.subheader("🤖 AI Query Assistant")
            st.markdown("Ask questions about your data in natural language!")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and isinstance(message["content"], pd.DataFrame):
                        st.dataframe(message["content"])
                    else:
                        st.write(message["content"])
            
            # Query input
            user_query = st.chat_input("Ask about your data (e.g., 'Show me all Cushion diamonds from January 2024')")
            
            if user_query:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                with st.chat_message("user"):
                    st.write(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Check if this is a clarification response
                        if st.session_state.awaiting_clarification:
                            # Combine original query with clarification
                            full_query = f"{st.session_state.original_query}. {user_query}"
                            st.session_state.awaiting_clarification = False
                        else:
                            full_query = user_query
                            
                        # Check if query is vague
                        is_vague, vague_response = is_query_vague(full_query, tokenizer, model)
                        
                        if is_vague and not st.session_state.awaiting_clarification:
                            # Ask for clarification
                            clarification = ask_for_clarification(full_query, tokenizer, model)
                            st.write(clarification)
                            st.session_state.chat_history.append({"role": "assistant", "content": clarification})
                            st.session_state.awaiting_clarification = True
                            st.session_state.original_query = user_query
                        else:
                            # Process the query
                            code = process_data_query(full_query, st.session_state.master_df, tokenizer, model)
                            
                            # Execute the code
                            result, error = execute_llm_code(code, st.session_state.master_df)
                            
                            if error:
                                st.error(f"Error executing analysis: {error}")
                                st.code(code, language='python')
                                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {error}"})
                            else:
                                # Generate summary
                                summary = generate_analysis_summary(full_query, result, tokenizer, model)
                                st.write(summary)
                                
                                # Display results
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result)
                                    
                                    # Offer download
                                    csv = result.to_csv(index=False)
                                    st.download_button(
                                        label="Download Results as CSV",
                                        data=csv,
                                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                elif result is not None:
                                    st.write(result)
                                
                                # Add to chat history
                                st.session_state.chat_history.append({"role": "assistant", "content": summary})
                                if isinstance(result, pd.DataFrame):
                                    st.session_state.chat_history.append({"role": "assistant", "content": result})
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.awaiting_clarification = False
                st.session_state.original_query = ""
                st.rerun()
                
    else:
        st.info("No data in master database. Upload an Excel file to get started!")
    
    # Reset button
    if st.sidebar.button("Reset Data Processing"):
        st.session_state.data_processed = False
        st.session_state.master_df = pd.DataFrame()
        st.session_state.chat_history = []
        st.session_state.awaiting_clarification = False
        st.session_state.original_query = ""
        st.rerun()
    
    # Clear history button
    if st.sidebar.button("Clear Upload History"):
        save_upload_history([])
        st.session_state.upload_history = []
        st.success("Upload history cleared!")
        st.rerun()
    
if __name__ == "__main__":
    main()
