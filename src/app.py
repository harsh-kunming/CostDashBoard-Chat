import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
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
from transformers import (
    TapasTokenizer, 
    TapasForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import warnings
warnings.filterwarnings('ignore')

# File history management functions
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

# Initialize models
@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    try:
        # Load TAPAS for table QA
        tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")
        tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")
        
        # Load Flan-T5 for intent detection and conversation
        flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        return {
            'tapas': {'tokenizer': tapas_tokenizer, 'model': tapas_model},
            'flan': {'tokenizer': flan_tokenizer, 'model': flan_model}
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# LLM Query Processing Functions
def check_query_clarity(query: str, models: dict) -> Tuple[bool, str]:
    """
    Use Flan-T5 to check if query is clear or needs clarification
    Returns: (is_clear, clarification_request)
    """
    flan_tokenizer = models['flan']['tokenizer']
    flan_model = models['flan']['model']
    
    prompt = f"""Analyze if this query about diamond inventory data is clear and specific enough to answer:
    Query: "{query}"
    
    If the query is clear and specific, respond with "CLEAR".
    If the query is vague or needs more information, respond with "CLARIFY: [specific question to ask user]"
    
    Examples:
    - "Show me diamonds" -> "CLARIFY: What specific attributes of diamonds would you like to see? (e.g., shape, color, price range, weight)"
    - "What is the average price of cushion cut diamonds in bucket A?" -> "CLEAR"
    - "Analysis" -> "CLARIFY: What type of analysis would you like? (e.g., price trends, inventory gaps, color distribution)"
    
    Response:"""
    
    inputs = flan_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = flan_model.generate(**inputs, max_length=100, temperature=0.7)
    
    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.strip().upper().startswith("CLEAR"):
        return True, ""
    elif response.strip().upper().startswith("CLARIFY:"):
        return False, response.replace("CLARIFY:", "").strip()
    else:
        # Fallback if model doesn't follow format
        return True, ""

def extract_query_intent(query: str, models: dict) -> dict:
    """
    Use Flan-T5 to extract intent and parameters from query
    """
    flan_tokenizer = models['flan']['tokenizer']
    flan_model = models['flan']['model']
    
    prompt = f"""Extract the intent and filters from this diamond inventory query:
    Query: "{query}"
    
    Identify:
    1. Action type: (filter/aggregate/analyze/compare)
    2. Filters: shape, color, bucket, month, year
    3. Metrics: price, weight, quantity, gap
    4. Analysis type: trend, distribution, summary
    
    Respond in this format:
    ACTION: [action]
    FILTERS: shape=[value], color=[value], bucket=[value], month=[value], year=[value]
    METRICS: [comma-separated metrics]
    ANALYSIS: [type]
    
    Example response:
    ACTION: filter
    FILTERS: shape=Cushion, color=FY, bucket=A
    METRICS: price, weight
    ANALYSIS: none
    """
    
    inputs = flan_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = flan_model.generate(**inputs, max_length=150, temperature=0.3)
    
    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse response into structured format
    intent = {
        'action': 'filter',
        'filters': {},
        'metrics': [],
        'analysis': 'none'
    }
    
    lines = response.strip().split('\n')
    for line in lines:
        if line.startswith('ACTION:'):
            intent['action'] = line.replace('ACTION:', '').strip().lower()
        elif line.startswith('FILTERS:'):
            filter_str = line.replace('FILTERS:', '').strip()
            if filter_str and filter_str != 'none':
                for f in filter_str.split(','):
                    if '=' in f:
                        key, value = f.strip().split('=')
                        if value and value.lower() != 'none':
                            intent['filters'][key.strip()] = value.strip()
        elif line.startswith('METRICS:'):
            metrics_str = line.replace('METRICS:', '').strip()
            if metrics_str and metrics_str != 'none':
                intent['metrics'] = [m.strip() for m in metrics_str.split(',')]
        elif line.startswith('ANALYSIS:'):
            intent['analysis'] = line.replace('ANALYSIS:', '').strip().lower()
    
    return intent

def process_with_tapas(query: str, df: pd.DataFrame, models: dict) -> str:
    """
    Process table question answering using TAPAS
    """
    tapas_tokenizer = models['tapas']['tokenizer']
    tapas_model = models['tapas']['model']
    
    # Prepare a subset of data for TAPAS (it has token limits)
    # Select relevant columns based on query
    relevant_cols = ['Product Id', 'Shape key', 'Color Key', 'Weight', 
                     'Average\nCost\n(USD)', 'Max Buying Price', 'Min Selling Price',
                     'Buckets', 'Month', 'Year', 'Max Qty', 'Min Qty']
    
    # Filter columns that exist in the dataframe
    available_cols = [col for col in relevant_cols if col in df.columns]
    subset_df = df[available_cols].head(100)  # TAPAS has input size limitations
    
    # Convert to format TAPAS expects
    table = subset_df.astype(str).values.tolist()
    
    # Tokenize
    inputs = tapas_tokenizer(
        table=table,
        queries=[query],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = tapas_model(**inputs)
    
    # Process outputs
    predicted_answer_coordinates = outputs.logits.argmax(dim=-1)
    
    # Extract answer from table
    answers = []
    for coordinates in predicted_answer_coordinates[0]:
        if coordinates < len(table) * len(table[0]):
            row_idx = coordinates // len(table[0])
            col_idx = coordinates % len(table[0])
            if row_idx < len(table) and col_idx < len(table[0]):
                answers.append(table[row_idx][col_idx])
    
    return ", ".join(answers) if answers else "No specific answer found in the data."

def execute_data_operation(df: pd.DataFrame, intent: dict) -> pd.DataFrame:
    """
    Execute data operations based on extracted intent
    """
    result_df = df.copy()
    
    # Apply filters
    for key, value in intent['filters'].items():
        if key == 'shape' and 'Shape key' in result_df.columns:
            result_df = result_df[result_df['Shape key'] == value]
        elif key == 'color' and 'Color Key' in result_df.columns:
            result_df = result_df[result_df['Color Key'] == value]
        elif key == 'bucket' and 'Buckets' in result_df.columns:
            result_df = result_df[result_df['Buckets'] == value]
        elif key == 'month' and 'Month' in result_df.columns:
            result_df = result_df[result_df['Month'] == value]
        elif key == 'year' and 'Year' in result_df.columns:
            try:
                year_val = int(value)
                result_df = result_df[result_df['Year'] == year_val]
            except:
                pass
    
    return result_df

def generate_natural_response(query: str, intent: dict, result_df: pd.DataFrame, models: dict) -> str:
    """
    Generate a natural language response based on the query results
    """
    flan_tokenizer = models['flan']['tokenizer']
    flan_model = models['flan']['model']
    
    # Create summary statistics
    summary = {
        'row_count': len(result_df),
        'avg_price': result_df['Max Buying Price'].mean() if 'Max Buying Price' in result_df.columns else 0,
        'total_weight': result_df['Weight'].sum() if 'Weight' in result_df.columns else 0,
        'unique_shapes': result_df['Shape key'].nunique() if 'Shape key' in result_df.columns else 0,
        'unique_colors': result_df['Color Key'].nunique() if 'Color Key' in result_df.columns else 0,
    }
    
    prompt = f"""Generate a helpful response for this diamond inventory query:
    Query: "{query}"
    
    Results summary:
    - Total items found: {summary['row_count']}
    - Average price: ${summary['avg_price']:.2f}
    - Total weight: {summary['total_weight']:.2f}
    - Unique shapes: {summary['unique_shapes']}
    - Unique colors: {summary['unique_colors']}
    
    Applied filters: {intent['filters']}
    
    Provide a concise, informative response that directly answers the user's question.
    """
    
    inputs = flan_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = flan_model.generate(**inputs, max_length=200, temperature=0.7)
    
    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add data summary
    if summary['row_count'] > 0:
        response += f"\n\nSummary of filtered data:\n"
        response += f"- Total diamonds: {summary['row_count']}\n"
        response += f"- Average buying price: ${summary['avg_price']:,.2f}\n"
        response += f"- Total weight: {summary['total_weight']:.2f} carats\n"
        
        if intent['action'] == 'analyze' and 'gap' in str(intent.get('analysis', '')):
            # Calculate gap analysis
            max_qty = result_df['Max Qty'].sum() if 'Max Qty' in result_df.columns else 0
            min_qty = result_df['Min Qty'].sum() if 'Min Qty' in result_df.columns else 0
            current_stock = len(result_df)
            
            if current_stock > max_qty:
                response += f"- Gap Analysis: EXCESS inventory ({current_stock - max_qty} units over maximum)\n"
            elif current_stock < min_qty:
                response += f"- Gap Analysis: SHORTAGE ({min_qty - current_stock} units below minimum)\n"
            else:
                response += f"- Gap Analysis: ADEQUATE inventory levels\n"
    
    return response

# Data processing functions
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
    """
    # Input validation
    if not isinstance(current_quarter_price, (int, float)) or not isinstance(previous_quarter_price, (int, float)):
        raise TypeError("Both prices must be numeric values")
    
    if previous_quarter_price <= 0:
        variance_percentage = 0.00001
    
    # Calculate variance percentage
    if previous_quarter_price !=0:
        variance_percentage = ((current_quarter_price - previous_quarter_price) / previous_quarter_price) * 100
    else:
        variance_percentage = ((current_quarter_price - previous_quarter_price) / (previous_quarter_price+current_quarter_price)) * 100
    return round(variance_percentage, 2)

def calculate_qoq_variance_series(price_data):
    """
    Calculate quarter-on-quarter variance for a series of quarterly prices.
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
    """
    master_df = load_data('kunmings.pkl')
    if (type(FILTE_YEAR)==str) & (str(FILTE_YEAR).isnumeric()):
        FILTE_YEAR = int(FILTE_YEAR)
    
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
    if 'models' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.models = load_models()
    
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
    
    # New AI Query Interface Section
    if not st.session_state.master_df.empty and st.session_state.models:
        st.markdown("---")
        st.subheader("🤖 AI Query Assistant")
        st.markdown("Ask questions about your diamond inventory in natural language.")
        
        # Create two columns for chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Query input
            user_query = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., 'Show me all cushion cut diamonds in bucket A' or 'What's the average price of FY colored diamonds?'",
                key="user_query_input"
            )
            
            # Process query button
            if st.button("Submit Query", type="primary"):
                if user_query:
                    # Add user query to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    
                    with st.spinner("Processing your query..."):
                        try:
                            # Load data if needed
                            if st.session_state.master_df.empty:
                                st.session_state.master_df = load_data('kunmings.pkl')
                            
                            # Check query clarity
                            is_clear, clarification = check_query_clarity(user_query, st.session_state.models)
                            
                            if not is_clear:
                                # Ask for clarification
                                response = f"I need more information to help you better. {clarification}"
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                            else:
                                # Extract intent
                                intent = extract_query_intent(user_query, st.session_state.models)
                                
                                # Execute data operation
                                result_df = execute_data_operation(st.session_state.master_df, intent)
                                
                                # Generate response
                                if intent['action'] in ['filter', 'aggregate', 'analyze']:
                                    response = generate_natural_response(user_query, intent, result_df, st.session_state.models)
                                else:
                                    # Use TAPAS for specific questions
                                    tapas_answer = process_with_tapas(user_query, result_df, st.session_state.models)
                                    response = f"Based on the data: {tapas_answer}"
                                
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                                
                                # Store results for display
                                if len(result_df) > 0:
                                    st.session_state.last_query_results = result_df
                                
                        except Exception as e:
                            error_response = f"I encountered an error processing your query: {str(e)}. Please try rephrasing your question."
                            st.session_state.chat_history.append({"role": "assistant", "content": error_response})
        
        with col2:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                if 'last_query_results' in st.session_state:
                    del st.session_state.last_query_results
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
            
            # Display query results if available
            if 'last_query_results' in st.session_state and len(st.session_state.last_query_results) > 0:
                st.markdown("### Query Results")
                
                # Display first 20 rows
                display_df = st.session_state.last_query_results.head(20)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                if len(st.session_state.last_query_results) > 20:
                    st.info(f"Showing first 20 rows of {len(st.session_state.last_query_results)} total results.")
                
                # Download button for query results
                csv = st.session_state.last_query_results.to_csv(index=False)
                st.download_button(
                    label="Download Query Results",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Original filter interface continues below...
    if not st.session_state.master_df.empty or uploaded_file is not None:
        st.markdown("---")
        st.subheader("📊 Manual Filters")
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
        
        # Rest of the original code continues unchanged...
        # Apply filters
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
            tab1, tab2 = st.tabs(["📊 Variance Trends", "📈 Summary Analytics"])
            with tab1:
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
            
            with tab2:
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
                file_name=f"gap_summary_excess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Download GAP Need Summary as CSV",
                data=gap_csv_need,
                file_name=f"gap_summary_need_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available for GAP analysis with current filters.")
            
        if not ((selected_month != "None") & (selected_year != "None") & (selected_shape != "None") & (selected_color != "None") & (selected_bucket != "None")):
            st.info("Please select all filter values except 'Select Variance Column' to view detailed metrics.")
        
    else:
        st.info("No data in master database. Upload an Excel file to get started!")
        
        # Still show AI Query Assistant if models are loaded and we have the pickle file
        try:
            if st.session_state.models and os.path.exists('kunmings.pkl'):
                st.session_state.master_df = load_data('kunmings.pkl')
                if not st.session_state.master_df.empty:
                    st.markdown("---")
                    st.subheader("🤖 AI Query Assistant")
                    st.info("You can still query the existing data using natural language!")
        except:
            pass
    
    # Reset button
    if st.sidebar.button("Reset Data Processing"):
        st.session_state.data_processed = False
        st.session_state.master_df = pd.DataFrame()
        st.rerun()
    
    # Clear history button
    if st.sidebar.button("Clear Upload History"):
        save_upload_history([])
        st.session_state.upload_history = []
        st.success("Upload history cleared!")
        st.rerun()
    
if __name__ == "__main__":
    main()
