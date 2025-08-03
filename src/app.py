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
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize the chatbot model
@st.cache_resource
def load_chatbot_model():
    """Load the Hugging Face model for chatbot functionality"""
    try:
        # Using Flan-T5 base model - lightweight and good for analytical tasks
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create a text generation pipeline
        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            device=-1  # Use CPU, set to 0 for GPU
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading chatbot model: {e}")
        return None

# Chatbot query processing functions
def extract_entities_from_query(query: str) -> Dict[str, str]:
    """Extract relevant entities from the user query"""
    query_lower = query.lower()
    
    entities = {
        'shape': None,
        'color': None,
        'bucket': None,
        'month': None,
        'year': None,
        'metric': None,
        'analysis_type': None
    }
    
    # Shape patterns
    shape_patterns = {
        'cushion': ['cushion', 'cushion shaped', 'cushion-shaped'],
        'oval': ['oval', 'oval shaped', 'oval-shaped'],
        'pear': ['pear', 'pear shaped', 'pear-shaped'],
        'radiant': ['radiant', 'radiant shaped', 'radiant-shaped'],
        'other': ['other', 'heart', 'emerald', 'marquise']
    }
    
    for shape, patterns in shape_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                entities['shape'] = shape.capitalize()
                break
    
    # Color patterns
    color_patterns = {
        'WXYZ': ['wxyz', 'w-x-y-z'],
        'FLY': ['fly'],
        'FY': ['fy color', 'fy'],
        'FIY': ['fiy'],
        'FVY': ['fvy']
    }
    
    for color, patterns in color_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                entities['color'] = color
                break
    
    # Bucket patterns
    bucket_ranges = stock_bucket.keys()
    for bucket in bucket_ranges:
        if bucket.lower() in query_lower:
            entities['bucket'] = bucket
    
    # Month patterns
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    
    for month in months:
        if month in query_lower:
            entities['month'] = month.capitalize()
            break
    
    # Year patterns
    year_match = re.search(r'20\d{2}', query_lower)
    if year_match:
        entities['year'] = int(year_match.group())
    
    # Metric patterns
    metric_patterns = {
        'buying price': ['buying price', 'max buying price', 'maximum buying price'],
        'selling price': ['selling price', 'min selling price', 'minimum selling price'],
        'average cost': ['average cost', 'avg cost', 'current average cost'],
        'variance': ['variance', 'variation', 'change'],
        'mom': ['month over month', 'mom', 'monthly'],
        'qoq': ['quarter over quarter', 'qoq', 'quarterly'],
        'gap': ['gap', 'gap analysis', 'stock gap'],
        'trend': ['trend', 'trending', 'trends'],
        'summary': ['summary', 'overview', 'summarize']
    }
    
    for metric, patterns in metric_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                entities['metric'] = metric
                break
    
    # Analysis type
    if any(word in query_lower for word in ['variance', 'change', 'mom', 'month over month']):
        entities['analysis_type'] = 'variance'
    elif any(word in query_lower for word in ['gap', 'stock gap', 'gap analysis']):
        entities['analysis_type'] = 'gap'
    elif any(word in query_lower for word in ['trend', 'trending']):
        entities['analysis_type'] = 'trend'
    elif any(word in query_lower for word in ['summary', 'overview']):
        entities['analysis_type'] = 'summary'
    else:
        entities['analysis_type'] = 'general'
    
    return entities

def process_chatbot_query(query: str, master_df: pd.DataFrame) -> str:
    """Process the user query and generate a response based on the data"""
    if master_df.empty:
        return "No data available. Please upload and process an Excel file first."
    
    # Extract entities from query
    entities = extract_entities_from_query(query)
    
    try:
        # Filter data based on extracted entities
        filtered_df = master_df.copy()
        
        if entities['shape']:
            filtered_df = filtered_df[filtered_df['Shape key'] == entities['shape']]
        if entities['color']:
            filtered_df = filtered_df[filtered_df['Color Key'] == entities['color']]
        if entities['bucket']:
            filtered_df = filtered_df[filtered_df['Buckets'] == entities['bucket']]
        if entities['month']:
            filtered_df = filtered_df[filtered_df['Month'] == entities['month']]
        if entities['year']:
            filtered_df = filtered_df[filtered_df['Year'] == entities['year']]
        
        if filtered_df.empty:
            return "No data found matching your query criteria. Please check your filters."
        
        # Generate response based on analysis type
        response = ""
        
        if entities['analysis_type'] == 'variance' and entities['metric']:
            # Calculate variance analysis
            if entities['metric'] == 'buying price':
                col = 'Max Buying Price'
            elif entities['metric'] == 'selling price':
                col = 'Min Selling Price'
            elif entities['metric'] == 'average cost':
                col = 'Buying Price Avg'
            else:
                col = 'Max Buying Price'  # default
            
            # Get variance analysis
            var_analysis = monthly_variance(filtered_df, col)
            
            # Find the most recent month's data
            if entities['month'] and entities['year']:
                month_data = var_analysis[
                    (var_analysis['Month'] == entities['month']) & 
                    (var_analysis['Year'] == entities['year'])
                ]
                if not month_data.empty:
                    mom_change = month_data['Monthly_change'].values[0]
                    qoq_change = month_data['qaurter_change'].values[0]
                    
                    response = f"For {entities['shape'] or 'all shapes'} diamonds"
                    if entities['color']:
                        response += f" with {entities['color']} color"
                    response += f":\n\n"
                    response += f"Month-over-Month variance in {entities['metric']}: {mom_change:.2f}%\n"
                    response += f"Quarter-over-Quarter variance: {qoq_change:.2f}%\n"
                else:
                    response = "No variance data available for the specified period."
            else:
                # Provide overall trend
                latest_data = var_analysis.iloc[-1]
                response = f"Latest {entities['metric']} variance for {entities['shape'] or 'all shapes'}:\n"
                response += f"Month-over-Month: {latest_data['Monthly_change']:.2f}%\n"
                response += f"Quarter-over-Quarter: {latest_data['qaurter_change']:.2f}%"
        
        elif entities['analysis_type'] == 'gap':
            # Calculate gap analysis
            max_qty = filtered_df['Max Qty'].max()
            min_qty = filtered_df['Min Qty'].min()
            stock_in_hand = filtered_df.shape[0]
            gap_value = gap_analysis(max_qty, min_qty, stock_in_hand)
            
            response = f"Gap Analysis for {entities['shape'] or 'all shapes'}"
            if entities['color']:
                response += f" with {entities['color']} color"
            response += f":\n\n"
            response += f"Stock in Hand: {stock_in_hand}\n"
            response += f"Maximum Quantity: {int(max_qty)}\n"
            response += f"Minimum Quantity: {int(min_qty)}\n"
            response += f"Gap: {int(gap_value)} ("
            response += "Excess" if gap_value > 0 else "Need" if gap_value < 0 else "Adequate"
            response += ")"
        
        elif entities['analysis_type'] == 'summary':
            # Provide summary statistics
            total_items = filtered_df.shape[0]
            avg_weight = filtered_df['Weight'].mean()
            total_value = filtered_df['Avg Cost Total'].sum()
            avg_buying_price = filtered_df['Max Buying Price'].mean()
            
            response = f"Summary for {entities['shape'] or 'all'} diamonds"
            if entities['color']:
                response += f" with {entities['color']} color"
            response += f":\n\n"
            response += f"Total Items: {total_items}\n"
            response += f"Average Weight: {avg_weight:.2f}\n"
            response += f"Total Value: ${total_value:,.2f}\n"
            response += f"Average Buying Price: ${avg_buying_price:,.2f}"
        
        else:
            # General query - provide basic statistics
            total_items = filtered_df.shape[0]
            unique_shapes = filtered_df['Shape key'].nunique()
            unique_colors = filtered_df['Color Key'].nunique()
            
            response = "Based on your query:\n\n"
            response += f"Total matching items: {total_items}\n"
            response += f"Unique shapes: {unique_shapes}\n"
            response += f"Unique colors: {unique_colors}\n"
            response += f"\nYou can ask more specific questions about:\n"
            response += "- Month-over-month variance in buying/selling prices\n"
            response += "- Gap analysis for specific diamond types\n"
            response += "- Trends and summaries for different periods"
        
        return response
        
    except Exception as e:
        return f"Error processing query: {str(e)}. Please try rephrasing your question."

def generate_ai_response(query: str, data_response: str, qa_pipeline) -> str:
    """Generate a more natural response using the LLM"""
    if qa_pipeline is None:
        return data_response
    
    # Create a prompt for the model
    prompt = f"""Based on the following data analysis, provide a clear and concise answer to the user's question.

User Question: {query}

Data Analysis Results:
{data_response}

Provide a natural, conversational response that directly answers the user's question:"""
    
    try:
        # Generate response using the model
        result = qa_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7)
        ai_response = result[0]['generated_text']
        
        # Combine AI response with data
        final_response = f"{ai_response}\n\n📊 **Detailed Data:**\n{data_response}"
        return final_response
    except:
        # Fallback to data response if model fails
        return data_response

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
    if 'chatbot_model' not in st.session_state:
        st.session_state.chatbot_model = None
        
    # Load chatbot model
    if st.session_state.chatbot_model is None:
        with st.spinner("Loading AI chatbot..."):
            st.session_state.chatbot_model = load_chatbot_model()
    
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
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Assistant", "📈 Analytics"])
    
    with tab1:
        # Main dashboard content
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
                    
        if not st.session_state.master_df.empty or uploaded_file is not None:
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
                
                # Data table and downloads section
                st.subheader("📊 Data Table")
                st.dataframe(
                    filter_data,
                    use_container_width=True,
                    hide_index=True
                        )
                # Download processed data
                st.subheader("💾 Download Filtered Data")
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
            
        else:
            st.info("No data in master database. Upload an Excel file to get started!")
    
    with tab2:
        # AI Assistant Tab
        st.subheader("🤖 AI Diamond Analytics Assistant")
        st.markdown("Ask questions about your diamond inventory data in natural language!")
        
        # Example queries
        with st.expander("📝 Example Queries", expanded=False):
            st.markdown("""
            - What is the month-over-month variance in buying price for Cushion shaped diamonds with FY color?
            - Show me the gap analysis for Oval diamonds in bucket C1
            - What's the trend for selling prices of Pear shaped diamonds?
            - Give me a summary of all WXYZ colored diamonds
            - What is the current average cost for Radiant diamonds in January 2024?
            - Show me the quarter-over-quarter change for FLY colored diamonds
            """)
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Chat input
        user_query = st.chat_input("Ask about your diamond data...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display user message
            st.chat_message("user").write(user_query)
            
            # Generate response
            with st.spinner("Analyzing your query..."):
                # Process the query
                data_response = process_chatbot_query(user_query, st.session_state.master_df)
                
                # Generate AI-enhanced response if model is available
                if st.session_state.chatbot_model:
                    final_response = generate_ai_response(user_query, data_response, st.session_state.chatbot_model)
                else:
                    final_response = data_response
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                
                # Display assistant response
                st.chat_message("assistant").write(final_response)
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with tab3:
        # Analytics Tab
        if not st.session_state.master_df.empty:
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
            
            # Add filter controls for analytics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                shapes_analytics = ["None"] + list(st.session_state.master_df['Shape key'].unique())
                selected_shape_analytics = st.selectbox("Filter by Shape", shapes_analytics, key="shape_analytics")
            with col2:
                colors_analytics = ["None"] + ['WXYZ','FLY','FY','FIY','FVY']
                selected_color_analytics = st.selectbox("Filter by Color", colors_analytics, key="color_analytics")
            with col3:
                buckets_analytics = ["None"] + list(stock_bucket.keys())
                selected_bucket_analytics = st.selectbox("Filter by Bucket", buckets_analytics, key="bucket_analytics")
            with col4:
                variance_columns_analytics = ["None"] + ['Current Average Cost','Max Buying Price','Min Selling Price']
                selected_variance_column_analytics = st.selectbox("Select Variance Column", variance_columns_analytics, key="variance_analytics")
            
            with viz_tab1:
                if selected_variance_column_analytics != "None":
                    trend_fig = create_trend_visualization(
                        st.session_state.master_df, 
                        selected_shape_analytics if selected_shape_analytics != "None" else None, 
                        selected_color_analytics if selected_color_analytics != "None" else None, 
                        selected_bucket_analytics if selected_bucket_analytics != "None" else None, 
                        selected_variance_column_analytics
                    )
                    st.plotly_chart(trend_fig, use_container_width=True)
                else:
                    st.info("Please select a variance column to view trend analysis.")
            
            with viz_tab2:
                summary_fig = create_summary_charts(
                    st.session_state.master_df, 
                    selected_shape_analytics if selected_shape_analytics != "None" else None, 
                    selected_color_analytics if selected_color_analytics != "None" else None, 
                    selected_bucket_analytics if selected_bucket_analytics != "None" else None
                )
                st.plotly_chart(summary_fig, use_container_width=True)
        else:
            st.info("No data available. Please upload and process an Excel file first.")
    
    # Reset button in sidebar
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
