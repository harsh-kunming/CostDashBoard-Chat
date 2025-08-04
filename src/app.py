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
import requests
import time
import pickle
import torch

# Import Hugging Face libraries
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM, 
        AutoModelForCausalLM,
        pipeline,
        T5ForConditionalGeneration,
        T5Tokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers library not installed. Some features will be limited.")

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Model loading strategies
MODEL_LOADING_STRATEGIES = {
    "API (Online)": "api",
    "Local Model": "local",
    "Pipeline": "pipeline",
    "Inference Client": "client"
}

# Initialize session state for model caching
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}
if 'current_model_strategy' not in st.session_state:
    st.session_state.current_model_strategy = "api"

# Initialize Hugging Face setup
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")

# Also try Streamlit secrets
if not HF_TOKEN:
    try:
        if hasattr(st, 'secrets') and 'HUGGINGFACE_TOKEN' in st.secrets:
            HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    except:
        pass

# Model configurations
MODEL_CONFIGS = {
    "google/flan-t5-small": {
        "type": "seq2seq",
        "size": "440MB",
        "description": "Smallest, fastest, good for basic tasks"
    },
    "google/flan-t5-base": {
        "type": "seq2seq", 
        "size": "990MB",
        "description": "Balanced performance and speed"
    },
    "microsoft/DialoGPT-small": {
        "type": "causal",
        "size": "351MB", 
        "description": "Conversational AI model"
    },
    "distilgpt2": {
        "type": "causal",
        "size": "353MB",
        "description": "Lightweight GPT-2"
    }
}

# Cache for loaded models
@st.cache_resource
def load_model_local(model_name: str, model_type: str):
    """Load model locally (downloaded to cache)"""
    try:
        with st.spinner(f"Loading {model_name} locally... This may take a few minutes the first time."):
            if model_type == "seq2seq":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:  # causal
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available():
                model = model.to(device)
                
            return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model locally: {str(e)}")
        return None, None, None

@st.cache_resource
def load_pipeline(model_name: str, task: str = "text2text-generation"):
    """Load model using pipeline (simpler interface)"""
    try:
        with st.spinner(f"Loading {model_name} pipeline..."):
            if "t5" in model_name or "flan" in model_name:
                task = "text2text-generation"
            elif "gpt" in model_name:
                task = "text-generation"
                
            pipe = pipeline(
                task,
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return pipe
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None

def load_inference_client(model_name: str, token: str = None):
    """Load model using Inference Client"""
    try:
        client = InferenceClient(model=model_name, token=token)
        return client
    except Exception as e:
        st.error(f"Error loading inference client: {str(e)}")
        return None

def generate_with_local_model(prompt: str, tokenizer, model, device, model_type: str):
    """Generate text using locally loaded model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if model_type == "seq2seq":
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95
                )
            else:  # causal
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For causal models, remove the input prompt from response
        if model_type == "causal" and prompt in response:
            response = response.replace(prompt, "").strip()
            
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_with_pipeline(prompt: str, pipe):
    """Generate text using pipeline"""
    try:
        result = pipe(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '').replace(prompt, '').strip()
        return str(result)
    except Exception as e:
        return f"Error with pipeline: {str(e)}"

def generate_with_inference_client(prompt: str, client):
    """Generate text using Inference Client"""
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        return response
    except Exception as e:
        return f"Error with inference client: {str(e)}"

def query_llm_multimethod(prompt: str, strategy: str = "api", model_name: str = None, token: str = None):
    """Query LLM using multiple strategies"""
    
    # Get model name and token from session state if not provided
    if not model_name:
        model_name = st.session_state.get('model_name', 'google/flan-t5-small')
    if not token:
        token = st.session_state.get('hf_token', '')
    
    # Check if in test mode
    if st.session_state.get('test_mode', False):
        return generate_fallback_response(
            prompt, 
            st.session_state.get('master_df', pd.DataFrame())
        )
    
    # Strategy: API (Original method)
    if strategy == "api":
        return query_llm_api(prompt, model_name, token)
    
    # Strategy: Local Model
    elif strategy == "local" and TRANSFORMERS_AVAILABLE:
        # Check if model is already loaded
        cache_key = f"local_{model_name}"
        if cache_key in st.session_state.loaded_models:
            tokenizer, model, device = st.session_state.loaded_models[cache_key]
        else:
            model_config = MODEL_CONFIGS.get(model_name, {"type": "seq2seq"})
            tokenizer, model, device = load_model_local(model_name, model_config["type"])
            if tokenizer and model:
                st.session_state.loaded_models[cache_key] = (tokenizer, model, device)
            else:
                return "Failed to load local model. Try API method or Test Mode."
        
        if tokenizer and model:
            model_type = MODEL_CONFIGS.get(model_name, {"type": "seq2seq"})["type"]
            return generate_with_local_model(prompt, tokenizer, model, device, model_type)
    
    # Strategy: Pipeline
    elif strategy == "pipeline" and TRANSFORMERS_AVAILABLE:
        cache_key = f"pipeline_{model_name}"
        if cache_key in st.session_state.loaded_models:
            pipe = st.session_state.loaded_models[cache_key]
        else:
            pipe = load_pipeline(model_name)
            if pipe:
                st.session_state.loaded_models[cache_key] = pipe
            else:
                return "Failed to load pipeline. Try API method or Test Mode."
        
        if pipe:
            return generate_with_pipeline(prompt, pipe)
    
    # Strategy: Inference Client
    elif strategy == "client" and HF_HUB_AVAILABLE:
        client = load_inference_client(model_name, token)
        if client:
            return generate_with_inference_client(prompt, client)
        else:
            return "Failed to load inference client. Try API method or Test Mode."
    
    # Fallback to API
    else:
        return query_llm_api(prompt, model_name, token)

def query_llm_api(prompt: str, model_name: str, token: str):
    """Original API-based query method"""
    current_headers = {}
    if token and token.strip():
        current_headers = {"Authorization": f"Bearer {token.strip()}"}
    
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(api_url, headers=current_headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            return str(result)
        else:
            return f"API Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Replace the original query_llm function
def query_llm(prompt: str, max_retries: int = 3) -> str:
    """Main query function that uses the selected strategy"""
    strategy = st.session_state.get('current_model_strategy', 'api')
    model_name = st.session_state.get('model_name', 'google/flan-t5-small')
    token = st.session_state.get('hf_token', '')
    
    for attempt in range(max_retries):
        try:
            response = query_llm_multimethod(prompt, strategy, model_name, token)
            
            # If response indicates a loading model, retry
            if "loading" in response.lower() and attempt < max_retries - 1:
                time.sleep(20)
                continue
                
            return response
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error after {max_retries} attempts: {str(e)}"
    
    return "Failed to get response. Try a different loading strategy or enable Test Mode."

# File history management functions (keep existing)
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

# LLM Query Processing Functions
def get_data_schema(df: pd.DataFrame) -> str:
    """Get a concise schema of the dataframe for LLM context"""
    schema = f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
    schema += "Columns:\n"
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        # Sample unique values for categorical columns
        if unique_count < 20 and dtype == 'object':
            unique_vals = df[col].dropna().unique()[:10]
            schema += f"- {col} ({dtype}): {unique_count} unique values, {null_count} nulls. Sample values: {list(unique_vals)}\n"
        else:
            schema += f"- {col} ({dtype}): {unique_count} unique values, {null_count} nulls\n"
    
    return schema

def get_data_sample(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Get a sample of the data for LLM context"""
    sample = df.head(n_rows).to_string()
    return f"Sample data (first {n_rows} rows):\n{sample}"

def generate_fallback_response(query: str, df: pd.DataFrame) -> str:
    """Generate a simple response when LLM is not available"""
    query_lower = query.lower()
    
    # Check for common query patterns
    if "how many" in query_lower or "count" in query_lower:
        return f"The dataset contains {len(df)} total records. You can use the filters in the Dashboard tab to explore specific subsets of data."
    
    elif "average" in query_lower or "mean" in query_lower:
        if "weight" in query_lower:
            avg_weight = df['Weight'].mean()
            return f"The average weight across all diamonds is {avg_weight:.2f}."
        elif "cost" in query_lower or "price" in query_lower:
            avg_cost = df['Avg Cost Total'].mean()
            return f"The average cost total is ${avg_cost:.2f}."
        else:
            return "I can help you calculate averages. Please specify which column you'd like to analyze (e.g., weight, cost, price)."
    
    elif "filter" in query_lower or "show" in query_lower:
        return """To filter data, I can help you with:
        - Shape: Cushion, Oval, Pear, Radiant, Other
        - Color: WXYZ, FLY, FY, FIY, FVY
        - Buckets: B1, B2, B3, B4, B5
        
        Please specify your filter criteria, or use the Dashboard tab for interactive filtering."""
    
    elif "gap" in query_lower:
        return "GAP analysis shows the difference between stock in hand and min/max quantity thresholds. Check the GAP Summary section in the Dashboard tab for detailed analysis."
    
    elif "trend" in query_lower:
        return "For trend analysis, please use the Dashboard tab and select the Variance Trends visualization. You can analyze trends for Max Buying Price, Current Average Cost, or Min Selling Price."
    
    else:
        return """I'm here to help you analyze your diamond inventory data. You can ask me to:
        - Filter data by shape, color, or bucket
        - Calculate averages, sums, or counts
        - Analyze trends and patterns
        - Perform GAP analysis
        
        For the best experience, please configure the model loading strategy in the AI Assistant Settings."""

def create_llm_prompt(query: str, data_context: str, conversation_history: List[Dict] = None) -> str:
    """Create a prompt for the LLM with data context"""
    # Flan-T5 works better with direct, instructional prompts
    prompt = f"""You are analyzing a diamond inventory dataset with the following structure:

{data_context}

Important columns:
- Shape key: Diamond shapes (Cushion, Oval, Pear, Radiant, Other)
- Color Key: Colors (WXYZ, FLY, FY, FIY, FVY)
- Buckets: Weight categories (B1, B2, B3, B4, B5)
- Weight: Diamond weight in carats
- Avg Cost Total: Average cost
- Max/Min Qty: Quantity thresholds
- Max Buying Price, Min Selling Price: Price boundaries

User Question: {query}

Provide a clear, specific answer. If the question is vague, ask for clarification."""
    
    return prompt

def parse_llm_response(response: str, df: pd.DataFrame) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict]]:
    """Parse LLM response and extract any data operations"""
    
    # Check if response is an error message
    if "Error" in response or "Failed" in response:
        return response, None, {}
    
    filtered_df = None
    analysis_results = {}
    
    # Simple keyword-based filtering as fallback
    response_lower = response.lower()
    
    # Check for filter intent
    if any(word in response_lower for word in ["filter", "show", "find", "get", "display"]):
        try:
            # Extract filter criteria from response
            filter_conditions = []
            
            # Check for shape mentions
            for shape in ['cushion', 'oval', 'pear', 'radiant', 'other']:
                if shape in response_lower:
                    filter_conditions.append(f"(df['Shape key'] == '{shape.capitalize()}')")
            
            # Check for color mentions
            for color in ['wxyz', 'fly', 'fy', 'fiy', 'fvy']:
                if color in response_lower:
                    filter_conditions.append(f"(df['Color Key'] == '{color.upper()}')")
            
            # Check for bucket mentions
            for bucket in ['b1', 'b2', 'b3', 'b4', 'b5']:
                if bucket in response_lower:
                    filter_conditions.append(f"(df['Buckets'] == '{bucket.upper()}')")
            
            # Apply filters if any were found
            if filter_conditions:
                filter_expr = " & ".join(filter_conditions)
                filtered_df = eval(f"df[{filter_expr}]")
                
        except Exception as e:
            # If filtering fails, continue without filtered data
            pass
    
    # Check for analysis intent
    if any(word in response_lower for word in ["average", "mean", "sum", "total", "count", "trend", "analysis"]):
        try:
            # Determine operation type
            if "average" in response_lower or "mean" in response_lower:
                operation = "mean"
            elif "sum" in response_lower or "total" in response_lower:
                operation = "sum"
            elif "count" in response_lower:
                operation = "count"
            else:
                operation = "mean"  # default
            
            # Determine column
            if "weight" in response_lower:
                column = "Weight"
            elif "price" in response_lower or "cost" in response_lower:
                if "buying" in response_lower:
                    column = "Max Buying Price"
                elif "selling" in response_lower:
                    column = "Min Selling Price"
                else:
                    column = "Avg Cost Total"
            else:
                column = "Weight"  # default
            
            # Determine groupby
            groupby = None
            if "by shape" in response_lower or "for each shape" in response_lower:
                groupby = "Shape key"
            elif "by color" in response_lower or "for each color" in response_lower:
                groupby = "Color Key"
            elif "by bucket" in response_lower or "for each bucket" in response_lower:
                groupby = "Buckets"
            
            analysis_results = {
                "operation": operation,
                "column": column
            }
            if groupby:
                analysis_results["groupby"] = groupby
                
        except Exception as e:
            # If analysis parsing fails, continue
            pass
    
    return response, filtered_df, analysis_results

def execute_data_operation(df: pd.DataFrame, operation: Dict) -> pd.DataFrame:
    """Execute a data operation based on LLM instructions"""
    try:
        if "groupby" in operation:
            grouped = df.groupby(operation["groupby"])
            if operation["operation"] == "mean":
                return grouped[operation["column"]].mean().reset_index()
            elif operation["operation"] == "sum":
                return grouped[operation["column"]].sum().reset_index()
            elif operation["operation"] == "count":
                return grouped[operation["column"]].count().reset_index()
        else:
            if operation["operation"] == "mean":
                return pd.DataFrame({operation["column"]: [df[operation["column"]].mean()]})
            elif operation["operation"] == "sum":
                return pd.DataFrame({operation["column"]: [df[operation["column"]].sum()]})
            elif operation["operation"] == "count":
                return pd.DataFrame({operation["column"]: [df[operation["column"]].count()]})
    except:
        return pd.DataFrame()

# Keep all existing data processing functions (load_data, save_data, etc.)
# ... [Include all the existing functions from the original code here] ...

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

# Include all other existing functions here...
# [Copy all the remaining functions from the original code]

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
    if 'query_results' not in st.session_state:
        st.session_state.query_results = []
    if 'test_mode' not in st.session_state:
        st.session_state.test_mode = False
    if 'allow_public' not in st.session_state:
        st.session_state.allow_public = True
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "google/flan-t5-small"
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = HF_TOKEN
    if 'current_model_strategy' not in st.session_state:
        st.session_state.current_model_strategy = "api"
        
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
    
    # Check if we have data loaded
    if st.session_state.master_df.empty:
        # Try to load from pickle file
        try:
            st.session_state.master_df = load_data('kunmings.pkl')
            st.session_state.data_processed = True
        except:
            pass
    
    if not st.session_state.master_df.empty:
        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["📊 Dashboard", "💬 AI Assistant"])
        
        with tab1:
            # Original dashboard functionality
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
                    
                # Add visualization section
                st.subheader("📈 Trend Analysis")
                
                # Create tabs for different visualizations
                tab1_1, tab1_2 = st.tabs(["📊 Variance Trends", "📈 Summary Analytics"])
                
                with tab1_1:
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
                
                with tab1_2:
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
        
        with tab2:
            # AI Assistant Tab
            st.subheader("🤖 AI Assistant for Data Analysis")
            st.markdown("Ask questions about your diamond inventory data in natural language!")
            
            # Show status based on configuration
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.session_state.get('test_mode', False):
                    st.info("🧪 Running in Test Mode - Limited functionality without API")
                elif st.session_state.current_model_strategy == "local":
                    st.success(f"✅ Using Local Model: {st.session_state.model_name}")
                elif st.session_state.current_model_strategy == "pipeline":
                    st.success(f"✅ Using Pipeline: {st.session_state.model_name}")
                elif st.session_state.current_model_strategy == "api":
                    if st.session_state.hf_token:
                        st.success(f"✅ API Ready with token")
                    else:
                        st.warning("⚠️ Using public API (rate limits apply)")
            
            with col2:
                # Quick strategy selector
                strategy = st.selectbox(
                    "Model Loading Strategy",
                    options=list(MODEL_LOADING_STRATEGIES.keys()),
                    index=list(MODEL_LOADING_STRATEGIES.values()).index(st.session_state.current_model_strategy),
                    help="Choose how to load and run the model"
                )
                new_strategy = MODEL_LOADING_STRATEGIES[strategy]
                if new_strategy != st.session_state.current_model_strategy:
                    st.session_state.current_model_strategy = new_strategy
                    st.info(f"Switched to {strategy}")
            
            # Show library status
            if not TRANSFORMERS_AVAILABLE and st.session_state.current_model_strategy in ["local", "pipeline"]:
                st.error("Transformers library not installed. Run: pip install transformers torch")
                st.info("Falling back to API mode")
                st.session_state.current_model_strategy = "api"
            
            # Example queries
            with st.expander("💡 Example Queries"):
                st.markdown("""
                **Try these example queries:**
                - Show me all cushion diamonds with color FLY
                - What's the average weight of diamonds in bucket B1?
                - Filter data for January 2024
                - Which shape has the highest average cost?
                - Show diamonds with negative gap values
                - Calculate the total value of oval diamonds
                - What are the trends for buying prices?
                """)
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "data" in message:
                        st.dataframe(message["data"], use_container_width=True)
            
            # User input
            user_query = st.chat_input("Ask a question about your data...")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_query)
                
                # Process query with LLM
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Prepare data context
                        data_schema = get_data_schema(st.session_state.master_df)
                        data_sample = get_data_sample(st.session_state.master_df)
                        data_context = f"{data_schema}\n\n{data_sample}"
                        
                        # Create prompt
                        prompt = create_llm_prompt(
                            user_query, 
                            data_context, 
                            st.session_state.chat_history[:-1]  # Exclude the last user message
                        )
                        
                        # Query LLM
                        llm_response = query_llm(prompt)
                        
                        # Parse response
                        response_text, filtered_data, analysis_results = parse_llm_response(
                            llm_response, 
                            st.session_state.master_df
                        )
                        
                        # Display response
                        st.write(response_text)
                        
                        # If data was filtered, display it
                        if filtered_data is not None and not filtered_data.empty:
                            st.dataframe(filtered_data, use_container_width=True)
                            
                            # Add download button for filtered data
                            csv = filtered_data.to_csv(index=False)
                            st.download_button(
                                label="Download Query Results",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Add to chat history with data
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response_text,
                                "data": filtered_data
                            })
                        else:
                            # Add to chat history without data
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response_text
                            })
                        
                        # Execute any analysis operations
                        if analysis_results:
                            try:
                                analysis_df = execute_data_operation(
                                    filtered_data if filtered_data is not None else st.session_state.master_df,
                                    analysis_results
                                )
                                if not analysis_df.empty:
                                    st.subheader("Analysis Results:")
                                    st.dataframe(analysis_df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error executing analysis: {str(e)}")
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
                
    else:
        st.info("No data in master database. Upload an Excel file to get started!")
        
    # AI Assistant Settings in Sidebar
    with st.sidebar.expander("⚙️ AI Assistant Settings", expanded=False):
        st.markdown("### Model Configuration")
        
        # Token input
        token_input = st.text_input(
            "Hugging Face Token",
            value=st.session_state.hf_token,
            type="password",
            help="Optional for API access"
        )
        
        if token_input != st.session_state.hf_token:
            st.session_state.hf_token = token_input
            st.success("Token updated!")
        
        # Model selection
        st.markdown("### Model Selection")
        
        model_type = st.radio(
            "Model Type",
            ["Small (Fast)", "Base (Balanced)", "Conversational"],
            index=0
        )
        
        model_map = {
            "Small (Fast)": "google/flan-t5-small",
            "Base (Balanced)": "google/flan-t5-base",
            "Conversational": "microsoft/DialoGPT-small"
        }
        
        new_model = model_map[model_type]
        if new_model != st.session_state.model_name:
            st.session_state.model_name = new_model
            st.session_state.loaded_models = {}  # Clear cache
            st.info(f"Model changed to {model_type}")
        
        st.markdown("### Loading Strategy")
        
        strategy_info = {
            "api": "Uses Hugging Face API (no download)",
            "local": "Downloads model locally (faster after first load)",
            "pipeline": "Simple interface with auto optimization",
            "client": "Advanced API with more control"
        }
        
        current_strategy = st.session_state.current_model_strategy
        st.info(strategy_info.get(current_strategy, "Unknown strategy"))
        
        # Test mode
        test_mode = st.checkbox(
            "Enable Test Mode",
            value=st.session_state.test_mode,
            help="Use basic pattern matching (no model required)"
        )
        
        if test_mode != st.session_state.test_mode:
            st.session_state.test_mode = test_mode
        
        # System requirements
        with st.expander("System Requirements"):
            st.markdown("""
            **API Mode:**
            - Internet connection
            - Optional: HF token
            
            **Local/Pipeline Mode:**
            - 2-4GB free RAM
            - Python packages: transformers, torch
            - First load downloads model (~500MB-1GB)
            
            **GPU Support:**
            - CUDA-capable GPU (optional)
            - Speeds up local inference
            """)
        
        # Model info
        if st.session_state.model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[st.session_state.model_name]
            st.markdown(f"""
            **Current Model:** {st.session_state.model_name}
            - Type: {config['type']}
            - Size: {config['size']}
            - Description: {config['description']}
            """)
    
    # Reset buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Data Processing"):
        st.session_state.data_processed = False
        st.session_state.master_df = pd.DataFrame()
        st.session_state.chat_history = []
        st.rerun()
    
    if st.sidebar.button("Clear Upload History"):
        save_upload_history([])
        st.session_state.upload_history = []
        st.success("Upload history cleared!")
        st.rerun()
    
    # Show current resource usage
    if st.session_state.current_model_strategy in ["local", "pipeline"] and st.session_state.loaded_models:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Resource Usage")
        st.sidebar.info(f"Models loaded: {len(st.session_state.loaded_models)}")
        if torch.cuda.is_available():
            st.sidebar.success("GPU: Available ✅")
        else:
            st.sidebar.info("GPU: Not available (using CPU)")
    
if __name__ == "__main__":
    main()
