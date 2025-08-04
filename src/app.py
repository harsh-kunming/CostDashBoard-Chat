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

# Add this new section to the main() function after the existing UI elements

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
    
    # File history management functions (keep all original functions at the top of the file)
    
if __name__ == "__main__":
    main()
