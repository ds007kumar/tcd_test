import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import duckdb
from io import BytesIO
import os
from dotenv import load_dotenv
import warnings
import hashlib
import logging  # NEW: Added logging for debugging

import requests
from urllib3.exceptions import InsecureRequestWarning

# import certifi
# import ssl
# import urllib.request

# ssl_context=ssl.create_default_context(certifi.where())
# urllib.request.urlopen("https://www.google.com",context=ssl_context)

# Disable only the InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

response = requests.get(
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json",
    verify=False  # 👈 disables SSL certificate verification
)

print(response.status_code)
print(response.text[:200])  # Just to preview the respons

# NEW: Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import custom modules
from src.data_processor import ServiceNowProcessor
from src.ml_model import TimeSeriesPredictor
from src.gemini_client import GeminiAnalyzer
from src.dashboard_DD import DashboardGenerator
from src.alerts import AlertManager
from src.categorization import Categorization

# --- Configuration & Setup ---
load_dotenv()
st.set_page_config(
    page_title="Incident Prediction & TCD Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NEW: Mock user credentials with hashed passwords (SHA-256)
# Purpose: Store two mock email-password pairs securely
MOCK_USERS = {
    "mala1@akzonobel.com": hashlib.sha256("password123".encode('utf-8')).hexdigest(),
    "wajahat2@akzonobel.com": hashlib.sha256("password456".encode('utf-8')).hexdigest()
}


def inject_custom_css(theme="light"):
    if theme == "light":
        bg_gradient = "linear-gradient(to top, #e2e8f0 0%, #f8fafc 60%, #ffffff 100%)"
        sidebar_gradient = "linear-gradient(to bottom, #cbd5e1 0%, #f1f5f9 100%)"
        text_color = "#111827"
        subheader_color = "#2563eb"
        button_bg = "#3b82f6"
        button_hover = "#1d4ed8"
        card_bg = "rgba(255, 255, 255, 0.95)"
        shadow = "0 12px 20px rgba(0, 0, 0, 0.1)"
        metric_bg = "#f1f5f9"
    else:
        bg_gradient = "linear-gradient(to top, #1e293b 0%, #334155 60%, #475569 100%)"
        sidebar_gradient = "linear-gradient(to bottom, #0f172a 0%, #1e293b 100%)"
        text_color = "#f9fafb"
        subheader_color = "#93c5fd"
        button_bg = "#2563eb"
        button_hover = "#1d4ed8"
        card_bg = "rgba(30, 41, 59, 0.85)"
        shadow = "0 12px 24px rgba(0, 0, 0, 0.3)"
        metric_bg = "#1e293b"

    st.markdown(f"""
        <style>
            .stApp {{
                background: {bg_gradient};
                color: {text_color};
                font-family: 'Segoe UI', 'Inter', sans-serif;
                transition: background 0.6s ease, color 0.6s ease;
                min-height: 100vh;
                position: relative;
            }}
            [data-testid="stHeader"] {{
                background: transparent !important;
                box-shadow: none !important;
            }}
            [data-testid="stDeployButton"] {{
                background: {bg_gradient} !important;
                color: {text_color} !important;
                border: none !important;
            }}
            [data-testid="stSidebar"] {{
                background: {sidebar_gradient};
                color: {text_color};
                padding: 16px;
                border-radius: 0 12px 12px 0;
                box-shadow: inset -4px 0 8px rgba(0,0,0,0.05);
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {subheader_color};
                font-weight: 600;
                letter-spacing: 0.5px;
                background: transparent !important;
            }}
            p, label, span {{
                color: {text_color};
                font-size: 0.95rem;
                line-height: 1.5;
            }}
            button {{
                background-color: {button_bg} !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 10px 20px !important;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            }}
            button:hover {{
                background-color: {button_hover} !important;
                transform: scale(1.03);
                box-shadow: 0 6px 12px rgba(0,0,0,0.25);
            }}
            .stMetric {{
                background: {metric_bg};
                border-radius: 12px;
                padding: 20px;
                box-shadow: {shadow};
                transition: all 0.3s ease;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }}
            .stMetric:hover {{
                transform: scale(1.05);
                box-shadow: 0 12px 24px rgba(0,0,0,0.2);
                cursor: pointer;
            }}
            .st-emotion-cache-1cpxqw2, .st-emotion-cache-16txtl3 {{
                background: {card_bg};
                border-radius: 16px;
                padding: 16px;
                box-shadow: {shadow};
                transition: all 0.4s ease;
            }}
            .st-emotion-cache-1cpxqw2:hover {{
                transform: scale(1.015);
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.1);
            }}
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
                color: {text_color} !important;
            }}
            .css-1y4p8pa, .css-1r6slb0, .element-container:has(.stPlotlyChart) {{
                background: white;
                border-radius: 16px;
                padding: 20px;
                box-shadow: {shadow};
                transition: all 0.4s ease;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }}
            .css-1y4p8pa:hover, .css-1r6slb0:hover, .element-container:has(.stPlotlyChart):hover {{
                transform: scale(1.015);
                box-shadow: 0 16px 32px rgba(0, 0, 0, 0.15);
                cursor: pointer;
            }}
            .logo-container {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 1;
            }}
            .logo-container img {{
                width: 200px;
                height: 200px;
                object-fit: contain;
            }}
            .login-container {{
                max-width: 300px;
                width: 100%;
                padding: 20px;
                background: {card_bg};
                border-radius: 16px;
                box-shadow: {shadow};
                text-align: center;
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
            }}
            .login-container input[type="text"], .login-container input[type="password"] {{
                width: 70%; /* Reduced from 100% */
                max-width: 200px; /* Added to cap the width */
                padding: 10px;
                margin: 10px auto; /* Center the inputs */
                border-radius: 8px;
                border: 1px solid rgba(0, 0, 0, 0.1);
                background: {metric_bg};
                color: {text_color};
                display: block; /* Ensure centering works */
            }}
            .login-container button {{
                width: 70%; /* Reduced to match inputs */
                max-width: 200px; /* Cap the button width */
                margin: 10px auto; /* Center the button */
                display: block; /* Ensure centering works */
            }}
        </style>
    """, unsafe_allow_html=True)

def show_login_page():
    # Apply CSS immediately for login page
    theme = st.session_state.get('theme', 'light')  # Use theme from session state or default to light
    inject_custom_css(theme)
    
    # Create logo container
    with st.container():
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        try:
            st.image("logo.jpg", width=200, use_container_width=False)
        except FileNotFoundError:
            st.warning("Logo file 'logo.jpg' not found. Please add your logo file to the project directory.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create a centered login container
    # st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.header("🔒 Login to Incident Prediction & TCD Agent")
    
    # Input fields for email and password
    email = st.text_input("Email", placeholder="Enter your email")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    
    # Login button
    if st.button("Login", type="primary"):
        # Hash the input password for comparison
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        
        # Check credentials
        if email in MOCK_USERS and MOCK_USERS[email] == hashed_password:
            st.session_state.is_authenticated = True
            st.success("✅ Login successful! Redirecting...")
            st.rerun()  # Refresh to show main app
        else:
            st.error("🚨 Invalid email or password. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data(show_spinner="Loading data...")
def load_data(file_bytes: BytesIO, file_name: str) -> pd.DataFrame:
    if file_name.endswith(".csv"):
        return duckdb.query("SELECT * FROM read_csv_auto(?)", params=[file_bytes.read()]).to_df()
    elif file_name.endswith(".xlsx"):
        return pd.read_excel(file_bytes)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return pd.DataFrame()

# MODIFIED: Replaced compute_file_hash with compute_dataframe_hash to hash DataFrame content
# Purpose: Ensures identical data in CSV or Excel produces the same hash, fixing the issue where Excel and CSV files with the same data were treated as different.
def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of DataFrame content."""
    # Convert DataFrame to a consistent string representation (sorted by columns and rows)
    df_str = df.sort_values(by=df.columns.tolist()).to_csv(index=False, encoding='utf-8')
    return hashlib.sha256(df_str.encode('utf-8')).hexdigest()

def main():
    # MODIFIED: Moved theme selection inside main to apply only after login
    # Purpose: Ensure theme is applied consistently after authentication
    theme = st.session_state.get('theme', 'light')  # Use theme from session state
    inject_custom_css(theme)
    
    st.title("🔍 Incident Prediction & TCD Reporting Agent")
    st.markdown("---")
    
    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = pd.DataFrame()
        st.session_state.categorized_data = pd.DataFrame()
        st.session_state.last_uploaded_file = None
        # MODIFIED: Changed last_file_hash to last_df_hash to reflect DataFrame-based hashing
        st.session_state.last_df_hash = None
        st.session_state.categorization_complete = False
        st.session_state.show_unlock_button = False
        st.session_state.data_processor = ServiceNowProcessor()
        st.session_state.predictor = TimeSeriesPredictor()
        # st.session_state.gemini_analyzer = GeminiAnalyzer()
        st.session_state.dashboard = DashboardGenerator()
        st.session_state.alert_manager = AlertManager()
        st.session_state.categorizer = Categorization()

    # Define folder for saving categorized data
    categorized_folder = "uploaded_categorized"
    os.makedirs(categorized_folder, exist_ok=True)

    # Centralized Data Upload
    st.sidebar.subheader("📂 Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload Incident CSV or Excel File", type=['csv', 'xlsx'])
    
    if uploaded_file:
        # MODIFIED: Removed file_bytes hashing and moved to DataFrame-based hashing
        # Purpose: Compute hash after loading DataFrame to ensure same data in CSV/Excel is recognized
        with st.spinner(f"Loading {uploaded_file.name}..."):
            st.session_state.raw_data = load_data(BytesIO(uploaded_file.getvalue()), uploaded_file.name)
            # NEW: Compute DataFrame hash
            df_hash = compute_dataframe_hash(st.session_state.raw_data)
            logger.debug(f"DataFrame hash: {df_hash}")
            categorized_file_path = os.path.join(categorized_folder, f"categorized_{df_hash}.csv")

            # MODIFIED: Check DataFrame hash instead of file hash
            # Purpose: Recognize identical data regardless of file format
            if df_hash == st.session_state.last_df_hash and os.path.exists(categorized_file_path):
                with st.spinner("Loading previously categorized data for this file..."):
                    st.session_state.categorized_data = pd.read_csv(categorized_file_path)
                    st.session_state.categorization_complete = True
                    st.sidebar.success("✅ Same data detected. Using cached categorized data.")
            else:
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.last_df_hash = df_hash
                st.session_state.categorized_data = pd.DataFrame()
                st.session_state.categorization_complete = False
                st.session_state.show_unlock_button = False
                st.sidebar.success(f"✅ Loaded {st.session_state.raw_data.shape[0]} records from {uploaded_file.name}.")

    # Sidebar Navigation
    st.sidebar.title("Navigation & Domain")
    page_options = ["Categorization"]
    if st.session_state.categorization_complete:
        page_options.extend(["Dashboard", "Predictions","Alerts", "Settings"])
                            #   "TCD Analysis", "Reports"])
    page = st.sidebar.selectbox("Select Page", page_options, index=0)
    
    # Domain filter
    selected_domain = "All"
    df_filtered = st.session_state.categorized_data.copy() if not st.session_state.categorized_data.empty else st.session_state.raw_data.copy()
    if page != "Categorization" and 'Domain' in df_filtered.columns:
        domain_options = ["All"] + df_filtered['Domain'].unique().tolist()
        selected_domain = st.sidebar.selectbox("Filter by Domain", options=domain_options)
        if selected_domain != "All":
            df_filtered = df_filtered[df_filtered['Domain'] == selected_domain]

    # Route pages
    if page == "Categorization":
        show_categorization(categorized_folder)
    elif st.session_state.categorization_complete:
        if page == "Dashboard":
            show_dashboard(df_filtered, selected_domain)
        elif page == "Predictions":
            show_predictions(df_filtered)
        # elif page == "TCD Analysis":
        #     show_tcd_analysis(df_filtered)
        elif page == "Alerts":
            show_alerts(df_filtered)
        # elif page == "Reports":
        #     show_reports(df_filtered)
        elif page == "Settings":
            show_settings()
    else:
        st.warning("Please complete categorization before accessing other pages.")

def show_categorization(categorized_folder: str):
    st.header("🗂️ Incident Categorization", divider="blue")
    st.markdown("""
        Categorize incidents using a pipeline: Keyword Matching → ML (SentenceTransformer + XGBoost) → Gemini LLM Fallback.
        Results show the distribution of predicted categories and confidence scores.
    """)

    if st.session_state.raw_data.empty:
        st.info("Please upload a CSV or Excel file in the sidebar to proceed with categorization.")
        return

    df = st.session_state.raw_data.copy()
    # MODIFIED: Added 'Created' to required_cols for forecasting compatibility
    # Purpose: Ensure forecasting for July 2025 can use the Created column
    required_cols = ['Short description', 'Service offering', 'Assignment group', 'Created']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"🚨 Missing required columns: {missing_cols}")
        return

    st.success(f"✅ Loaded **{len(df)} incidents** from uploaded file.")
    st.subheader("📊 Initial Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # NEW: Check for cached categorized data before running categorization
    # Purpose: Avoid re-categorization if the same data has been processed
    df_hash = compute_dataframe_hash(df)
    categorized_file_path = os.path.join(categorized_folder, f"categorized_{df_hash}.csv")
    logger.debug(f"Checking cached file: {categorized_file_path}")
    if df_hash == st.session_state.last_df_hash and os.path.exists(categorized_file_path):
        try:
            with st.spinner("Loading previously categorized data for this file..."):
                st.session_state.categorized_data = pd.read_csv(categorized_file_path)
                st.session_state.categorization_complete = True
                st.success("✅ This file has already been categorized. Loaded existing data.")
                
                # Display results directly
                st.subheader("🎉 Categorization Results", divider="gray")
                categorized_df = st.session_state.categorized_data

                # Predicted category distribution
                st.subheader("📈 Predicted Category Distribution")
                predicted_counts = categorized_df['Predicted Category'].value_counts().reset_index()
                predicted_counts.columns = ['Predicted Category', 'Incident Count']
                st.dataframe(predicted_counts, use_container_width=True)

                # Bar graph
                fig_category = px.bar(
                    predicted_counts,
                    x='Predicted Category',
                    y='Incident Count',
                    title="Incident Count by Predicted Category",
                    color='Predicted Category',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_category, use_container_width=True)

                # Classification method summary
                st.subheader("📊 Classification Method Summary")
                method_counts = categorized_df['Classification Method'].value_counts()
                confidence_avg = categorized_df[categorized_df['Classification Method'] != 'LLM']['Confidence'].mean()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Incidents", len(categorized_df))
                col2.metric("Avg. Confidence (Keyword/ML)", f"{confidence_avg:.2%}" if not pd.isna(confidence_avg) else "N/A")
                col3.metric("LLM Fallback Cases", len(categorized_df[categorized_df['Classification Method'] == 'LLM']))

                # Pie chart
                st.subheader("📊 Classification Method Distribution")
                fig_method = px.pie(
                    method_counts.reset_index(),
                    names='Classification Method',
                    values='count',
                    title="Distribution of Classification Methods",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_method, use_container_width=True)

                # Confidence distribution
                st.subheader("📉 Confidence Distribution (Keyword & ML)")
                confidence_data = categorized_df[categorized_df['Classification Method'] != 'LLM']['Confidence']
                if not confidence_data.empty:
                    fig_confidence = px.histogram(
                        confidence_data,
                        x="Confidence",
                        nbins=20,
                        title="Confidence Score Distribution",
                        color_discrete_sequence=['#636EFA']
                    )
                    st.plotly_chart(fig_confidence, use_container_width=True)
                else:
                    st.info("No confidence scores available (all cases classified by LLM).")

                # Results preview
                st.subheader("🔍 Results Preview")
                method_filter = st.multiselect(
                    "Filter by Classification Method",
                    options=categorized_df['Classification Method'].unique(),
                    default=categorized_df['Classification Method'].unique()
                )
                filtered_df = categorized_df[categorized_df['Classification Method'].isin(method_filter)]
                st.dataframe(filtered_df.head(50), use_container_width=True)

                # Download categorized data
                st.subheader("💾 Download Results")
                csv = BytesIO(categorized_df.to_csv(index=False).encode('utf-8'))
                st.download_button(
                    label="Download Categorized Data as CSV",
                    data=csv,
                    file_name="categorized_incidents.csv",
                    mime="text/csv",
                    type="secondary"
                )

                # Unlock button
                st.info("Ready to see other modules?")
                if st.button("Unlock Pages"):
                    st.session_state.categorization_complete = True
                    st.session_state.show_unlock_button = False
                    st.rerun()
                return
        except Exception as e:
            st.warning(f"Failed to load cached categorized data: {str(e)}. Proceeding with categorization.")
            logger.error(f"Error loading cached file {categorized_file_path}: {str(e)}")

    if st.button("🚀 Run Categorization", type="primary"):
        st.subheader("🔄 Classification Pipeline Progress", divider="gray")
        keyword_progress = st.progress(0, text="Step 1: Keyword Matching...")
        ml_progress = st.progress(0, text="Step 2: Machine Learning...")
        llm_progress = st.progress(0, text="Step 3: Gemini LLM Fallback...")
        log_container = st.container()
        
        try:
            with st.spinner("Performing keyword-based classification..."):
                keyword_progress.progress(33, text="Step 1: Keyword Matching (33% complete)")
                categorized_df = st.session_state.categorizer.process_incidents(df)
                log_container.info(f"Completed keyword matching for **{len(categorized_df)} incidents**.")
                keyword_progress.progress(100, text="Step 1: Keyword Matching (Complete)")

            low_conf_ml = len(categorized_df[categorized_df['Classification Method'] == 'ML'])
            if low_conf_ml > 0:
                with st.spinner(f"Applying ML classification to {low_conf_ml} low-confidence cases..."):
                    ml_progress.progress(50, text=f"Step 2: ML Classification ({low_conf_ml} cases, 50% complete)")
                    log_container.info(f"Applied ML classification to **{low_conf_ml} low-confidence cases**.")
                    ml_progress.progress(100, text="Step 2: ML Classification (Complete)")
            else:
                ml_progress.progress(100, text="Step 2: ML Classification (Skipped)")
                log_container.info("No low-confidence cases required ML classification.")

            llm_cases = len(categorized_df[categorized_df['Classification Method'] == 'LLM'])
            if llm_cases > 0:
                with st.spinner(f"Using Gemini LLM fallback for {llm_cases} cases..."):
                    llm_progress.progress(50, text=f"Step 3: Gemini LLM Fallback ({llm_cases} cases, 50% complete)")
                    log_container.info(f"Completed Gemini LLM fallback for **{llm_cases} cases**.")
                    llm_progress.progress(100, text="Step 3: Gemini LLM Fallback (Complete)")
            else:
                llm_progress.progress(100, text="Step 3: Gemini LLM Fallback (Skipped)")
                log_container.info("No cases required Gemini LLM fallback.")

            # MODIFIED: Use DataFrame hash for saving categorized data
            # Purpose: Ensure cached file matches DataFrame content
            file_hash = compute_dataframe_hash(df)
            categorized_file_path = os.path.join(categorized_folder, f"categorized_{file_hash}.csv")
            os.makedirs(categorized_folder, exist_ok=True)
            categorized_df.to_csv(categorized_file_path, index=False)
            st.session_state.categorized_data = categorized_df
            # NEW: Update last_df_hash after categorization
            st.session_state.last_df_hash = file_hash
            st.session_state.show_unlock_button = True
            st.subheader("🎉 Categorization Results", divider="gray")
            st.success("Categorization completed successfully! Data saved.")

            # Display predicted category distribution
            st.subheader("📈 Predicted Category Distribution")
            predicted_counts = categorized_df['Predicted Category'].value_counts().reset_index()
            predicted_counts.columns = ['Predicted Category', 'Incident Count']
            st.dataframe(predicted_counts, use_container_width=True)

            # Bar graph for predicted category distribution
            fig_category = px.bar(
                predicted_counts,
                x='Predicted Category',
                y='Incident Count',
                title="Incident Count by Predicted Category",
                color='Predicted Category',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_category, use_container_width=True)

            # Classification method summary
            st.subheader("📊 Classification Method Summary")
            method_counts = categorized_df['Classification Method'].value_counts()
            confidence_avg = categorized_df[categorized_df['Classification Method'] != 'LLM']['Confidence'].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Incidents", len(categorized_df))
            col2.metric("Avg. Confidence (Keyword/ML)", f"{confidence_avg:.2%}" if not pd.isna(confidence_avg) else "N/A")
            col3.metric("LLM Fallback Cases", llm_cases)

            # Pie chart for classification method distribution
            st.subheader("📊 Classification Method Distribution")
            fig_method = px.pie(
                method_counts.reset_index(),
                names='Classification Method',
                values='count',
                title="Distribution of Classification Methods",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_method, use_container_width=True)

            # Confidence distribution
            st.subheader("📉 Confidence Distribution (Keyword & ML)")
            confidence_data = categorized_df[categorized_df['Classification Method'] != 'LLM']['Confidence']
            if not confidence_data.empty:
                fig_confidence = px.histogram(
                    confidence_data,
                    x="Confidence",
                    nbins=20,
                    title="Confidence Score Distribution",
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
            else:
                st.info("No confidence scores available (all cases classified by LLM).")

            # Results preview
            st.subheader("🔍 Results Preview")
            method_filter = st.multiselect(
                "Filter by Classification Method",
                options=categorized_df['Classification Method'].unique(),
                default=categorized_df['Classification Method'].unique()
            )
            filtered_df = categorized_df[categorized_df['Classification Method'].isin(method_filter)]
            st.dataframe(filtered_df.head(50), use_container_width=True)

            # Download categorized data
            st.subheader("💾 Download Results")
            csv = BytesIO(categorized_df.to_csv(index=False).encode('utf-8'))
            st.download_button(
                label="Download Categorized Data as CSV",
                data=csv,
                file_name="categorized_incidents.csv",
                mime="text/csv",
                type="secondary"
            )

        except Exception as e:
            st.error(f"🚨 Error during categorization: {e}")
            log_container.error(f"Error details: {str(e)}")

    # Show unlock button if categorization is done
    if st.session_state.show_unlock_button:
        st.info("Ready to see other modules?")
        if st.button("Unlock Pages"):
            st.session_state.categorization_complete = True
            st.session_state.show_unlock_button = False
            st.rerun()

# def show_predictions():
#     st.header("🔮 Incident Predictions")
 
#     if 'predictor' not in st.session_state:
#         st.warning("⚠️ Predictor is not initialized.")
#         return
 
#     predictor = st.session_state.predictor
 
#     uploaded_file = st.file_uploader("📂 Upload Excel File (e.g. incidents_2024_2025.xlsx)", type=["xlsx"])
#     if uploaded_file is None:
#         st.info("Please upload an Excel file to proceed.")
#         return
 
#     model_path = "saved_models/predictor_model.pkl"
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
 
#     if st.button("Generate Predictions"):
#         with st.spinner("📊 Loading data..."):
#             load_result = predictor.load_data(uploaded_file)
#             if load_result['status'] != 'success':
#                 st.error(load_result['message'])
#                 return
#             st.success(load_result['message'])
 
#         train_months = pd.date_range("2024-01-01", "2025-05-01", freq="MS").strftime("%Y-%m").tolist()
#         target_month = "2025-06"
 
#         if not os.path.exists(model_path):
#             with st.spinner("🔍 Training models for the first time and saving..."):
#                 train_result = predictor.train_models(train_months=train_months)
#                 if train_result['status'] != 'success':
#                     st.error(train_result['message'])
#                     return
#                 st.success(f"✅ Trained {len(train_result['trained_categories'])} categories.")
 
#                 save_result = predictor.save_models(model_path)
#                 if save_result['status'] == 'success':
#                     st.success("📁 Models saved successfully.")
#                 else:
#                     st.warning(save_result['message'])
#         else:
#             with st.spinner("📂 Loading existing models..."):
#                 load_model_result = predictor.load_models(model_path)
#                 if load_model_result['status'] == 'success':
#                     st.success("✅ Loaded saved models.")
#                 else:
#                     st.warning(load_model_result['message'])
 
#         with st.spinner("🔮 Forecasting incidents..."):
#             forecast_result = predictor.forecast_next_period(target_month=target_month, train_months=train_months)
#             if forecast_result['status'] != 'success':
#                 st.error(forecast_result['message'])
#                 return
 
#             top_forecasts = forecast_result['top_3']
#             os.makedirs("data", exist_ok=True)
#             with open("data/top_predictions.json", "w") as f:
#                 json.dump(top_forecasts, f, indent=2, default=str)
#             st.subheader("🔥 Top High-Risk Predictions")
#             df_forecasts = pd.DataFrame(top_forecasts)
#             df_filtered = df_forecasts[df_forecasts['forecast'] > 0]
#             st.dataframe(df_filtered, use_container_width=True)
 
#             st.subheader("📊 Prediction Confidence Scatter Plot")
#             fig = px.scatter(
#                 df_filtered,
#                 x='category',
#                 y='forecast',
#                 size='forecast',
#                 color='category',
#                 title='Forecast Confidence by Category',
#                 labels={'forecast': 'Predicted Incident Count'},
#                 color_continuous_scale='Viridis'
#             )
#             st.plotly_chart(fig, use_container_width=True)
 
#             st.subheader("📈 Forecast Plots for Top Categories")
#             for item in top_forecasts:
#                 category = item['category']
#                 fig = predictor.generate_forecast_plot(category, target_month=target_month)
#                 if fig:
#                     st.markdown(f"**📌 {category}**")
#                     st.pyplot(fig)
#                 else:
#                     st.warning(f"Plot not available for {category}")
 
#         st.success("✅ Prediction process completed.")
def show_predictions(df: pd.DataFrame):
    st.header("🔮 Incident Predictions")
 
    if 'predictor' not in st.session_state:
        st.warning("⚠️ Predictor is not initialized.")
        return
 
    predictor = st.session_state.predictor
 
    # uploaded_file = st.file_uploader(
    #     "📂 Upload Excel File (e.g. incidents_2024_2025.xlsx)",
    #     type=["xlsx"]
    # )
    # if uploaded_file is None:
    #     st.info("Please upload an Excel file to proceed.")
    #     return
 
    model_path = "saved_models/predictor_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
 
    if st.button("Generate Predictions"):
        with st.spinner("📊 Loading data..."):
            load_result = predictor.load_data(df)
            if load_result['status'] != 'success':
                st.error(load_result['message'])
                return
            st.success(load_result['message'])
 
        # ✅ Dynamically get training months & next target month
        max_month = predictor.cat_ts.index.max()
        target_month_dt = (max_month + pd.offsets.MonthBegin(1))
        target_month = target_month_dt.strftime("%Y-%m")
 
        train_months = predictor.cat_ts.index.strftime("%Y-%m").tolist()
 
        if not os.path.exists(model_path):
            with st.spinner("🔍 Training models for the first time and saving..."):
                train_result = predictor.train_models(train_months=train_months)
                if train_result['status'] != 'success':
                    st.error(train_result['message'])
                    return
                st.success(f"✅ Trained {len(train_result['trained_categories'])} categories.")
 
                save_result = predictor.save_models(model_path)
                if save_result['status'] == 'success':
                    st.success("📁 Models saved successfully.")
                else:
                    st.warning(save_result['message'])
        else:
            with st.spinner("📂 Loading existing models..."):
                load_model_result = predictor.load_models(model_path)
                if load_model_result['status'] == 'success':
                    st.success("✅ Loaded saved models.")
                else:
                    st.warning(load_model_result['message'])
 
        with st.spinner(f"🔮 Forecasting incidents for {target_month}..."):
            forecast_result = predictor.forecast_next_period(
                target_month=target_month,
                train_months=train_months
            )
            if forecast_result['status'] != 'success':
                st.error(forecast_result['message'])
                return
 
            top_forecasts = forecast_result['top_3']
            os.makedirs("data", exist_ok=True)
            with open("data/top_predictions.json", "w") as f:
                json.dump(top_forecasts, f, indent=2, default=str)
 
            st.subheader(f"🔥 Top High-Risk Predictions for {target_month}")
            df_forecasts = pd.DataFrame(top_forecasts)
            df_filtered = df_forecasts[df_forecasts['forecast'] > 0]
            st.dataframe(df_filtered, use_container_width=True)
 
            st.subheader("📊 Prediction Confidence Scatter Plot")
            fig = px.scatter(
                df_filtered,
                x='category',
                y='forecast',
                size='forecast',
                color='category',
                title='Forecast Confidence by Category',
                labels={'forecast': 'Predicted Incident Count'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
 
            st.subheader("📈 Forecast Plots for Top Categories")
            for item in top_forecasts:
                category = item['category']
                fig = predictor.generate_forecast_plot(category, target_month=target_month)
                if fig:
                    st.markdown(f"**📌 {category}**")
                    st.pyplot(fig)
                else:
                    st.warning(f"Plot not available for {category}")
 
        st.success("✅ Prediction process completed.")

# def show_dashboard(df: pd.DataFrame, selected_domain: str):
#     st.title("📊 Incident Visualization Dashboard")
    
#     if df.empty:
#         st.error(f"No data available for the selected domain: '{selected_domain}'.")
#         return

#     st.info(f"Displaying data for '{selected_domain}' domain. Total records: {df.shape[0]}")
    
#     st.subheader("📌 Key Performance Indicators")
#     kpis = st.session_state.dashboard.get_kpi_metrics(df)
#     kpi_cols = st.columns(len(kpis))
#     for i, (label, value) in enumerate(kpis.items()):
#         kpi_cols[i].metric(label, f"{value}")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("📈 Incident Trends Over Time by Category")
#         st.plotly_chart(st.session_state.dashboard.incident_trends_over_time(df), use_container_width=True)
#         st.subheader("📊 Incident Category Distribution")
#         st.plotly_chart(st.session_state.dashboard.category_distribution(df), use_container_width=True)
#     with col2:
#         st.subheader("🔥 Heatmap of Incidents by Hour & Weekday")
#         st.plotly_chart(st.session_state.dashboard.severity_by_hour_heatmap(df), use_container_width=True)
#         st.subheader("🌍 Geographic Distribution")
#         st.plotly_chart(st.session_state.dashboard.geographic_map(df), use_container_width=True)

#     st.subheader("📉 Mean Time to Resolution (MTTR) by Category")
#     st.plotly_chart(st.session_state.dashboard.mttr_trend(df), use_container_width=True)

#     st.subheader("📊 Priority vs Incident State Distribution")
#     chart_type = st.radio("Choose chart type:", options=["Bar Chart", "Sunburst"], horizontal=True)
#     category_options = df['Predicted Category'].dropna().unique().tolist()
#     selected_category = st.selectbox("Filter by Predicted Category", options=["All"] + category_options, key="category_filter_1")
#     if chart_type == "Bar Chart":
#         if selected_category == "All":
#             st.plotly_chart(st.session_state.dashboard.priority_vs_state_bar_all(df), use_container_width=True)
#         else:
#             st.plotly_chart(st.session_state.dashboard.priority_vs_state_bar_category(df, selected_category), use_container_width=True)
#     else:
#         if selected_category == "All":
#             st.plotly_chart(st.session_state.dashboard.priority_vs_state_distribution(df), use_container_width=True)
#         else:
#             st.plotly_chart(st.session_state.dashboard.priority_vs_state_sunburst_by_category(df, selected_category), use_container_width=True)


    
    
#     st.subheader("📦 Statistical Inference: Boxplot by Category")
#     category_options = df['Predicted Category'].dropna().unique().tolist()
#     selected_category = st.selectbox("Select a Category", sorted(category_options), key="boxplot_category_filter")
#     fig = st.session_state.dashboard.generate_boxplot_for_category(df, selected_category)
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("🌳 Treemap Visualizations")
#     treemaps = st.session_state.dashboard.hierarchical_treemap(df)
#     for fig in treemaps:
#         st.plotly_chart(fig, use_container_width=True)

#     st.subheader("🔠 Category-wise Bigram Word Clouds")
#     wordcloud_key = f'bigram_clouds_{selected_domain}'
#     if wordcloud_key not in st.session_state:
#         st.session_state[wordcloud_key] = None
#     if st.button("Generate Bigram Word Clouds", key="wordcloud_button"):
#         with st.spinner("Generating bigram word clouds..."):
#             st.session_state[wordcloud_key] = st.session_state.dashboard.wordcloud_bigrams_by_category(df)

#     if st.session_state[wordcloud_key] is not None:
#         cols = st.columns(3)
#         for i, (category, img_base64) in enumerate(st.session_state[wordcloud_key].items()):
#             with cols[i % 3]:
#                 st.markdown(f"**{category}**")
#                 st.image(f"data:image/png;base64,{img_base64}", use_container_width=True)
#     else:
#         st.info("Click the button above to generate the word clouds.")

#     st.subheader("🔄 Incident Sankey Flow")
#     fig_sankey = st.session_state.dashboard.incident_sankey_channel_category_priority_state(df)
#     st.plotly_chart(fig_sankey, use_container_width=True)

#     st.subheader("📊 Incident Faceted Multidimensional Bar-Chart")
#     fig_1 = st.session_state.dashboard.facet_bar_chart(df)
#     st.plotly_chart(fig_1, use_container_width=True)

#     st.subheader("♻️ Reopen Insights")
#     total_reopened, reopen_rate = st.session_state.dashboard.reopen_analysis(df)
#     col1, col2 = st.columns(2)
#     col1.metric("🔁 Total Reopened Incidents", total_reopened)
#     col2.metric("📉 Reopen Rate (%)", f"{reopen_rate:.2f}%")
#     st.plotly_chart(st.session_state.dashboard.category_wise_reopens(df), use_container_width=True)

#     st.subheader("♻️ Reopen Insights - LLM Powered")
#     gemini_insights_key = f'gemini_reopen_insights_{selected_domain}'
#     if gemini_insights_key not in st.session_state:
#         st.session_state[gemini_insights_key] = None
#     df_reopened_subset = st.session_state.dashboard.get_reopened_incidents_subset(df)
#     if not df_reopened_subset.empty:
#         if st.button("Generate LLM Insights", key="llm_insights_button"):
#             with st.spinner("Analyzing with Gemini Flash 2.5..."):
#                 prompt_text = st.session_state.dashboard.format_reopen_prompt(df_reopened_subset)
#                 st.session_state[gemini_insights_key] = st.session_state.dashboard.get_gemini_reopen_insights(prompt_text)
#         if st.session_state[gemini_insights_key] is not None:
#             st.write(st.session_state[gemini_insights_key])
#         else:
#             st.info("Click the button above to generate LLM-powered insights.")
#     else:
#         st.info("No reopened incidents found to analyze.")

#     st.subheader("🗃 Raw Data Table")
#     st.dataframe(df.head(50))

#     # New Sections
#         # ... (keep existing code up to Priority vs Incident State)
#     # At the top, initialize session state keys if not present
#     if 'recurring_data' not in st.session_state:
#         st.session_state.recurring_data = None
#     if 'rca_data' not in st.session_state:
#         st.session_state.rca_data = None

#     st.subheader("🔄 Recurring Incidents Analysis")
#     if st.button("Generate Recurring Incidents Analysis"):
        
#         st.session_state.recurring_data = st.session_state.dashboard.analyze_recurring_incidents(df, eps=0.3)  # Pass eps if using the updated function
#         if st.session_state.recurring_data and st.session_state.recurring_data['clusters']:
#             for cluster in st.session_state.recurring_data['clusters']:
#                 with st.expander(f"{cluster['summary']} | Occurrences: {cluster['occurrences']} | Avg MTTR: {cluster['avg_mttr']} hours"):
#                     st.dataframe(pd.DataFrame(cluster['incidents']))
#         else:
#             st.info("No recurring clusters found.")


#     st.subheader("🔍 RCA and Recommendations")
#     if st.button("Generate RCA for All Clusters"):
#         if st.session_state.recurring_data and st.session_state.recurring_data['clusters']:
#             st.session_state.rca_data = st.session_state.dashboard.generate_rca_for_clusters(st.session_state.recurring_data['clusters'])
#         else:
#             st.warning("Run Recurring Incidents Analysis first.")

#     if st.session_state.rca_data:
#         for i, rca in enumerate(st.session_state.rca_data):
#             with st.expander(f"RCA for Cluster {i+1}"):
#                 st.write(f"**Root Cause Analysis:** {rca['rca']}")
#                 st.write(f"**Recommendations:** {rca['recommendations']}")
#     else:
#         st.info("No RCA generated yet.")

#     st.subheader("😊 Sentiment Analysis")
#     sentiment_data = st.session_state.dashboard.sentiment_analysis(df)
#     overall = sentiment_data['overall']
#     st.write(f"**Overall Average Polarity:** {overall['Average_Polarity']:.2f} | **Average Subjectivity:** {overall['Average_Subjectivity']:.2f}")
#     st.write(f"**Positive:** {overall['Positive_Percentage']:.2f}% | **Negative:** {overall['Negative_Percentage']:.2f}% | **Neutral:** {overall['Neutral_Percentage']:.2f}%")
    
#     st.subheader("Sentiment Per Category")
#     per_cat_df = pd.DataFrame(sentiment_data['per_category']).T.reset_index()
#     per_cat_df.columns = ['Category', 'Avg Polarity', 'Avg Subjectivity', 'Positive %', 'Negative %', 'Neutral %']
#     st.dataframe(per_cat_df, use_container_width=True)
    
#     # Bar chart for visualization
#     fig_sentiment = px.bar(per_cat_df, x='Category', y=['Positive %', 'Negative %', 'Neutral %'], title="Sentiment Distribution by Category")
#     st.plotly_chart(fig_sentiment, use_container_width=True)

#     st.subheader("📝 LDA Topic Modeling")
#     num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)
#     topic_data = st.session_state.dashboard.lda_topic_modeling(df, num_topics)
#     for topic, words in topic_data['topics'].items():
#         st.write(f"**{topic}:** {words}")
    
#     # Distribution pie chart
#     dist_df = pd.DataFrame(list(topic_data['distribution'].items()), columns=['Topic', 'Proportion'])
#     fig_topic = px.pie(dist_df, names='Topic', values='Proportion', title="Topic Distribution")
#     st.plotly_chart(fig_topic, use_container_width=True)
    
#     # Examples
#     st.subheader("Topic Examples")
#     for topic, ex_list in topic_data['examples'].items():
#         st.write(f"**{topic}:**")
#         for ex in ex_list:
#             st.write(f"- {ex[:200]}...")  # Truncate for display

#     # ... (keep the statistical inference boxplot section)

def show_dashboard(df: pd.DataFrame, selected_domain: str):
    st.title("📊 Incident Visualization Dashboard")
    
    if df.empty:
        st.error(f"No data available for the selected domain: '{selected_domain}'.")
        return

    st.info(f"Displaying data for '{selected_domain}' domain. Total records: {df.shape[0]}")
    
    st.subheader("📌 Key Performance Indicators")
    kpis_key = f'kpis_{selected_domain}'
    if kpis_key not in st.session_state:
        st.session_state[kpis_key] = st.session_state.dashboard.get_kpi_metrics(df)
    kpis = st.session_state[kpis_key]
    kpi_cols = st.columns(len(kpis))
    for i, (label, value) in enumerate(kpis.items()):
        kpi_cols[i].metric(label, f"{value}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Incident Trends Over Time by Category")
        trends_key = f'incident_trends_{selected_domain}'
        if trends_key not in st.session_state:
            st.session_state[trends_key] = st.session_state.dashboard.incident_trends_over_time(df)
        st.plotly_chart(st.session_state[trends_key], use_container_width=True)
        
        st.subheader("📊 Incident Category Distribution")
        cat_dist_key = f'category_distribution_{selected_domain}'
        if cat_dist_key not in st.session_state:
            st.session_state[cat_dist_key] = st.session_state.dashboard.category_distribution(df)
        st.plotly_chart(st.session_state[cat_dist_key], use_container_width=True)
    
    with col2:
        st.subheader("🔥 Heatmap of Incidents by Hour & Weekday")
        heatmap_key = f'severity_heatmap_{selected_domain}'
        if heatmap_key not in st.session_state:
            st.session_state[heatmap_key] = st.session_state.dashboard.severity_by_hour_heatmap(df)
        st.plotly_chart(st.session_state[heatmap_key], use_container_width=True)
        
        st.subheader("🌍 Geographic Distribution")
        geo_key = f'geographic_map_{selected_domain}'
        if geo_key not in st.session_state:
            st.session_state[geo_key] = st.session_state.dashboard.geographic_map(df)
        st.plotly_chart(st.session_state[geo_key], use_container_width=True)

    st.subheader("📉 Mean Time to Resolution (MTTR) by Category")
    mttr_key = f'mttr_trend_{selected_domain}'
    if mttr_key not in st.session_state:
        st.session_state[mttr_key] = st.session_state.dashboard.mttr_trend(df)
    st.plotly_chart(st.session_state[mttr_key], use_container_width=True)

    st.subheader("📊 Priority vs Incident State Distribution")
    chart_type = st.radio("Choose chart type:", options=["Bar Chart", "Sunburst"], horizontal=True)
    category_options = df['Predicted Category'].dropna().unique().tolist()
    selected_category = st.selectbox("Filter by Predicted Category", options=["All"] + category_options, key="category_filter_1")
    priority_state_key = f'priority_state_{selected_domain}_{chart_type}_{selected_category}'
    if priority_state_key not in st.session_state:
        if chart_type == "Bar Chart":
            if selected_category == "All":
                st.session_state[priority_state_key] = st.session_state.dashboard.priority_vs_state_bar_all(df)
            else:
                st.session_state[priority_state_key] = st.session_state.dashboard.priority_vs_state_bar_category(df, selected_category)
        else:
            if selected_category == "All":
                st.session_state[priority_state_key] = st.session_state.dashboard.priority_vs_state_distribution(df)
            else:
                st.session_state[priority_state_key] = st.session_state.dashboard.priority_vs_state_sunburst_by_category(df, selected_category)
    st.plotly_chart(st.session_state[priority_state_key], use_container_width=True)

    st.subheader("📦 Statistical Inference: Boxplot by Category")
    category_options = df['Predicted Category'].dropna().unique().tolist()
    selected_category = st.selectbox("Select a Category", sorted(category_options), key="boxplot_category_filter")
    boxplot_key = f'boxplot_{selected_domain}_{selected_category}'
    if boxplot_key not in st.session_state:
        st.session_state[boxplot_key] = st.session_state.dashboard.generate_boxplot_for_category(df, selected_category)
    st.plotly_chart(st.session_state[boxplot_key], use_container_width=True)

    st.subheader("🌳 Treemap Visualizations")
    treemap_key = f'treemaps_{selected_domain}'
    if treemap_key not in st.session_state:
        st.session_state[treemap_key] = st.session_state.dashboard.hierarchical_treemap(df)
    for fig in st.session_state[treemap_key]:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔠 Category-wise Bigram Word Clouds")
    wordcloud_key = f'bigram_clouds_{selected_domain}'
    if wordcloud_key not in st.session_state:
        st.session_state[wordcloud_key] = None
    if st.button("Generate Bigram Word Clouds", key=f"wordcloud_button_{selected_domain}"):
        with st.spinner("Generating bigram word clouds..."):
            st.session_state[wordcloud_key] = st.session_state.dashboard.wordcloud_bigrams_by_category(df)
    if st.session_state[wordcloud_key] is not None:
        cols = st.columns(3)
        for i, (category, img_base64) in enumerate(st.session_state[wordcloud_key].items()):
            with cols[i % 3]:
                st.markdown(f"**{category}**")
                st.image(f"data:image/png;base64,{img_base64}", use_container_width=True)
    else:
        st.info("Click the button above to generate the word clouds.")

    st.subheader("🔄 Incident Sankey Flow")
    sankey_key = f'sankey_flow_{selected_domain}'
    if sankey_key not in st.session_state:
        st.session_state[sankey_key] = st.session_state.dashboard.incident_sankey_channel_category_priority_state(df)
    st.plotly_chart(st.session_state[sankey_key], use_container_width=True)

    st.subheader("📊 Incident Faceted Multidimensional Bar-Chart")
    facet_key = f'facet_bar_{selected_domain}'
    if facet_key not in st.session_state:
        st.session_state[facet_key] = st.session_state.dashboard.facet_bar_chart(df)
    st.plotly_chart(st.session_state[facet_key], use_container_width=True)

    st.subheader("♻️ Reopen Insights")
    reopen_key = f'reopen_metrics_{selected_domain}'
    if reopen_key not in st.session_state:
        total_reopened, reopen_rate = st.session_state.dashboard.reopen_analysis(df)
        st.session_state[reopen_key] = {'total_reopened': total_reopened, 'reopen_rate': reopen_rate}
    col1, col2 = st.columns(2)
    col1.metric("🔁 Total Reopened Incidents", st.session_state[reopen_key]['total_reopened'])
    col2.metric("📉 Reopen Rate (%)", f"{st.session_state[reopen_key]['reopen_rate']:.2f}%")
    
    reopen_chart_key = f'reopen_chart_{selected_domain}'
    if reopen_chart_key not in st.session_state:
        st.session_state[reopen_chart_key] = st.session_state.dashboard.category_wise_reopens(df)
    st.plotly_chart(st.session_state[reopen_chart_key], use_container_width=True)

    st.subheader("♻️ Reopen Insights - LLM Powered")
    gemini_insights_key = f'gemini_reopen_insights_{selected_domain}'
    if gemini_insights_key not in st.session_state:
        st.session_state[gemini_insights_key] = None
    df_reopened_subset = st.session_state.dashboard.get_reopened_incidents_subset(df)
    if not df_reopened_subset.empty:
        if st.button("Generate LLM Insights", key=f"llm_insights_button_{selected_domain}"):
            with st.spinner("Analyzing with Gemini Flash 2.5..."):
                prompt_text = st.session_state.dashboard.format_reopen_prompt(df_reopened_subset)
                st.session_state[gemini_insights_key] = st.session_state.dashboard.get_gemini_reopen_insights(prompt_text)
        if st.session_state[gemini_insights_key] is not None:
            st.write(st.session_state[gemini_insights_key])
        else:
            st.info("Click the button above to generate LLM-powered insights.")
    else:
        st.info("No reopened incidents found to analyze.")

    st.subheader("🗃 Raw Data Table")
    st.dataframe(df.head(50))

    st.subheader("😊 Sentiment Analysis")
    sentiment_key = f'sentiment_data_{selected_domain}'
    if sentiment_key not in st.session_state:
        st.session_state[sentiment_key] = st.session_state.dashboard.sentiment_analysis(df)
    sentiment_data = st.session_state[sentiment_key]
    overall = sentiment_data['overall']
    st.write(f"**Overall Average Polarity:** {overall['Average_Polarity']:.2f} | **Average Subjectivity:** {overall['Average_Subjectivity']:.2f}")
    st.write(f"**Positive:** {overall['Positive_Percentage']:.2f}% | **Negative:** {overall['Negative_Percentage']:.2f}% | **Neutral:** {overall['Neutral_Percentage']:.2f}%")
    
    st.subheader("Sentiment Per Category")
    per_cat_df = pd.DataFrame(sentiment_data['per_category']).T.reset_index()
    per_cat_df.columns = ['Category', 'Avg Polarity', 'Avg Subjectivity', 'Positive %', 'Negative %', 'Neutral %']
    st.dataframe(per_cat_df, use_container_width=True)
    
    fig_sentiment_key = f'fig_sentiment_{selected_domain}'
    if fig_sentiment_key not in st.session_state:
        st.session_state[fig_sentiment_key] = px.bar(per_cat_df, x='Category', y=['Positive %', 'Negative %', 'Neutral %'], title="Sentiment Distribution by Category")
    st.plotly_chart(st.session_state[fig_sentiment_key], use_container_width=True)

    # st.subheader("📝 LDA Topic Modeling")
    # num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, key=f"lda_slider_{selected_domain}")
    # lda_key = f'lda_data_{selected_domain}_{num_topics}'
    # if lda_key not in st.session_state:
    #     st.session_state[lda_key] = st.session_state.dashboard.lda_topic_modeling(df, num_topics)
    # topic_data = st.session_state[lda_key]
    # for topic, words in topic_data['topics'].items():
    #     st.write(f"**{topic}:** {words}")
    
    # dist_df = pd.DataFrame(list(topic_data['distribution'].items()), columns=['Topic', 'Proportion'])
    # fig_topic_key = f'fig_topic_{selected_domain}_{num_topics}'
    # if fig_topic_key not in st.session_state:
    #     st.session_state[fig_topic_key] = px.pie(dist_df, names='Topic', values='Proportion', title="Topic Distribution")
    # st.plotly_chart(st.session_state[fig_topic_key], use_container_width=True)
    
    # st.subheader("Topic Examples")
    # for topic, ex_list in topic_data['examples'].items():
    #     st.write(f"**{topic}:**")
    #     for ex in ex_list:
    #         st.write(f"- {ex[:200]}...")  # Truncate for display

    # recurring_key = f'recurring_data_{selected_domain}'
    # if recurring_key not in st.session_state:
    #     st.session_state[recurring_key] = None

    # st.subheader("🔄 Recurring Incidents Analysis")
    # if st.button("Generate Recurring Incidents Analysis", key=f"recurring_button_{selected_domain}"):
    #     with st.spinner("Analyzing with Gemini Flash 2.0...may take upto 1-2 minutes for 500 incident volume"):
    #         st.session_state[recurring_key] = st.session_state.dashboard.analyze_recurring_incidents(df, eps=0.2, batch_size=64)
    # if st.session_state[recurring_key] and st.session_state[recurring_key]['clusters']:
    #     for cluster in st.session_state[recurring_key]['clusters']:
    #         with st.expander(f"{cluster['summary']} | Occurrences: {cluster['occurrences']} | Avg MTTR: {cluster['avg_mttr']} hours"):
    #             st.dataframe(pd.DataFrame(cluster['incidents']))
    # else:
    #     st.info("No recurring clusters found or analysis not run yet.")

    # rca_key = f'rca_data_{selected_domain}'
    # if rca_key not in st.session_state:
    #     st.session_state[rca_key] = None

    # st.subheader("🔍 RCA and Recommendations")
    # if st.button("Generate RCA for All Clusters", key=f"rca_button_{selected_domain}"):
    #     if st.session_state[recurring_key] and st.session_state[recurring_key]['clusters']:
    #         with st.spinner("Analyzing with Gemini Flash 2.0...may take upto 1-2 minutes for 500 incident volume"):
    #             st.session_state[rca_key] = st.session_state.dashboard.generate_rca_for_clusters(st.session_state[recurring_key]['clusters'])
    #     else:
    #         st.warning("Run Recurring Incidents Analysis first.")
    # if st.session_state[rca_key]:
    #     for i, rca in enumerate(st.session_state[rca_key]):
    #         with st.expander(f"RCA for Cluster {i+1}"):
    #             st.write(f"**Root Cause Analysis:** {rca['rca']}")
    #             st.write(f"**Recommendations:** {rca['recommendations']}")
    # else:
    #     st.info("No RCA generated yet.")

def show_alerts(df: pd.DataFrame):
    st.header("🚨 Alert Management")
    
    if df.empty:
        st.error("No categorized data available for alerts.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        ml_alerts = []
        json_path = "data/top_predictions.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                ml_alerts = json.load(f)
        
        if ml_alerts:
            st.markdown("### 🤖 ML-Based Predictions")
            for idx, alert in enumerate(ml_alerts):
                alert_id = f"ML_{idx+1}"
                severity = "Critical" if alert.get("confidence score", 100) > 70 else "Warning"
                emoji = "🔴" if severity == "Critical" else "🟡"
                with st.expander(f"{emoji} {alert_id} - {severity}"):
                    st.write(f"**Message:** Predicted spike in `{alert['category']}` incidents")
                    st.write(f"**Forecast Value:** {alert['forecast']}")
                    st.write(f"**Month:** {alert['month']}")
                    if "confidence score" in alert:
                        st.write(f"**Confidence Score:** {alert['confidence score']}")
        else:
            st.info("No ML-based alerts to show. Run predictions to generate alerts.")
    
    with col2:
        st.subheader("📊 Weekly High-Priority Incident Alerts")
        try:
            incident_alerts = st.session_state.alert_manager.generate_weekly_priority_alerts(df)
            if not incident_alerts:
                st.success("✅ No high-priority incidents found in the last week.")
            else:
                for idx, incident in enumerate(incident_alerts):
                    alert_id = f"INC_{idx+1}"
                    emoji = "🔴" if incident['priority'].lower() == "critical" else "🟠"
                    with st.expander(f"{emoji} {alert_id} - {incident['priority']} Priority"):
                        st.write(f"**Number:** {incident['Number']}")
                        st.write(f"**Opened:** {incident['Opened']}")
                        st.write(f"**Short Description:** {incident['short_description']}")
                        st.write(f"**Assignment Group:** {incident['assignment_group']}")
                        st.write(f"**Category:** {incident['Predicted Category']}")
                        st.write(f"**Service Offering:** {incident['service_offering']}")
        except Exception as e:
            st.error(f"❌ Failed to process alerts: {e}")

def show_settings():
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("API Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
        servicenow_url = st.text_input("ServiceNow Instance URL", placeholder="https://your-instance.service-now.com")
        servicenow_user = st.text_input("ServiceNow Username")
        servicenow_pass = st.text_input("ServiceNow Password", type="password")
    
    with col2:
        st.subheader("Model Configuration")
        model_refresh = st.selectbox("Model Refresh Interval", ["Daily", "Weekly", "Monthly"])
        prediction_window = st.selectbox("Default Prediction Window", ["24 hours", "7 days", "30 days"])
        alert_frequency = st.selectbox("Alert Check Frequency", ["Every 5 minutes", "Every 15 minutes", "Every hour"])
    
    if st.button("Save Settings"):
        st.success("✅ Settings saved successfully!")
        config = {
            "gemini_api_key": gemini_api_key,
            "servicenow_url": servicenow_url,
            "model_refresh": model_refresh,
            "prediction_window": prediction_window,
            "alert_frequency": alert_frequency
        }

# def show_tcd_analysis(df: pd.DataFrame):
#     st.header("⚙️ TCD (Technical Change Decision) Analysis")
    
#     if df.empty:
#         st.error("No categorized data available for TCD analysis.")
#         return

#     st.subheader("Change Impact Analysis")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("**Recent Changes:**")
#         changes = [
#             {"Change ID": "CHG001234", "Type": "Software Update", "Risk": "Medium", "Status": "Approved"},
#             {"Change ID": "CHG001235", "Type": "Config Change", "Risk": "Low", "Status": "Implemented"},
#             {"Change ID": "CHG001236", "Type": "Hardware Upgrade", "Risk": "High", "Status": "Pending"}
#         ]
#         df_changes = pd.DataFrame(changes)
#         st.dataframe(df_changes, use_container_width=True)
    
#     with col2:
#         st.write("**Risk Assessment:**")
#         risk_data = {"High": 15, "Medium": 25, "Low": 60}
#         fig = px.bar(x=list(risk_data.keys()), y=list(risk_data.values()), title='Change Risk Distribution')
#         st.plotly_chart(fig, use_container_width=True)
    
#     st.subheader("AI-Powered TCD Recommendations")
#     if st.button("Generate TCD Analysis with Gemini"):
#         with st.spinner("Analyzing with Gemini AI..."):
#             analysis = st.session_state.gemini_analyzer.analyze_tcd_data(changes)
#             st.write("**Gemini Analysis Results:**")
#             st.info("""
#             🤖 **AI Recommendation:** Based on the recent changes analysis:
#             - **CHG001236** (Hardware Upgrade) requires additional review due to high risk classification
#             - Recommend implementing **phased rollout** for software updates during low-traffic periods
#             - **Configuration changes** show positive trend with minimal incident correlation
#             - Consider implementing **automated rollback procedures** for high-risk changes
#             """)

# def show_reports(df: pd.DataFrame):
#     st.header("📋 Executive Reports")
    
#     if df.empty:
#         st.error("No categorized data available for reports.")
#         return

#     report_type = st.selectbox(
#         "Select Report Type",
#         ["Weekly Summary", "Monthly Analysis", "TCD Impact Report", "Predictive Insights"]
#     )
    
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
#     with col2:
#         end_date = st.date_input("End Date", datetime.now())
    
#     if st.button("Generate Report"):
#         with st.spinner("Generating report..."):
#             st.success("✅ Report generated successfully!")
#             st.subheader(f"{report_type} - {start_date} to {end_date}")
#             st.write("**Executive Summary:**")
#             st.info("""
#             📊 **Key Metrics:**
#             - Total incidents: 156 (↓ 12% from previous period)
#             - Average resolution time: 4.2 hours (↓ 15%)
#             - TCD-related incidents: 8 (↓ 20%)
#             - Prediction accuracy: 87%
            
#             🎯 **Recommendations:**
#             - Continue focus on proactive monitoring
#             - Implement suggested SOP updates
#             - Review change management process for high-risk changes
#             """)
#             if st.button("📥 Download Report"):
#                 st.success("Report downloaded successfully!")

# NEW: Modified entry point to check authentication
# Purpose: Require login before accessing the main app
if __name__ == "__main__":
    # Initialize authentication state
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
        st.session_state.theme = "light"  # Default theme

    if not st.session_state.is_authenticated:
        show_login_page()
    else:
        main()