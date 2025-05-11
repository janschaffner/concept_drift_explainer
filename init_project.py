import os
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    
    # Define the base directory (where this script is located)
    base_dir = Path(__file__).parent
    
    # Define the directories to create
    directories = [
        "backend/api",
        "backend/core",
        "backend/data",
        "backend/utils",
        "frontend/pages",
        "frontend/components",
        "data/raw",
        "data/processed",
        "data/outputs",
        "models",
        "config",
        "tests"
    ]
    
    # Create the directories
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create an __init__.py file in each directory
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
    
    print("Directory structure created successfully!")
    
    # Create placeholder .gitkeep files for empty data directories
    for data_dir in ["data/raw", "data/processed", "data/outputs"]:
        gitkeep_file = base_dir / data_dir / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    # Create .env.example file if it doesn't exist
    env_example = base_dir / ".env.example"
    if not env_example.exists():
        with open(env_example, "w") as f:
            f.write("""# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Other API Keys (as needed)
# CONFLUENCE_API_KEY=your_confluence_api_key_here
# SHAREPOINT_API_KEY=your_sharepoint_api_key_here

# Environment settings
ENVIRONMENT=development  # development, testing, production
""")
    
    # Create configuration settings file
    config_dir = base_dir / "config"
    settings_file = config_dir / "settings.py"
    if not settings_file.exists():
        with open(settings_file, "w") as f:
            f.write("""import os
from dotenv import load_dotenv
from pathlib import Path

# Find the project root directory
project_root = Path(__file__).parent.parent

# Load environment variables from .env file
load_dotenv(project_root / '.env')

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CONFLUENCE_API_KEY = os.getenv('CONFLUENCE_API_KEY')
SHAREPOINT_API_KEY = os.getenv('SHAREPOINT_API_KEY')

# Environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Default paths
DATA_RAW_PATH = project_root / 'data' / 'raw'
DATA_PROCESSED_PATH = project_root / 'data' / 'processed' 
DATA_OUTPUTS_PATH = project_root / 'data' / 'outputs'

# Create directories if they don't exist
DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
DATA_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

# Default analysis settings
DEFAULT_TIME_WINDOW = 30  # days
DEFAULT_LLM_MODEL = 'gpt-4o'
DEFAULT_LLM_TEMPERATURE = 0.0

# Function to get configuration as a dictionary
def get_config():
    return {
        'openai_api_key': OPENAI_API_KEY,
        'confluence_api_key': CONFLUENCE_API_KEY,
        'sharepoint_api_key': SHAREPOINT_API_KEY,
        'environment': ENVIRONMENT,
        'data_raw_path': str(DATA_RAW_PATH),
        'data_processed_path': str(DATA_PROCESSED_PATH),
        'data_outputs_path': str(DATA_OUTPUTS_PATH),
        'default_time_window': DEFAULT_TIME_WINDOW,
        'default_llm_model': DEFAULT_LLM_MODEL,
        'default_llm_temperature': DEFAULT_LLM_TEMPERATURE,
    }
""")
    
    # Create requirements.txt file if it doesn't exist
    req_file = base_dir / "requirements.txt"
    if not req_file.exists():
        with open(req_file, "w") as f:
            f.write("""# Core dependencies
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn

# LLM integration
openai

# Environment variables
python-dotenv

# Data processing
python-docx  # For Word documents
python-pptx  # For PowerPoint files
beautifulsoup4  # For HTML parsing (e.g., intranet)
PyPDF2  # For PDF files

# Frontend
streamlit
streamlit-agraph  # For network visualizations
streamlit-timeline  # For timeline visualizations

# Testing
pytest
""")
    
    # Create main application file if it doesn't exist
    frontend_dir = base_dir / "frontend"
    app_file = frontend_dir / "app.py"
    if not app_file.exists():
        with open(app_file, "w") as f:
            f.write("""import streamlit as st
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.components.sidebar import render_sidebar
from config.settings import get_config, OPENAI_API_KEY

# Get configuration
config = get_config()

def main():
    st.set_page_config(
        page_title="Context-Aware Process Drift Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Context-Aware Process Drift Analysis")
    
    # Render sidebar
    selected_option = render_sidebar()
    
    # Main content based on sidebar selection
    if selected_option == "Upload Data":
        render_upload_page()
    elif selected_option == "Configure Analysis":
        render_analysis_config_page()
    elif selected_option == "Results":
        render_results_page()
    elif selected_option == "About":
        render_about_page()

def render_upload_page():
    st.header("Upload Data")
    
    # Upload CV4CDD output
    st.subheader("CV4CDD Change Point Data")
    change_point_file = st.file_uploader(
        "Upload CV4CDD JSON output file", 
        type=["json"]
    )
    
    # Upload event log
    st.subheader("Event Log")
    event_log_file = st.file_uploader(
        "Upload event log file (optional)", 
        type=["csv", "xes"]
    )
    
    # Upload context data
    st.subheader("Context Data")
    st.write("Upload files containing context information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Internal Context")
        organigram_files = st.file_uploader(
            "Upload organigram files", 
            type=["xlsx", "csv", "json"], 
            accept_multiple_files=True
        )
        
        internal_docs = st.file_uploader(
            "Upload internal documents", 
            type=["docx", "pdf", "pptx", "txt", "html"], 
            accept_multiple_files=True
        )
    
    with col2:
        st.write("External Context")
        regulation_files = st.file_uploader(
            "Upload regulation documents", 
            type=["docx", "pdf", "txt"], 
            accept_multiple_files=True
        )
        
        external_docs = st.file_uploader(
            "Upload external context documents", 
            type=["docx", "pdf", "txt", "html"], 
            accept_multiple_files=True
        )
    
    if st.button("Process Uploaded Files"):
        # Add processing logic here
        st.success("Files processed successfully!")
        # Store processed data in session state for next steps

def render_analysis_config_page():
    st.header("Configure Analysis")
    
    # Time window configuration
    st.subheader("Temporal Analysis Settings")
    time_window = st.slider(
        "Time window for context analysis (days)",
        min_value=1,
        max_value=90,
        value=config['default_time_window']
    )
    
    # LLM configuration
    st.subheader("LLM Settings")
    
    # Pre-fill with environment variable if available
    api_key_default = OPENAI_API_KEY or ""
    api_key = st.text_input(
        "OpenAI API Key", 
        value=api_key_default,
        type="password",
        help="Enter your OpenAI API key or set it in the .env file"
    )
    
    model = st.selectbox(
        "Select LLM Model",
        ["gpt-4o", "gpt-4-turbo", "gpt-4"],
        index=0 if config['default_llm_model'] == "gpt-4o" else 1
    )
    
    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=config['default_llm_temperature'],
        step=0.1
    )
    
    # Context weighting
    st.subheader("Context Weighting")
    weight_internal = st.slider(
        "Weight of internal context factors",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    weight_external = 1.0 - weight_internal
    st.write(f"Weight of external context factors: {weight_external}")
    
    if st.button("Run Analysis"):
        if not api_key and not OPENAI_API_KEY:
            st.error("Please provide an OpenAI API key in the field above or in the .env file")
        else:
            # Add analysis logic here
            st.success("Analysis started! Please check the Results page when complete.")
            # Store analysis parameters in session state
            st.session_state.analysis_configured = True
            st.session_state.analysis_parameters = {
                "time_window": time_window,
                "api_key": api_key if api_key else OPENAI_API_KEY,
                "model": model,
                "temperature": temperature,
                "weight_internal": weight_internal,
                "weight_external": weight_external
            }

def render_results_page():
    st.header("Analysis Results")
    
    # Check if analysis has been run
    if "analysis_results" not in st.session_state:
        st.info("No analysis results available. Please run an analysis first.")
        return
    
    # Tabs for different result views
    tab1, tab2, tab3 = st.tabs(["Timeline View", "Detailed Analysis", "Export"])
    
    with tab1:
        st.subheader("Process Change Timeline with Context")
        # Timeline visualization code would go here
        st.write("Timeline visualization placeholder")
    
    with tab2:
        st.subheader("Detailed Change Point Analysis")
        # Detailed analysis view would go here
        st.write("Detailed analysis placeholder")
    
    with tab3:
        st.subheader("Export Results")
        export_format = st.selectbox(
            "Export format",
            ["PDF", "Excel", "JSON"]
        )
        
        if st.button("Export"):
            st.success(f"Results exported in {export_format} format")

def render_about_page():
    st.header("About")
    st.write('''
    # Context-Aware Process Drift Analysis
    
    This application bridges the gap between the detection of concept drifts in event logs
    and their practical interpretation by integrating external and internal context dimensions
    into process mining sense-making.
    
    ## How it works
    
    1. Upload change points detected by CV4CDD and context data
    2. Configure temporal analysis parameters and LLM settings
    3. AI analyzes correlations between process changes and contextual factors
    4. Review results showing which context factors likely influenced process changes
    
    ## About the project
    
    This tool was developed as part of a Master's thesis project focused on enhancing
    process mining with contextual understanding using large language models.
    ''')

if __name__ == "__main__":
    main()
""")
    
    # Create sidebar component if it doesn't exist
    components_dir = frontend_dir / "components"
    sidebar_file = components_dir / "sidebar.py"
    if not sidebar_file.exists():
        with open(sidebar_file, "w", encoding="utf-8") as f:
            f.write("""import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("Navigation")
        
        selected = st.radio(
            "Select a page",
            ["Upload Data", "Configure Analysis", "Results", "About"]
        )
        
        st.divider()
        
        st.write("Project Progress")
        progress = {
            "Data Uploaded": False,
            "Analysis Configured": False,
            "Analysis Complete": False
        }
        
        # Update progress based on session state
        if "change_points_uploaded" in st.session_state:
            progress["Data Uploaded"] = st.session_state.change_points_uploaded
        
        if "analysis_configured" in st.session_state:
            progress["Analysis Configured"] = st.session_state.analysis_configured
        
        if "analysis_complete" in st.session_state:
            progress["Analysis Complete"] = st.session_state.analysis_complete
        
        # Display progress
        for step, completed in progress.items():
            st.write(f"{'✅' if completed else '◻️'} {step}")
            
    return selected
""")
    
    # Create the change point parser if it doesn't exist
    core_dir = base_dir / "backend" / "core"
    parser_file = core_dir / "change_point_parser.py"
    if not parser_file.exists():
        with open(parser_file, "w") as f:
            f.write("""import json
import pandas as pd
from datetime import datetime

class ChangePointParser:
    \"\"\"
    Parser for CV4CDD change point output
    \"\"\"
    
    def __init__(self):
        self.change_points = None
        self.change_points_df = None
    
    def load_from_file(self, file_path):
        \"\"\"
        Load change points from JSON file
        \"\"\"
        with open(file_path, 'r') as file:
            self.change_points = json.load(file)
        
        # Convert to DataFrame for easier processing
        self.to_dataframe()
        return self.change_points
    
    def load_from_memory(self, json_data):
        \"\"\"
        Load change points from JSON data in memory
        \"\"\"
        self.change_points = json_data
        
        # Convert to DataFrame for easier processing
        self.to_dataframe()
        return self.change_points
    
    def to_dataframe(self):
        \"\"\"
        Convert change points to pandas DataFrame
        
        Note: This method needs to be adapted based on the actual structure
        of the CV4CDD output
        \"\"\"
        if not self.change_points:
            return None
        
        # This is a placeholder - actual implementation depends on CV4CDD output format
        change_points_list = []
        
        # Assuming change_points has a list of changes with timestamp and attributes
        for cp in self.change_points.get("change_points", []):
            change_points_list.append({
                "timestamp": cp.get("timestamp"),
                "process_id": cp.get("process_id"),
                "change_type": cp.get("change_type"),
                "confidence": cp.get("confidence"),
                "affected_attributes": cp.get("affected_attributes")
            })
        
        self.change_points_df = pd.DataFrame(change_points_list)
        
        # Convert timestamps to datetime objects
        if "timestamp" in self.change_points_df.columns:
            self.change_points_df["timestamp"] = pd.to_datetime(self.change_points_df["timestamp"])
        
        return self.change_points_df
    
    def get_time_range(self):
        \"\"\"
        Get the time range covered by the change points
        \"\"\"
        if self.change_points_df is None or "timestamp" not in self.change_points_df.columns:
            return None, None
            
        min_time = self.change_points_df["timestamp"].min()
        max_time = self.change_points_df["timestamp"].max()
        
        return min_time, max_time
    
    def filter_by_timeframe(self, start_date, end_date):
        \"\"\"
        Filter change points by timeframe
        \"\"\"
        if self.change_points_df is None:
            return None
            
        mask = (self.change_points_df["timestamp"] >= start_date) & \
               (self.change_points_df["timestamp"] <= end_date)
               
        return self.change_points_df[mask]
""")
    
    # Create the LLM integration module if it doesn't exist
    llm_file = core_dir / "llm_integration.py"
    if not llm_file.exists():
        with open(llm_file, "w") as f:
            f.write("""import openai
import json
import time
from typing import Dict, List, Any
from config.settings import OPENAI_API_KEY

class LLMAnalyzer:
    \"\"\"
    Integration with GPT-4o for context-aware analysis of change points
    \"\"\"
    
    def __init__(self, api_key=None, model="gpt-4o", temperature=0.0):
        \"\"\"
        Initialize the LLM analyzer
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key (defaults to env variable if not provided)
        model : str
            Model to use (default: gpt-4o)
        temperature : float
            Temperature parameter for the LLM (0.0-1.0)
        \"\"\"
        # Use the provided API key or fall back to the environment variable
        self.api_key = api_key or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY in .env or pass as parameter.")
            
        openai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
    
    def create_context_prompt(self, 
                             change_point: Dict[str, Any], 
                             internal_context: List[Dict[str, Any]], 
                             external_context: List[Dict[str, Any]],
                             event_log_sample: Dict[str, Any] = None) -> str:
        \"\"\"
        Create a prompt for the LLM to analyze the change point with context
        
        Parameters:
        -----------
        change_point : Dict
            Change point details
        internal_context : List[Dict]
            Internal context events around the change point timeframe
        external_context : List[Dict]
            External context events around the change point timeframe
        event_log_sample : Dict, optional
            Sample of the event log before and after the change point
            
        Returns:
        --------
        str
            Formatted prompt for the LLM
        \"\"\"
        
        # Format the timestamp for better readability
        timestamp = change_point.get("timestamp", "")
        if timestamp:
            try:
                # Format date if it's a datetime object or string
                if hasattr(timestamp, "strftime"):
                    formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Assuming ISO format string
                    from datetime import datetime
                    formatted_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_date = timestamp
        else:
            formatted_date = "Unknown date"
        
        # Build the prompt
        prompt = f\"\"\"You are an expert process mining analyst with deep knowledge of business processes and organizational change management. Your task is to analyze a detected change point in a process and explain its potential causes based on contextual information.

CHANGE POINT DETAILS:
- Timestamp: {formatted_date}
- Process ID: {change_point.get('process_id', 'Unknown')}
- Change Type: {change_point.get('change_type', 'Unknown')}
- Confidence: {change_point.get('confidence', 'Unknown')}
- Affected Attributes: {change_point.get('affected_attributes', [])}

INTERNAL CONTEXT EVENTS (organizational changes, system updates, etc.):
\"\"\"
        
        if internal_context:
            for idx, event in enumerate(internal_context, 1):
                prompt += f"{idx}. {event.get('date', 'Unknown date')}: {event.get('description', 'No description')}\\n"
                if event.get('details'):
                    prompt += f"   Details: {event['details']}\\n"
                prompt += f"   Source: {event.get('source', 'Unknown source')}\\n\\n"
        else:
            prompt += "No internal context events found in the specified timeframe.\\n\\n"
            
        prompt += "EXTERNAL CONTEXT EVENTS (regulations, market changes, etc.):\\n"
        
        if external_context:
            for idx, event in enumerate(external_context, 1):
                prompt += f"{idx}. {event.get('date', 'Unknown date')}: {event.get('description', 'No description')}\\n"
                if event.get('details'):
                    prompt += f"   Details: {event['details']}\\n"
                prompt += f"   Source: {event.get('source', 'Unknown source')}\\n\\n"
        else:
            prompt += "No external context events found in the specified timeframe.\\n\\n"
        
        if event_log_sample:
            prompt += "EVENT LOG SAMPLE:\\n"
            prompt += f"Before change point: {event_log_sample.get('before', 'No data')}\\n\\n"
            prompt += f"After change point: {event_log_sample.get('after', 'No data')}\\n\\n"
        
        prompt += \"\"\"ANALYSIS TASK:
1. Analyze the potential relationship between the context events and the detected change point.
2. Identify which context events (if any) are likely to have caused or influenced the process change.
3. Explain your reasoning and the potential mechanisms of influence.
4. Rate your confidence in the causal relationship (Low/Medium/High) and explain why.
5. If multiple factors could have contributed, rank them by likely importance.

YOUR ANALYSIS:\"\"\"

        return prompt
    
    def analyze_change_point(self, 
                            change_point: Dict[str, Any], 
                            internal_context: List[Dict[str, Any]], 
                            external_context: List[Dict[str, Any]],
                            event_log_sample: Dict[str, Any] = None) -> Dict[str, Any]:
        \"\"\"
        Use the LLM to analyze a change point with context
        
        Parameters as in create_context_prompt()
            
        Returns:
        --------
        Dict
            LLM analysis results
        \"\"\"
        
        # Create the prompt
        prompt = self.create_context_prompt(
            change_point, 
            internal_context, 
            external_context,
            event_log_sample
        )
        
        # Call the LLM API
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert process mining analyst specializing in concept drift analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            analysis_text = response.choices[0].message.content
            
            # Attempt to extract structured information from the analysis
            # Note: This is a simplistic approach; a more robust parsing could be implemented
            
            # Extract confidence rating
            confidence_level = "Unknown"
            if "confidence: high" in analysis_text.lower():
                confidence_level = "High"
            elif "confidence: medium" in analysis_text.lower():
                confidence_level = "Medium"
            elif "confidence: low" in analysis_text.lower():
                confidence_level = "Low"
            
            # Return the analysis results
            return {
                "change_point_id": change_point.get("id", ""),
                "change_point_timestamp": change_point.get("timestamp", ""),
                "analysis_text": analysis_text,
                "confidence_level": confidence_level,
                "prompt_used": prompt,
                "model_used": self.model,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            # Handle API errors
            return {
                "change_point_id": change_point.get("id", ""),
                "error": str(e),
                "prompt_used": prompt,
                "model_used": self.model,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
""")
    
    # Create a README.md file if it doesn't exist
    readme_file = base_dir / "README.md"
    if not readme_file.exists():
        with open(readme_file, "w") as f:
            f.write("""# Context-Aware Process Drift Analysis

This project bridges the gap between the detection of concept drifts in event logs and their practical interpretation by integrating external and internal context dimensions into process mining sense-making.

## Overview

The tool uses LLMs (specifically GPT-4o) to connect change points detected by CV4CDD with relevant contextual data, such as organizational charts and regulations, to provide meaningful explanations for process changes.

## Features

- Process CV4CDD JSON outputs with change points
- Connect change points with internal and external context data
- Analyze relationships between process changes and contextual factors using LLMs
- Provide timeline visualization of process changes with context
- Export analysis results in various formats

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\\Scripts\\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Run the application: `streamlit run frontend/app.py`

## Project Structure

- `backend/`: Backend code for data processing and analysis
- `frontend/`: Streamlit frontend code
- `data/`: Directory for raw, processed, and output data
- `config/`: Configuration settings
- `tests/`: Test files

""")
    
    print("Project files created successfully!")

if __name__ == "__main__":
    create_directory_structure()
    print("Project initialization complete!")