import os
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
