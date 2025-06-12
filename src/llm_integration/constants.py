# src/llm_integration/constants.py

"""Constants for the LLM integration package."""

# knowledge_base.py defaults
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_KNOWLEDGE_BASE_PATH = "data/knowledge_base"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# prompt_engineering.py defaults
DEFAULT_PROMPT_LIBRARY_PATH = "data/prompts"

# retrieval_augmentation.py defaults
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS = 150
DEFAULT_TEMPERATURE = 0.7

# output_formats.py defaults
DEFAULT_OUTPUT_DIR = "output" # General output, might conflict if other modules use "output"

# report_generation.py defaults
DEFAULT_REPORT_DIR = "reports" # Specific to LLM reports
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5  # seconds

# continuous_learning.py defaults
DEFAULT_UPDATE_INTERVAL = 24  # hours
DEFAULT_CL_OUTPUT_DIR = "updates" # Renamed to avoid clash with general DEFAULT_OUTPUT_DIR

# Note: DEFAULT_REPORT_TYPES from report_generation.py is a dict and will be handled separately if needed.
