import sys
from src.utils.logger import get_logger # Using the centralized logger

logger = get_logger(__name__)

def check_python_version(required_major: int, required_minor: int) -> bool:
    """
    Checks if the current Python version meets the specified major and minor versions.
    Logs an error and returns False if not, or logs success and returns True if it meets requirements.
    """
    current_major = sys.version_info.major
    current_minor = sys.version_info.minor

    if current_major < required_major or \
       (current_major == required_major and current_minor < required_minor):
        error_msg = f"Python {required_major}.{required_minor}+ is required. You are using {current_major}.{current_minor}"
        logger.error(error_msg)
        print(f"Error: {error_msg}") # Also print to console for script visibility
        return False

    success_msg = f"Python version check passed: Running {current_major}.{current_minor} (meets requirement >= {required_major}.{required_minor})"
    logger.info(success_msg)
    # print(f"Info: {success_msg}") # Optional: print success to console
    return True

# Optional: get_project_root() can be added later if deemed necessary.
# from pathlib import Path
# def get_project_root() -> Path:
#     """Returns the project root directory by assuming utils is one level down from src, and src is in root."""
#     return Path(__file__).parent.parent.parent.resolve()
