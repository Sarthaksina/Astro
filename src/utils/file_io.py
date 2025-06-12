import os
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger # Assuming logger is desired

logger = get_logger(__name__)

def save_dataframe(df: pd.DataFrame, file_path_str: str, create_dirs: bool = True) -> bool:
    """
    Saves a DataFrame to disk.
    The format (CSV, Parquet, Pickle) is determined from the file_path_str extension.
    Defaults to Parquet if no recognized extension is provided or if the extension is missing.
    Logs a warning if defaulting to Parquet due to an unrecognized extension.

    Args:
        df: Pandas DataFrame to save.
        file_path_str: String representation of the full file path.
        create_dirs: If True, creates parent directories if they don't exist.

    Returns:
        True if successful, False otherwise.
    """
    original_file_path_str = file_path_str # For logging purposes
    file_path = Path(file_path_str)

    if create_dirs:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {file_path.parent}: {e}")
            return False

    try:
        if file_path.suffix == '.csv':
            df.to_csv(file_path, index=(isinstance(df.index, pd.MultiIndex))) # Handle MultiIndex appropriately
            logger.info(f"DataFrame saved to {file_path}")
            return True
        elif file_path.suffix == '.parquet':
            df.to_parquet(file_path)
            logger.info(f"DataFrame saved to {file_path}")
            return True
        elif file_path.suffix in ['.pickle', '.pkl']:
            df.to_pickle(file_path)
            logger.info(f"DataFrame saved to {file_path}")
            return True
        else:
            if file_path.suffix: # If there was an extension, but it was unrecognized
                logger.warning(f"Unrecognized extension '{file_path.suffix}' for {original_file_path_str}. Defaulting to Parquet format.")
            else: # No extension provided
                logger.info(f"No extension for {original_file_path_str}. Defaulting to Parquet format.")

            # Ensure the new path has .parquet extension
            file_path_parquet = file_path.with_suffix('.parquet')
            df.to_parquet(file_path_parquet)
            logger.info(f"DataFrame saved to {file_path_parquet}")
            return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path_str} (or its .parquet version): {e}")
        return False
