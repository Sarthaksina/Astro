@echo off
REM Run Cosmic Market Oracle with Python 3.10
REM Usage: run_with_python310 <command> [arguments]

echo Running Cosmic Market Oracle with Python 3.10.0...

IF "%1"=="" (
    echo Available commands:
    echo   data     - Run data acquisition
    echo   optimize - Run hyperparameter optimization
    echo   predict  - Run prediction pipeline
    echo   mlflow   - Run MLflow UI
    echo.
    echo Example: %0 predict --start_date 2023-01-01 --end_date 2023-12-31
    exit /b
)

py -3.10 run_cosmic_oracle.py %*
