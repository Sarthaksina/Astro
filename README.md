# The Cosmic Market Oracle

An AI-powered financial forecasting system that integrates Vedic astrological principles with 200 years of market data to predict market inflection points.

## Project Overview

This pioneering project fuses historical Dow Jones Industrial Average (DJI) data with Vedic astrological positions through advanced AI techniques to create a revolutionary market prediction system. By analyzing cosmic patterns that potentially influence market psychology and economic cycles, we aim to develop a robust system capable of identifying market inflection points with unprecedented accuracy.

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with TimescaleDB extension (for time-series data storage)
- GPU access for model training (local or cloud-based)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sarthaksina/Astro.git
   cd Astro
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Project Structure

```
├── src/                             # Source code
│   ├── astro_engine/                # Vedic astrological calculation engine
│   ├── data_acquisition/            # Data collection modules
│   ├── data_processing/             # Data cleaning and transformation
│   ├── feature_engineering/         # Feature creation from raw data
│   ├── models/                      # ML/DL model implementations
│   ├── api/                         # FastAPI implementation
│   └── visualization/               # Data visualization tools
│
├── tests/                           # Unit and integration tests
├── notebooks/                       # Jupyter notebooks for exploration
├── config/                          # Configuration files
├── scripts/                         # Operational scripts
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation
└── README.md                        # Project documentation
```

## Core Components

1. **Vedic Astrological Engine**: Comprehensive calculation modules for planetary positions, nakshatras, and dashas
2. **Data Acquisition System**: Collection of historical market data and astrological ephemeris
3. **Feature Engineering**: Conversion of astrological phenomena into ML-compatible inputs
4. **Model Development**: Implementation of ensemble models combining multiple AI paradigms
5. **Evaluation Framework**: Comprehensive testing across different market regimes

## Development Workflow

1. Set up the development environment as described above
2. Refer to `TASK.md` for current development priorities
3. Follow the architectural guidelines in `PLANNING.md`
4. Create unit tests for all new functionality
5. Document code with Google-style docstrings

## License

This project is proprietary and confidential. All rights reserved.

## Acknowledgments

- Swiss Ephemeris for high-precision astronomical calculations
- TimescaleDB for efficient time-series data storage
- VAST.ai and ThunderStorm for cloud GPU infrastructure