# Cosmic Market Oracle Project - Folder Structure

```
cosmic-market-oracle/
│
├── .github/                           # GitHub workflows and CI/CD configuration
│   ├── workflows/                     # GitHub Actions workflows
│   │   ├── ci.yml                     # Continuous integration
│   │   ├── deploy.yml                 # Deployment pipeline
│   │   └── gpu-tests.yml              # Specialized GPU testing workflow
│   └── ISSUE_TEMPLATE/               # Issue templates
│
├── config/                            # Configuration files
│   ├── cloud/                         # Cloud provider configurations
│   │   ├── vast_ai.yaml               # VAST.ai specific configuration
│   │   └── thunderstorm.yaml          # ThunderStorm specific configuration
│   ├── models/                        # Model configuration files
│   │   ├── foundation/                # Base model configs
│   │   ├── deep_learning/             # Deep learning model configs
│   │   ├── reinforcement/             # RL model configs
│   │   └── llm/                       # LLM integration configs
│   └── pipeline/                      # Data pipeline configurations
│       ├── data_ingestion.yaml        # Data acquisition configs
│       ├── feature_engineering.yaml   # Feature generation configs
│       └── airflow_dags/              # Apache Airflow DAG definitions
│
├── data/                              # Data storage (gitignored, managed separately)
│   ├── raw/                           # Raw input data
│   │   ├── financial/                 # Financial market data
│   │   │   ├── dji/                   # Dow Jones Industrial Average data
│   │   │   ├── constituents/          # Individual stock data
│   │   │   └── indicators/            # Market indicators
│   │   └── astrological/              # Astrological data
│   │       ├── ephemeris/             # Planetary position data
│   │       ├── aspects/               # Planetary aspect data
│   │       └── dashas/                # Dasha period data
│   ├── processed/                     # Processed, cleaned data
│   ├── features/                      # Engineered features
│   └── models/                        # Trained model weights and artifacts
│
├── docs/                              # Documentation
│   ├── architecture/                  # System architecture documentation
│   ├── data/                          # Data dictionaries and schemas
│   ├── models/                        # Model documentation
│   ├── api/                           # API documentation
│   ├── user_guides/                   # User guides and tutorials
│   └── research/                      # Research findings and papers
│
├── infrastructure/                    # Infrastructure as code
│   ├── docker/                        # Docker configurations
│   │   ├── base/                      # Base Docker images
│   │   ├── dev/                       # Development environment
│   │   ├── gpu/                       # GPU-specific containers
│   │   │   ├── training/              # Training-optimized containers
│   │   │   └── inference/             # Inference-optimized containers
│   │   └── monitoring/                # Monitoring containers
│   ├── terraform/                     # Terraform configurations
│   └── scripts/                       # Infrastructure management scripts
│       ├── gpu_management/            # GPU instance management
│       │   ├── vast_ai/               # VAST.ai specific scripts
│       │   └── thunderstorm/          # ThunderStorm specific scripts
│       ├── backup/                    # Backup and recovery scripts
│       └── monitoring/                # Resource monitoring scripts
│
├── notebooks/                         # Jupyter notebooks
│   ├── exploratory/                   # Exploratory data analysis
│   ├── feature_engineering/           # Feature engineering explorations
│   ├── model_prototyping/             # Model prototyping
│   ├── visualization/                 # Visualization notebooks
│   └── research/                      # Research experiments
│
├── src/                               # Source code
│   ├── data/                          # Data processing code
│   │   ├── acquisition/               # Data acquisition modules
│   │   │   ├── financial/             # Financial data acquisition
│   │   │   └── astrological/          # Astrological data acquisition
│   │   ├── processing/                # Data processing modules
│   │   └── validation/                # Data validation modules
│   │
│   ├── astro_engine/                  # Vedic astrological computation engine
│   │   ├── core/                      # Core astrological calculations
│   │   │   ├── ephemeris/             # Planetary position calculators
│   │   │   ├── nakshatras/            # Nakshatra calculations
│   │   │   └── dashas/                # Dasha system modules
│   │   ├── divisional_charts/         # Divisional chart (varga) calculators
│   │   ├── yogas/                     # Planetary combination detectors
│   │   └── financial/                 # Financial astrology specific modules
│   │
│   ├── features/                      # Feature engineering code
│   │   ├── financial/                 # Financial feature extractors
│   │   ├── astrological/              # Astrological feature extractors
│   │   ├── hybrid/                    # Combined feature generators
│   │   └── automated/                 # Automated feature discovery
│   │
│   ├── models/                        # Model implementation
│   │   ├── foundation/                # Foundation models
│   │   │   ├── gradient_boosting/     # Gradient boosting models
│   │   │   ├── ensemble/              # Ensemble methods
│   │   │   └── kernels/               # Specialized kernel methods
│   │   ├── deep_learning/             # Deep learning models
│   │   │   ├── time_series/           # Time series models
│   │   │   ├── transformers/          # Transformer architectures
│   │   │   └── neural_ode/            # Neural ODE models
│   │   ├── reinforcement/             # Reinforcement learning models
│   │   │   ├── policy/                # Policy optimization methods
│   │   │   ├── environments/          # Market simulation environments
│   │   │   └── rewards/               # Custom reward functions
│   │   └── llm/                       # Large language model integration
│   │       ├── knowledge_injection/   # Astrological knowledge integration
│   │       ├── reasoning/             # Specialized reasoning modules
│   │       └── explanation/           # Explanation generation
│   │
│   ├── agents/                        # Multi-agent system
│   │   ├── coordinator/               # Strategic coordination
│   │   ├── analyzers/                 # Specialized analysis agents
│   │   ├── executors/                 # Tactical execution agents
│   │   └── meta_learning/             # Meta-learning agents
│   │
│   ├── evaluation/                    # Evaluation and testing code
│   │   ├── backtesting/               # Backtesting framework
│   │   ├── metrics/                   # Performance metrics
│   │   ├── visualization/             # Result visualization
│   │   └── significance/              # Statistical significance testing
│   │
│   ├── api/                           # API implementation
│   │   ├── rest/                      # REST API endpoints
│   │   ├── graphql/                   # GraphQL API
│   │   └── websocket/                 # WebSocket API for real-time updates
│   │
│   ├── ui/                            # User interface code
│   │   ├── dashboard/                 # Dashboard components
│   │   ├── visualizations/            # Custom visualizations
│   │   ├── reports/                   # Report generation
│   │   └── alerts/                    # Alert system
│   │
│   ├── utils/                         # Utility functions and tools
│   │   ├── gpu/                       # GPU management utilities
│   │   ├── monitoring/                # System monitoring
│   │   ├── logging/                   # Logging configuration
│   │   └── profiling/                 # Performance profiling
│   │
│   └── pipelines/                     # End-to-end processing pipelines
│       ├── training/                  # Model training pipelines
│       ├── inference/                 # Inference pipelines
│       ├── research/                  # Research experiment pipelines
│       └── deployment/                # Model deployment pipelines
│
├── tests/                             # Test suite
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   ├── system/                        # End-to-end system tests
│   └── performance/                   # Performance benchmarks
│
├── tools/                             # Development and utility tools
│   ├── benchmarking/                  # GPU benchmarking tools
│   ├── data_generators/               # Synthetic data generators
│   ├── visualization/                 # Visualization tools
│   └── deployment/                    # Deployment utilities
│
├── scripts/                           # Operational scripts
│   ├── setup/                         # Environment setup scripts
│   ├── data_management/               # Data management scripts
│   ├── training/                      # Training orchestration scripts
│   └── deployment/                    # Deployment scripts
│
├── .gitignore                         # Git ignore configuration
├── .dockerignore                      # Docker ignore file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── README.md                          # Project overview
└── LICENSE                            # Project license
```

## Key Structure Highlights

### Core Components

1. **Vedic Astrological Engine** (`src/astro_engine/`):
   - Comprehensive calculation modules for planetary positions, nakshatras, dashas
   - Financial astrology specific components
   - Integration with Swiss Ephemeris and NASA JPL ephemeris

2. **Advanced Model Architecture** (`src/models/`):
   - Hierarchical organization of various model types
   - Specialized implementations for time series, cyclical data
   - Integration framework for multi-paradigm approaches

3. **Multi-Agent System** (`src/agents/`):
   - Strategic coordination and specialized analysis agents
   - Knowledge-sharing and consensus mechanisms
   - Meta-learning capabilities for system improvement

### Infrastructure Management

1. **Cloud GPU Management** (`infrastructure/scripts/gpu_management/`):
   - Provider-specific scripts for VAST.ai and ThunderStorm
   - Cost optimization and monitoring tools
   - Automated instance management

2. **Docker Environments** (`infrastructure/docker/`):
   - Specialized containers for different workloads
   - Development and production environments
   - GPU-optimized configurations

### Data Pipeline

1. **Data Acquisition** (`src/data/acquisition/`):
   - Financial market data collection from multiple sources
   - Astrological data generation and validation

2. **Feature Engineering** (`src/features/`):
   - Specialized extractors for financial and astrological features
   - Hybrid feature generators combining multiple domains
   - Automated feature discovery mechanisms

### Evaluation and Deployment

1. **Backtesting Framework** (`src/evaluation/backtesting/`):
   - Walk-forward testing with temporal causality preservation
   - Market regime-specific performance assessment
   - Statistical significance testing

2. **User Interface** (`src/ui/`):
   - Interactive dashboards for prediction visualization
   - Customizable reporting system
   - Alert mechanisms for significant configurations

This structure is designed to support the ambitious scope of the Cosmic Market Oracle project while providing clear organization for the complex integration of astrological data, financial markets, and advanced AI techniques.
