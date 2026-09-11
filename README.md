# 🏠 End-to-End House Price Prediction

An end-to-end machine learning project for predicting house prices using the **Ames Housing dataset**. Built with clean software engineering principles, featuring a modular architecture powered by **Strategy**, **Factory**, and **Template Method** design patterns, and orchestrated end to end with **ZenML** and **MLflow**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture & Design Patterns](#architecture--design-patterns)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Modules](#modules)
  - [Source Modules (`src/`)](#source-modules-src)
  - [Analysis Modules (`analysis/`)](#analysis-modules-analysis)
  - [Pipeline & Steps](#pipeline--steps)
- [Usage](#usage)
- [Results](#results)
- [Project Status](#project-status)
- [License](#license)

---

## Overview

This project predicts residential property sale prices in Ames, Iowa using a **Linear Regression** model built on top of scikit-learn. The entire ML workflow — from raw data ingestion through model evaluation — is implemented as modular, swappable components that can be orchestrated via ZenML pipelines.

**Key highlights:**
- 🧩 **Modular design** — every processing step is strategy-based and interchangeable at runtime
- 📊 **Comprehensive EDA** — Jupyter notebook with reusable analysis classes
- 🔧 **Production-ready patterns** — Factory, Strategy, and Template Method design patterns
- 📈 **Experiment tracking** — MLflow integration via ZenML experiment tracker
- 🚀 **Pipeline orchestration** — ZenML training, deployment, and inference pipelines

---

## Architecture & Design Patterns

The codebase consistently applies clean design patterns:

### Strategy Pattern
Every processing step follows the same structure:
1. **Abstract base class** — defines the interface (e.g., `MissingValueHandlingStrategy`)
2. **Concrete strategies** — implement specific algorithms (e.g., `DropMissingValuesStrategy`, `FillMissingValuesStrategy`)
3. **Context class** — delegates to the chosen strategy (e.g., `MissingValueHandler`)

### Factory Pattern
`DataIngestorFactory` in `ingest_data.py` returns the correct data ingestor based on file extension.

### Template Method Pattern
Analysis classes like `MultiVariateAnalysisTemplate` and `MissingValueAnalysisTemplate` define skeleton algorithms with abstract sub-steps.

---

## Project Structure

```
End_to_End_Price_Prediction/
├── run_pipeline.py                      # CLI: run the training pipeline
├── run_deployment.py                    # CLI: deploy + inference (--stop-service to tear down)
├── pyproject.toml                       # Project config & dependencies (uv)
├── requirements.txt                     # Pip dependencies
├── .python-version                      # Python version pin (3.10)
│
├── data/
│   └── archive.zip                      # Raw dataset (zipped)
│
├── extracted_data/
│   └── AmesHousing.csv                  # Extracted CSV (~2930 rows, 82 columns)
│
├── analysis/
│   ├── EDA.ipynb                        # Exploratory Data Analysis notebook
│   └── analyze_src/                     # Reusable analysis classes
│       ├── basic_data_inspection.py     # Data types & summary statistics
│       ├── univariate_analysis.py       # Histograms, KDE, count plots
│       ├── bivariate_analysis.py        # Scatter plots, box plots
│       ├── multivariate_analysis.py     # Correlation heatmaps, pair plots
│       └── missing_values_analysis.py   # Missing value counts & heatmaps
│
├── src/                                 # Core ML logic
│   ├── ingest_data.py                   # Data ingestion from zip files
│   ├── handle_missing_value.py          # Missing value handling strategies
│   ├── outlier_detection.py             # Outlier detection (Z-Score, IQR)
│   ├── feature_engineering.py           # Feature transformations & encoding
│   ├── data_splitter.py                 # Train/test splitting
│   ├── model_building.py               # Model training (Linear Regression)
│   └── model_evaluator.py              # Model evaluation (MSE, R²)
│
├── steps/                               # ZenML step wrappers
│   ├── data_ingestion_step.py
│   ├── handle_missing_values_step.py
│   ├── outlier_detection_step.py
│   ├── feature_engineering_step.py
│   ├── data_splitter_step.py
│   ├── model_building_step.py
│   ├── model_evaluator_step.py
│   ├── dynamic_importer.py
│   ├── model_loader.py
│   ├── prediction_service_loader.py
│   └── predictor.py
│
└── pipelines/                           # ZenML pipeline definitions
    ├── training_pipeline.py             # ingest → clean → split → train → evaluate
    └── deployment_pipeline.py           # continuous deployment + inference pipelines
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python ≥ 3.10 |
| Package Manager | [uv](https://github.com/astral-sh/uv) |
| ML Framework | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Statistics | statsmodels |
| Pipeline Orchestration | ZenML |
| Experiment Tracking | MLflow |
| CLI | click |

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)
```bash
# Clone the repository
git clone https://github.com/<your-username>/End_to_End_Price_Prediction.git
cd End_to_End_Price_Prediction

# Install dependencies
uv sync
```

### Using pip
```bash
# Clone the repository
git clone https://github.com/<your-username>/End_to_End_Price_Prediction.git
cd End_to_End_Price_Prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

| Detail | Value |
|--------|-------|
| **Name** | Ames Housing Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/marcopale/housing) |
| **Rows** | 2,930 residential home sales |
| **Features** | ~82 columns (numerical + categorical) |
| **Target** | `SalePrice` — property sale price in USD |
| **Location** | Ames, Iowa |

The raw data is stored as `data/archive.zip` and extracted to `extracted_data/AmesHousing.csv` during the ingestion step.

---

## Pipeline

The ML pipeline follows this data flow:

```
📦 archive.zip
    ↓
📊 Data Ingestion (extract zip → load CSV)
    ↓
🔍 Missing Value Handling (drop / fill with mean, median, mode, constant)
    ↓
📈 Outlier Detection & Handling (Z-Score → cap at 1st/99th percentile)
    ↓
⚙️  Feature Engineering (log transform on Gr Liv Area)
    ↓
✂️  Train/Test Split (numeric features only, 75/25)
    ↓
🏗️ Model Building (StandardScaler + Linear Regression pipeline)
    ↓
📏 Model Evaluation (Mean Squared Error, R² Score)
    ↓
🚀 Deployment (MLflow model server) → batch Inference (sample data)
```

---

## Modules

### Source Modules (`src/`)

#### `ingest_data.py` — Data Ingestion
- `ZipDataIngestor` — extracts `.zip` files, validates a single CSV exists, returns a DataFrame
- `DataIngestorFactory` — factory that returns the correct ingestor by file extension

#### `handle_missing_value.py` — Missing Value Handling
- `DropMissingValuesStrategy` — drops rows/columns based on axis and threshold
- `FillMissingValuesStrategy` — fills with mean, median, mode, or a constant value

#### `outlier_detection.py` — Outlier Detection
- `ZScoreOutlierDetection` — flags values where |z-score| exceeds threshold (default: 3)
- `IQROutlierDetection` — flags values outside Q1 − 1.5×IQR to Q3 + 1.5×IQR
- `OutlierDetector` — supports remove or cap handling (pipeline default: cap at 1st/99th percentile, preserving all rows), plus boxplot visualization

#### `feature_engineering.py` — Feature Engineering
- `LogTransformation` — applies `log(1+x)` to reduce skewness
- `StandardScaling` — standardizes to mean=0, std=1
- `MinMaxScaling` — rescales to [0, 1] or custom range
- `OneHotEncoding` — converts categorical features to binary vectors

#### `data_splitter.py` — Data Splitting
- `SimpleTrainTestSplit` — sklearn's `train_test_split` with configurable test size and random state; the ZenML step selects numeric features only before splitting

#### `model_building.py` — Model Training
- `LinearRegressionStrategy` — builds an sklearn `Pipeline` with `StandardScaler` → `LinearRegression`

#### `model_evaluator.py` — Model Evaluation
- `RegressionModelEvaluationStrategy` — computes Mean Squared Error (MSE) and R² Score

---

### Analysis Modules (`analysis/`)

| Module | Purpose |
|--------|---------|
| `EDA.ipynb` | Full exploratory data analysis notebook |
| `basic_data_inspection.py` | Data types info and descriptive statistics |
| `univariate_analysis.py` | Distribution plots (histograms + KDE, count plots) |
| `bivariate_analysis.py` | Relationship plots (scatter plots, box plots) |
| `multivariate_analysis.py` | Correlation heatmaps and pair plots |
| `missing_values_analysis.py` | Missing value identification and heatmap visualization |

---

### Pipeline & Steps

The `steps/` directory contains ZenML step wrappers around the `src/` strategies, and `pipelines/` defines:

- **`ml_pipeline`** (`training_pipeline.py`) — ingest → missing values → outliers → feature engineering → split → train → evaluate; registered as the ZenML model `prices_predictor`
- **`continuous_deployment_pipeline`** (`deployment_pipeline.py`) — runs `ml_pipeline`, then (re)deploys the trained model using ZenML's built-in `mlflow_model_deployer_step`
- **`inference_pipeline`** (`deployment_pipeline.py`) — loads sample data (`dynamic_importer`), fetches the live MLflow prediction service (`prediction_service_loader`), and runs batch predictions against it (`predictor`)

This provides:

- **Reproducible pipeline runs** — every step tracked, versioned, and cached
- **Experiment tracking** — parameters, metrics, and model artifacts logged to MLflow
- **Model serving** — prediction service via MLflow deployment

---

## Usage

### One-time setup

```bash
uv sync          # install all dependencies (includes zenml[local])

# Initialize the ZenML workspace and register the MLflow stack
zenml init
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register local_mlflow -e mlflow_tracker -d mlflow_deployer -a default -o default --set
```

### Running the pipelines

```powershell
# Required on every new terminal on Windows (console defaults to cp1252; ZenML needs UTF-8)
$env:PYTHONUTF8='1'

# Training only: ingest → clean → split → train → evaluate (~10s)
uv run python run_pipeline.py

# Full deployment: train, deploy the model via MLflow, run batch inference
uv run python run_deployment.py

# Stop the prediction service when done
uv run python run_deployment.py --stop-service
```

### Inspecting results

```bash
# MLflow UI — run the exact command printed at the end of run_pipeline.py, e.g.:
mlflow ui --backend-store-uri "sqlite:///C:\Users\<user>\AppData\Roaming\zenml\local_stores\<id>\mlflow.db"
# then open http://127.0.0.1:5000

# ZenML dashboard (pipelines, runs, stacks)
zenml login --local
```

### Running the EDA Notebook

```bash
jupyter notebook analysis/EDA.ipynb
```

> **Windows note:** MLflow model *serving* (the daemon-based prediction server used by `run_deployment.py`) is not supported on native Windows. Training and experiment tracking work fully; for the live prediction service, run under WSL.

---

## Results

Current `prices_predictor` model (Linear Regression on numeric features, trained via `run_pipeline.py`):

| Metric | Value |
|--------|-------|
| Mean Squared Error | ~8.0 × 10⁸ |
| R² Score | 0.875 |

---

## Project Status

| Component | Status |
|-----------|--------|
| Core ML Logic (`src/`) | ✅ Complete |
| EDA & Analysis | ✅ Complete |
| Design Patterns | ✅ Implemented |
| ZenML Steps (`steps/`) | ✅ Complete |
| ZenML Pipeline (`pipelines/`) | ✅ Complete |
| MLflow Integration | ✅ Complete (tracking verified; serving limited on Windows) |
| Documentation | ✅ Complete |

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.