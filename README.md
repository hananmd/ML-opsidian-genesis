# ML Opsidian: Genesis

## Team [OHIO ML GENESIS ARC]
- **ABDULLAH & LAWSAN:** Core ML / Data Science (Kaggle Leaderboard Focus)
- **JANAGAN:** MLOps / Model Serving (FastAPI + Docker)
- **HANAN MD:** Infrastructure / CI/CD / Integration (GitHub Actions + Render)

## Project Overview
This repository contains the end-to-end MLOps pipeline for the IEEE UCSC ML Opsidian: Genesis competition. 
We are building a robust tabular data classifier using XGBoost/LightGBM, serving it via FastAPI, and deploying it with automated CI/CD.

## 📂 Project Structure
- `data/`: Raw and processed datasets (ignored by Git).
- `notebooks/`: EDA and experimental modeling (Jupyter Notebooks).
- `src/`: Modular Python code for features and models.
- `api/`: FastAPI application for model serving.
- `.github/workflows/`: CI/CD pipelines for automated testing.

## 🛠️ Setup Instructions
1. Clone the repo:
   ```bash
   git clone https://github.com/hananmd/ml-opsidian-genesis.git
   cd ml-opsidian-genesis