<div align="center">

<img src="https://img.shields.io/badge/IEEE-UCSC%20Student%20Branch-00629B?style=for-the-badge&logo=ieee&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Round%201-30%20May%202026-FF6F00?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Round%202-9%20June%202026-6A1B9A?style=for-the-badge"/>

<br/><br/>

```
███╗   ███╗██╗      ██████╗ ██████╗ ███████╗██╗██████╗ ██╗ █████╗ ███╗   ██╗
████╗ ████║██║     ██╔═══██╗██╔══██╗██╔════╝██║██╔══██╗██║██╔══██╗████╗  ██║
██╔████╔██║██║     ██║   ██║██████╔╝███████╗██║██║  ██║██║███████║██╔██╗ ██║
██║╚██╔╝██║██║     ██║   ██║██╔═══╝ ╚════██║██║██║  ██║██║██╔══██║██║╚██╗██║
██║ ╚═╝ ██║███████╗╚██████╔╝██║     ███████║██║██████╔╝██║██║  ██║██║ ╚████║
╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
```

### ⚡ Genesis — IEEE UCSC ML Competition 2026

*End-to-end tabular ML pipeline with automated MLOps deployment*

</div>

---

## 🧠 Team — OHIO ML GENESIS ARC

| Member | Role | Responsibility |
|--------|------|----------------|
| **Abdullah** | Core ML / Data Science | EDA, Feature Engineering, Model Training |
| **Lawsan** | Core ML / Data Science | XGBoost/LightGBM, Cross-Validation, Leaderboard Strategy |
| **Janagan** | MLOps / Model Serving | FastAPI, Docker, MLflow, Data Drift |
| **Hanan MD** | Infrastructure & Integration | GitHub Actions CI/CD, Cloud Deployment, Viva |

---

## 🏆 Competition Overview

> **ML Opsidian: Genesis** is a two-phase ML competition organized by the IEEE Student Branch at UCSC.  
> The real differentiator isn't just the model — it's building a **production-ready MLOps pipeline** around it.

| Phase | Date | Focus | Metric |
|-------|------|--------|--------|
| 🥇 Initial Round | 30 May 2026 | Build best ML model from scratch | Kaggle Leaderboard Score |
| 🚀 Final Round | 9 June 2026 | Wrap model in production MLOps pipeline | Deployment + Viva |

**Key Rules:**
- ❌ No pretrained models or external datasets in Round 1
- ✅ External tools & APIs allowed in Round 2 (with documentation)
- ✅ XGBoost, LightGBM, scikit-learn trained from scratch — all valid

---

## 📂 Project Structure

```
ml-opsidian-genesis/
│
├── 📁 data/
│   ├── raw/                  # Original competition dataset (git-ignored)
│   └── processed/            # Cleaned & feature-engineered data (git-ignored)
│
├── 📁 notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis
│   └── experiments.ipynb     # Model experimentation
│
├── 📁 src/
│   ├── features/             # Feature engineering scripts
│   ├── models/               # Model training & evaluation
│   └── utils/                # Helper functions
│
├── 📁 api/
│   └── main.py               # FastAPI application for model serving
│
├── 📁 models/
│   └── *.pkl                 # Saved model artifacts (git-ignored)
│
├── 📁 .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD pipeline
│
├── 📄 requirements.txt       # Project dependencies
├── 📄 Dockerfile             # Container definition
└── 📄 README.md
```

---

## ⚙️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-02A388?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?style=flat-square&logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Latest-2496ED?style=flat-square&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12.1-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat-square&logo=render&logoColor=white)

</div>

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/hananmd/ml-opsidian-genesis.git
cd ml-opsidian-genesis
```

### 2. Install Dependencies

> Install **only what your track needs** — the full `requirements.txt` is for the server/deployment.

```bash
# Person 1 & 2 — ML Track
pip install pandas numpy scikit-learn xgboost lightgbm

# Person 3 — MLOps Track
pip install fastapi uvicorn mlflow

# Full install (CI server / deployment)
pip install -r requirements.txt
```

### 3. Create Your Feature Branch

```bash
# Abdullah
git checkout -b feature/data-preprocessing

# Lawsan
git checkout -b feature/model-training

# Janagan
git checkout -b feature/docker-fastapi

# Hanan
git checkout -b feature/ci-cd-pipeline
```

> ⚠️ **Never push directly to `main`.** Always open a Pull Request and wait for review.

---

## 🔄 Git Workflow

```
main (protected — always stable)
  │
  ├── feature/data-preprocessing   ← Abdullah
  ├── feature/model-training        ← Lawsan
  ├── feature/docker-fastapi        ← Janagan
  └── feature/ci-cd-pipeline        ← Hanan
```

**Daily workflow:**
```bash
git checkout main && git pull origin main     # sync latest
git checkout feature/your-branch             # switch to your branch
# ... do your work ...
git add . && git commit -m "add: your message"
git push origin feature/your-branch          # push
# open Pull Request on GitHub
```

**Commit message prefixes:**
```
add:     new feature or file
fix:     bug fix
update:  modifying existing code
remove:  deleting something
docs:    README or comments
```

---

## 🤖 CI/CD Pipeline

Every push triggers the automated pipeline:

```
Push / PR to main
      │
      ▼
┌─────────────────────────┐
│  1. Checkout code        │
│  2. Setup Python 3.9     │
│  3. Cache dependencies   │
│  4. Install requirements │
│  5. Flake8 lint check    │
│  6. Import sanity check  │
└─────────────────────────┘
      │
      ▼
   ✅ Green → Ready to merge
   ❌ Red   → Fix before merging
```

---

## 📊 MLOps Pipeline (Final Round)

```
Raw Data
   │
   ▼
Feature Engineering (src/features/)
   │
   ▼
Model Training — XGBoost / LightGBM
   │
   ├──► MLflow Experiment Tracking
   │
   ▼
FastAPI Serving (api/main.py)
   │
   ▼
Docker Container
   │
   ▼
Render.com Deployment
   │
   ▼
Evidently AI — Data Drift Monitoring
```

---

## 📅 10-Day Timeline

| Days | Abdullah & Lawsan | Janagan | Hanan |
|------|-------------------|---------|-------|
| 1–2 | Kaggle Intro ML + Pandas | Docker basics | ✅ Git setup + CI/CD |
| 3–4 | Feature Engineering | FastAPI model serving | GitHub Actions |
| 5–6 | XGBoost/LightGBM + CV | MLflow tracking | Cloud deployment |
| 7–8 | Ensemble + submission | Data drift concepts | Pipeline architecture |
| 9–10 | Mock competition | Wire model into pipeline | Rehearse viva |

---

<div align="center">

**ML Opsidian: Genesis** · IEEE Student Branch UCSC · 2026

*Built with precision. Deployed with confidence.*

</div>