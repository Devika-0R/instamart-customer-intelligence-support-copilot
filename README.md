# Swiggy Instamart – Customer Intelligence & Support Copilot

## Project Overview

The **Customer Intelligence & Support Copilot** is an end-to-end data analytics and AI application built using synthetic Instamart-style grocery data.

The project combines data engineering, SQL analytics, customer segmentation, machine learning, privacy-safe support ticket classification, Retrieval-Augmented Generation (RAG), and an AI-powered refund agent.

All customer records, transactions, support tickets and policy documents used in this project are fictional.

## Technologies Used

- Python
- Pandas and NumPy
- MySQL / SQLite
- Scikit-learn
- Power BI concepts and data visualization
- TF-IDF and cosine similarity
- Jupyter Notebook
- JSON and JSONL

## 1. Data Foundation & Customer Intelligence (Parts A+B)

### Part A – Data Foundation

This section focuses on preparing reliable data for analysis.

**Key activities:**
- Generated synthetic customer, product, order and support-ticket datasets.
- Inspected data quality, missing values, duplicate records and inconsistent entries.
- Cleaned customer, product and order datasets.
- Standardized city and gender values.
- Handled missing and invalid values using documented cleaning rules.
- Stored cleaned datasets in CSV format.
- Created a SQLite database for structured data analysis.
- Performed SQL queries using filtering, joins, grouping, aggregation, window functions and ranking.
- Conducted exploratory data analysis (EDA) to understand sales, customer behaviour and product performance.

**Key results:**
- 1,200 customer records processed.
- 40 products available in the dataset.
- 12,309 unique orders after cleaning.
- 11,110 completed orders.
- Total completed-order revenue: ₹29,98,668.80.
- Bengaluru recorded the highest city-level revenue.

### Part B – Customer Segmentation & Lapse Prediction

This section focuses on understanding customer purchasing behaviour and identifying customers who may stop purchasing.

**Key activities:**
- Created Recency, Frequency and Monetary (RFM) features.
- Applied K-Means clustering to identify customer segments.
- Compared different cluster counts and examined PCA explained variance.
- Developed business-oriented segment names and recommendations.
- Built a customer lapse prediction model using Logistic Regression and Random Forest.
- Used preprocessing pipelines, feature scaling, encoding and GridSearchCV.
- Evaluated model performance using accuracy, precision, recall, F1-score and confusion matrices.
- Generated a retention list for Premium customers.

**Model results:**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 72.84% | 64.44% | 65.17% | 64.80% |
| Random Forest | 71.98% | 63.33% | 64.04% | 63.69% |

Logistic Regression was selected based on cross-validation F1-score.

The Premium retention list is illustrative because the Premium segment is derived from the full RFM observation period. It should not be interpreted as a leakage-free model evaluation.

## 2. Customer Support & AI Copilot (Parts C+D)

### Part C – Support Ticket Triage with Guardrails

This section develops a privacy-conscious support ticket classification and routing pipeline.

**Key activities:**
- Processed synthetic customer support tickets.
- Added input validation to skip empty tickets and flag unusually long tickets.
- Redacted email addresses, phone numbers and order IDs before model calls.
- Implemented zero-shot and few-shot classification prompts.
- Added retry handling with exponential backoff and jitter.
- Validated model responses using JSON parsing and keyword-based fallback validation.
- Routed tickets to appropriate support teams.
- Added token-budget controls to manage model usage.
- Sent invalid, long, empty and budget-exhausted tickets for human review.

**Classification results:**

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 7,372 | 12,520 |
| Tokens per classified ticket | 50.15 | 85.17 |

Few-shot prompting improved classification accuracy by approximately 14.29 percentage points, while consuming approximately 69.8% more tokens.

**Final ticket status:**

| Status | Tickets |
|---|---:|
| Classified JSON | 135 |
| Classified keyword | 12 |
| Flagged invalid output | 9 |
| Skipped empty | 4 |
| Flagged long | 4 |
| **Total** | **164** |

JSON and keyword classifications are automatically validated by programmatic rules. Invalid outputs and guardrail cases are reserved for human review rather than automatically guessed.

### Part D – Policy RAG Assistant & Refund Agent

This section combines policy retrieval with a bounded, database-backed agent workflow.

**Policy RAG Assistant**

- Loaded synthetic policy documents from text files.
- Split documents into overlapping text chunks.
- Converted chunks into TF-IDF vectors.
- Retrieved relevant policy passages using cosine similarity.
- Generated answers using the retrieved context.
- Included a fallback response when policy information cannot be found.
- Evaluated retrieval and answer results against expected policy documents and facts.

**Refund Agent**

- Connected to the SQLite order database.
- Implemented order lookup using customer order IDs.
- Added policy search functionality.
- Created a refund calculation tool.
- Implemented refund issuance and human escalation tools.
- Recorded refund and escalation actions in a JSONL ledger.
- Used a bounded agent loop with iteration, token and time limits.
- Saved execution checkpoints in JSON format.
- Included timeout and simulated crash handling.

**Example refund workflow:**

1. Retrieve the order from SQLite.
2. Search the relevant refund policy.
3. Calculate the eligible refund amount.
4. Issue the refund or escalate the case.
5. Record the action in the refund ledger.

A demonstration order was successfully processed with a refund amount of ₹128.25, using five LLM calls and 837 tokens.

The refund workflow is a simulation. The ledger records actions locally and does not perform real payment transactions.

## Project Structure

```text
instamart_customer_copilot/
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
│
├── src/
│   ├── data_pipeline.py
│   ├── customer_models.py
│   ├── triage.py
│   ├── policy_rag.py
│   └── refund_agent.py
│
├── notebooks/
│   └── Customer_Intelligence_Support_Copilot.ipynb
│
├── outputs/
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   └── refund_ledger.jsonl
│
├── state/
│   └── checkpoint JSON files
│
├── generate_data.py
├── mock_llm.py
├── requirements.txt
└── README.md
```

## Setup and Execution

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd instamart_customer_copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate synthetic data

```bash
python generate_data.py
```

### 5. Run the notebook

Open the notebook:

```text
notebooks/Customer_Intelligence_Support_Copilot.ipynb
```

Run the notebook cells in sequence.

The reusable business logic is maintained in the `src/` Python modules. The notebook imports and calls these functions to display results, perform evaluations and visualize outputs.

## Reproduce the Project

Run the following commands from the project root in a clean Python environment.

```bash
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate the synthetic datasets:

```bash
python generate_data.py
```

Run the project pipeline:

```bash
python run_project.py
```

The pipeline executes the data preparation, customer segmentation, lapse prediction, support ticket triage and policy retrieval stages.

For the complete interactive analysis and refund-agent demonstration, open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run its cells in order.

## Limitations

- All datasets and policy documents are synthetic.
- `mock_llm.py` provides an offline, deterministic simulation rather than a production language model.
- TF-IDF retrieval uses lexical similarity and may not capture every semantic relationship.
- The lapse model and customer segments are based on the available synthetic dataset.
- The refund ledger is a local audit record, not a real payment integration.
- A production deployment would require secure access controls, monitoring, robust validation, human approval workflows and integration with authorized business systems.

## Conclusion

This project demonstrates how data analytics, machine learning and generative AI techniques can work together to support customer intelligence and customer service operations.

It brings together data preparation, SQL analysis, customer segmentation, lapse prediction, privacy-aware ticket triage, policy-based question answering and automated refund workflows in one integrated application.