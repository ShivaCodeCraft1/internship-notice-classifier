# Internship Notice Triage Classifier

An NLP-based text classification system that uses fine-tuned transformer models to automatically classify internship and job opportunities into relevant occupational categories.

## 📌 Problem Statement

Students receive internship and placement opportunities from multiple sources such as career offices, job portals, LinkedIn, and student groups. Manually reviewing every opportunity is time-consuming and makes it easy to miss relevant roles.

This project applies **Natural Language Processing (NLP)** and **transformer-based language models** to automatically analyze job descriptions and classify them into occupational categories.

The goal is to make internship and job-search workflows faster by automatically identifying the type of opportunity from its textual description.

## 🎯 Objectives

* Automatically classify job postings into occupational categories
* Apply transformer-based NLP models to real-world job descriptions
* Compare different BERT-based architectures
* Handle class imbalance through dataset balancing
* Optimize model hyperparameters
* Evaluate models using standard classification metrics
* Analyze model performance and classification errors

## 🤖 Models Used

Three transformer-based language models are evaluated:

* **BERT** — `bert-base-uncased`
* **RoBERTa** — `roberta-base`
* **DeBERTa** — `microsoft/deberta-base`

Each model is fine-tuned on the same job-posting dataset and evaluated using the same experimental setup.

## 🔄 Project Workflow

```text
Job Postings
     ↓
Data Collection
     ↓
Data Cleaning & Preprocessing
     ↓
Dataset Construction
     ↓
Class Balancing
     ↓
Train / Test Split
     ↓
Transformer Tokenization
     ↓
Fine-Tuning
     ↓
Hyperparameter Optimization
     ↓
Model Evaluation
     ↓
Error Analysis
```

## 📂 Project Structure

```text
├── list_job_scraper.py       # Collect job posting URLs
├── content_job_scraper.py    # Extract job posting details
├── build_dataset.py          # Build the unified dataset
├── balance_dataset.py        # Balance classes and split data
├── train_bert.py             # Fine-tune BERT
├── train_deberta.py          # Fine-tune DeBERTa
├── train_roberta.py          # Fine-tune RoBERTa
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* RoBERTa
* DeBERTa
* Optuna
* Pandas
* Scikit-learn
* Weights & Biases

## 📊 Dataset Preparation

The project builds a unified dataset from job postings collected from **CareerOneStop** and associates each posting with an occupational category based on **O*NET** classifications.

The dataset preparation pipeline:

1. Collect job posting URLs
2. Extract job titles, companies, descriptions, and URLs
3. Remove insufficient or low-quality descriptions
4. Construct a unified dataset
5. Encode occupational categories into numerical labels
6. Balance classes using undersampling
7. Perform a stratified train/test split

A minimum description length of **50 words** is used during dataset preparation.

## ⚖️ Class Balancing

Job-posting datasets naturally contain an unequal number of examples across occupational categories.

To reduce class imbalance, the project uses **undersampling**, reducing larger classes to the size of the minority class.

A **stratified train/test split** is then performed to preserve the class distribution.

## 🧠 Model Training

Each transformer model is fine-tuned using the Hugging Face training framework.

Hyperparameters are optimized using **Optuna**.

### Search Space

| Hyperparameter | Values                 |
| -------------- | ---------------------- |
| Learning Rate  | `2e-5`, `3e-5`, `5e-5` |
| Batch Size     | `8`, `16`, `32`        |
| Weight Decay   | `0.01`, `0.1`          |
| Epochs         | `10`                   |

The experiments are tracked using **Weights & Biases**.

## 📈 Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

F1-score is particularly useful for this task because it provides a balance between precision and recall when dealing with multiple occupational categories.

## 🔍 Error Analysis

Model predictions are analyzed to identify commonly confused occupational categories.

Examples of potential classification challenges include job descriptions that contain overlapping skills such as:

```text
Python + SQL + Machine Learning
```

which may contain language associated with multiple occupational categories.

Error analysis helps identify these cases and provides insights into where the model can be improved.

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

```bash
python build_dataset.py
```

### 3. Balance and Split the Dataset

```bash
python balance_dataset.py
```

### 4. Train a Model

```bash
# BERT
python train_bert.py

# RoBERTa
python train_roberta.py

# DeBERTa
python train_deberta.py
```

### 5. Weights & Biases

To track experiments:

```bash
wandb login
```

## 💡 Key Learning Outcomes

Through this project, the following concepts are explored:

* Text preprocessing
* NLP classification
* Transformer architectures
* BERT-based fine-tuning
* Tokenization
* Transfer learning
* Class imbalance
* Stratified data splitting
* Hyperparameter optimization
* Model evaluation
* Confusion matrix analysis
* Error analysis

## 🔮 Future Improvements

* Add a dedicated **internship relevance classifier**
* Add **SDE / Data Science / AI-ML / Core** role categories
* Add deadline and urgency detection
* Add personalized relevance scoring
* Build a simple web dashboard for internship triage
* Compare transformer models against a TF-IDF + Logistic Regression baseline
* Support classification of notices from multiple input sources

## 👨‍💻 Project Focus

This project focuses on applying **transformer-based NLP models to a practical recruitment and internship-search problem**, with emphasis on model comparison, fine-tuning, evaluation, and error analysis.
