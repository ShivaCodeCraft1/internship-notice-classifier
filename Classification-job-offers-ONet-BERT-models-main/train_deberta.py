import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
)
import torch
import optuna
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
SAVE_DIR = "./experiments/"

# Load dataset
df_train = pd.read_csv("dataset_train.csv")
df_test = pd.read_csv("dataset_test.csv")

dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)

# Tokenizer (microsoft/deberta-base)
MODEL_NAME = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    return tokenizer(
        batch["description"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_test = dataset_test.map(tokenize_function, batched=True)

dataset_train = dataset_train.rename_column("Label", "label")
dataset_test = dataset_test.rename_column("Label", "label")

num_classes = df_train["Label"].nunique()
print(f"Number of classes: {num_classes}")

class_names = None

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }
    f1_per_class = f1_score(labels, predictions, average=None)
    for i, f1_val in enumerate(f1_per_class):
        label = class_names[i] if class_names else f"class_{i}"
        metrics[f"f1_{label}"] = f1_val
    return metrics

# Model (will be model-specific)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
)

def save_predictions_csv(trainer, dataset, class_names=None):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    texts = dataset["description"]

    df_result = pd.DataFrame(
        {
            "text": texts,
            "true_label": [class_names[i] if class_names else i for i in labels],
            "predicted_label": [class_names[i] if class_names else i for i in preds],
        }
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    df_result.to_csv(os.path.join(SAVE_DIR, f"predictions_{MODEL_NAME.replace('/', '_')}.csv"), index=False)

def train_model_with_hyperparams(learning_rate, batch_size, weight_decay, num_train_epochs):
    wandb.init(
        project="job_classification",
        config={
            "model": MODEL_NAME,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
        },
        name=f"{MODEL_NAME}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}_ep{num_train_epochs}",
        reinit=True,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(
            SAVE_DIR,
            f"{MODEL_NAME.replace('/', '_')}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}_ep{num_train_epochs}",
        ),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    results = trainer.evaluate()
    wandb.log(results)

    save_predictions_csv(trainer, dataset_test, class_names=class_names)
    return results["eval_f1"]

# Optuna search space
used_combinations = set()
def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", [2e-5, 3e-5, 5e-5])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.1])
    num_train_epochs = 10

    combination_key = (learning_rate, batch_size, weight_decay, num_train_epochs)
    if combination_key in used_combinations:
        raise optuna.exceptions.TrialPruned()
    used_combinations.add(combination_key)

    return train_model_with_hyperparams(learning_rate, batch_size, weight_decay, num_train_epochs)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    wandb.finish()
    print(f"Best hyperparameters for {MODEL_NAME}: {study.best_params}")

