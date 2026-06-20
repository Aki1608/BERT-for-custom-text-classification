import os
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from data_processor import prepare_data

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro") # Good for category imbalances
    return {"accuracy": acc, "f1": f1}

def main():
    MODEL_NAME = "bert-base-uncased"
    PROJECT_NAME = "bert-ticket-router"
    
    # 1. Fetch prepared data from data_processor
    tokenized_datasets, id2label, label2id, tokenizer = prepare_data(MODEL_NAME)
    
    # 2. Configure W&B tracking environment variables
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    wandb.init(project=PROJECT_NAME, name="bert-base-initial-run")
    
    # 3. Initialize the base model with classification heads sized for our labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Define training configurations
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb"
    )
    
    # 5. Execute Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,  # <--- UPDATED to support transformers v4.46+
        compute_metrics=compute_metrics
    )
    
    print("Launching Model Training Loop...")
    trainer.train()
    
    # Save optimized weights locally for our predictor engine to utilize
    print("Saving optimized fine-tuned model...")
    trainer.save_model("./final_ticket_router_model")
    
    wandb.finish()
    print("Training finished successfully.")

if __name__ == "__main__":
    main()