from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(model_name="bert-base-uncased", max_length=128):
    print("Loading Banking77 dataset from Hugging Face Hub...")
    # Added trust_remote_code=True for older datasets
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    
    # Extract string labels and build mapping dictionaries
    labels = dataset["train"].features["label"].names
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    print("Tokenizing datasets in batches...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Clean up columns so the dataset contains only tensors required by PyTorch/BERT
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.with_format("torch")
    
    return tokenized_datasets, id2label, label2id, tokenizer