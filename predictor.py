import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TicketPredictor:
    def __init__(self, model_path="./final_ticket_router_model"):
        print(f"Loading custom fine-tuned routing model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval() # Shift model to evaluation state to freeze dropouts
        
    def predict(self, ticket_text):
        if not ticket_text.strip():
            return "Please provide a valid ticket description.", {}
            
        # Tokenize user text string into model inputs
        inputs = self.tokenizer(
            ticket_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128, 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Convert outputs into raw relative probability distributions (0% to 100%)
            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze()
            
        # Extract the highest scoring class index
        predicted_class_id = torch.argmax(probabilities).item()
        predicted_label = self.model.config.id2label[predicted_class_id]
        
        # Build dictionary showing confidence values for top routing possibilities
        top_probs, top_indices = torch.topk(probabilities, k=5)
        confidences = {
            self.model.config.id2label[idx.item()]: float(prob) 
            for prob, idx in zip(top_probs, top_indices)
        }
        
        return predicted_label, confidences