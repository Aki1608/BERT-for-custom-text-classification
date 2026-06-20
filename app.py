import gradio as gr
from predictor import TicketPredictor

# Initialize the predictor instance globally
try:
    predictor = TicketPredictor("./final_ticket_router_model")
except Exception as e:
    print("Warning: Trained model directory not found yet. Please run train.py first.")
    predictor = None

def route_ticket(text):
    if predictor is None:
        return "System Error: The fine-tuned model has not been trained yet. Please run 'python train.py' in your console first.", {}
    
    category, confidence_scores = predictor.predict(text)
    return f"🏷️ **Assigned Routing Department:** `{category.upper()}`", confidence_scores

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎫 Smart Support Ticket Router")
    gr.Markdown("Input a raw customer query below to run it through your fine-tuned BERT model and discover the correct operational routing lane.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Ingest Ticket Details")
            ticket_input = gr.Textbox(
                label="Customer Communication Text", 
                placeholder="Example: I lost my credit card while traveling abroad and need an emergency replacement immediately...", 
                lines=5
            )
            submit_btn = gr.Button("Analyze & Route Intent", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Core Routing Determinations")
            routing_output = gr.Markdown(value="*Awaiting input analysis...*")
            confidence_chart = gr.Label(label="Top Prediction Confidences", num_top_classes=5)
            
    # Bind actions
    submit_btn.click(
        fn=route_ticket, 
        inputs=[ticket_input], 
        outputs=[routing_output, confidence_chart]
    )

if __name__ == "__main__":
    # Launch local server engine
    demo.launch(server_name="127.0.0.1", server_port=7860)