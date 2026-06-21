# Smart Support Ticket Router (BERT Fine-Tuning)

An enterprise-grade NLP pipeline that fine-tunes a Transformer model (`bert-base-uncased`) to automatically categorize and route customer support tickets. This project uses the Hugging Face Trainer API for optimized training, **Weights & Biases (W&B)** for real-time experiment tracking, and **Gradio** for a clean, interactive user interface.

Trained on the `PolyAI/banking77` dataset, this model can accurately classify incoming customer queries into one of 77 specific banking operational departments (e.g., lost cards, failed transfers, pin resets).

---

## Core Features

* **Custom BERT Fine-Tuning:** Specializes a pre-trained Google BERT model to understand domain-specific customer service language.
* **Live Experiment Tracking:** Fully integrated with Weights & Biases (W&B) to monitor training loss, validation accuracy, F1-scores, and resource utilization in real-time.
* **Robust Data Processing:** Safely loads, tokenizes, and maps integer-to-string labels using the Hugging Face datasets library.
* **Automated Checkpointing:** Automatically saves the best-performing model weights during the evaluation phase.
* **Interactive UI:** Features a Gradio-based web dashboard allowing users to input fake support tickets and see real-time routing predictions and confidence scores.

---

## Project Structure

* `data_processor.py`: Orchestrates dataset downloading, label extraction (ID to string mapping), and text tokenization.
* `train.py`: The core training engine. Initializes the model, sets hyperparameters, connects to W&B, and runs the Hugging Face Trainer loop.
* `predictor.py`: The inference backend. Loads the locally saved fine-tuned model and processes raw string inputs into mathematical category predictions.
* `app.py`: The Gradio web interface connecting user inputs to the inference backend.
* `requirements.txt`: The lean, unpinned dependency manifest (with a specific datasets version pinned to support legacy Hugging Face scripts).

---

## Prerequisites

**1. Python Version:** This project requires **Python 3.10, 3.11, or 3.12**.

**2. Weights & Biases Account:**
You will need a free account at wandb.ai to track the training experiments.

---

## Installation & Setup

**1. Clone the repository:**

    git clone https://github.com/yourusername/BERT-for-custom-text-classification.git
    cd BERT-for-custom-text-classification

**2. Create and activate a virtual environment:**

* **Windows (PowerShell):**

    python -m venv venv
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
    .\venv\Scripts\activate

* **Linux / macOS / GitHub Codespaces:**

    python -m venv venv
    source venv/bin/activate

**3. Install dependencies:**

    pip install --upgrade pip
    pip install -r requirements.txt

**4. Set up Environment Variables:**
Create a `.env` file in the root directory and add your Weights & Biases API key:

    WANDB_API_KEY=your_wandb_api_key_here

---

## Running the Pipeline

### Step 1: Train the Model
Before you can run the UI, you must train the BERT model. Make sure your virtual environment is active, then run:

    python train.py

*Note: This will output a link in your terminal to your live W&B dashboard. Once training finishes (typically 10-20 minutes depending on your hardware), a new folder named `final_ticket_router_model` will be generated.*

### Step 2: Launch the Web UI
Once the model is fully trained and saved locally, start the Gradio interface:

    python app.py

Open the provided `localhost` URL (usually http://127.0.0.1:7860) in your browser to start routing fake customer support tickets!

---
