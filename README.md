# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

**IIT Jodhpur | PGD AI Programme | MLOps**

Fine-tunes `distilbert-base-cased` on UCSD Goodreads reviews to classify books into 7 genres.  
Experiment tracking with **Weights & Biases** and model hosting on **Hugging Face Hub**.

---

## Project Description

This project implements a complete MLOps workflow for text classification. A pre-trained DistilBERT model is fine-tuned on Goodreads book reviews from the UCSD Book Graph dataset to predict book genres (poetry, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult). The entire training pipeline is modularised into clean Python scripts, tracked with W&B, and the final model is published to the Hugging Face Hub.

---

## Repository Structure

```
├── data.py           # Data loading, sampling, train/test split, tokenizer encoding
├── train.py          # Model loading, Trainer setup, W&B logging, HF Hub push
├── eval.py           # Final evaluation, classification report, W&B Artifact upload
├── utils.py          # Label maps, MyDataset class, compute_metrics
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/g25ait2032-prog/nagaananth.git
cd nagaananth
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

### 4. Run training

```bash
python train.py
```

### 5. Run evaluation

```bash
python eval.py
```

> **GPU note:** Run on Google Colab (free tier) for a GPU. On CPU, reduce `reviews_per_genre` in `data.py` to `200` to keep training time manageable.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.6464   |
| F1 Score  | 0.6452   |
| Eval Loss | 0.9723   |



---

## Links

- **Hugging Face model:** https://huggingface.co/nagaananth/distilbert-goodreads-genres
- **W&B dashboard:** https://wandb.ai/g25ait2032-iit-jodhpur/mlops-assignment2
