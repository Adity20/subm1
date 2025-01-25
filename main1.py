from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gdown
import os

app = Flask(__name__)

# Load the BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Function to generate embeddings for a single input text
def generate_single_embedding(text):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze(0).to(device) for key, val in encoding.items()}
        output = model(**encoding)
        return output.last_hidden_state[:, 0, :].cpu().numpy()

# Function to download files from Google Drive
def download_from_google_drive(url, output):
    gdown.download(url, output, quiet=False)

# Load data and embeddings
file_name = "./filtered_combined.xlsx"
model_file = "./biobert_embeddings.pt"

if not os.path.exists(file_name) or not os.path.exists(model_file):
    file_url = 'https://drive.google.com/uc?id=1TvKYsQ5ctKylFlV5KOzqdGkdFpFhDAZK'
    model_file_url = 'https://drive.google.com/uc?id=1W4UdgxLl7EnjSMvsBUBR4rZtsZQoLjyH'
    download_from_google_drive(file_url, file_name)
    download_from_google_drive(model_file_url, model_file)

# Load dataset and embeddings
df = pd.read_excel(file_name, engine="openpyxl")
df["Combined_Text"] = df["Combined Column"].fillna("")
embeddings = torch.load(model_file, map_location="cpu")

# Function to get top N similar trials
def get_similar_trials(query_embedding, embeddings, top_n=10):
    query_embedding_cpu = query_embedding.cpu().detach().numpy()
    embeddings_cpu = embeddings.cpu().detach().numpy()
    similarities = cosine_similarity(query_embedding_cpu, embeddings_cpu)
    similar_indices = similarities.argsort(axis=1)[:, -top_n-1:-1][:, ::-1]
    return similar_indices, similarities

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")  # Create an HTML form for input

@app.route("/find_similar_trials", methods=["POST"])
def find_similar_trials():
    search_type = request.form.get("search_type")
    input_text = request.form.get("input_text")
    top_n = int(request.form.get("top_n", 10))

    if search_type == "NCT_ID":
        nct_id = input_text.strip()
        nct_id_to_index = {nct: idx for idx, nct in enumerate(df["nct_id"])}
        if nct_id in nct_id_to_index:
            query_idx = nct_id_to_index[nct_id]
            query_embedding = embeddings[query_idx].unsqueeze(0).to(device)
        else:
            return jsonify({"error": f"NCT ID {nct_id} not found in the dataset."})
    else:
        query_embedding = torch.tensor(generate_single_embedding(input_text)).to(device)

    # Get similar trials
    similar_indices, similarities = get_similar_trials(query_embedding, embeddings, top_n=top_n)
    similar_trials = df.iloc[similar_indices[0]].copy()
    similar_trials["Similarity Score"] = [
        similarities[0, idx] for idx in similar_indices[0]
    ]

    # Save results to an Excel file
    output_file = "similar_trials_results.xlsx"
    similar_trials.to_excel(output_file, index=False)

    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
