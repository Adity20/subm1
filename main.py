import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gdown
import os

# Function to download files from Google Drive
def download_from_google_drive(file_url, output_path):
    gdown.download(file_url, output_path, quiet=False)

# Load the BioBERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# Function to generate embeddings for a single input text
def generate_single_embedding(text, tokenizer, model):
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

# Load the dataset and embeddings
@st.cache_data
def load_data_and_embeddings():
    file_name = "./filtered_combined.xlsx"
    model_file = "./biobert_embeddings.pt"

    # Check if the files are available
    if not os.path.exists(file_name) or not os.path.exists(model_file):
        st.write("Downloading required files from Google Drive...")
        # Replace these with your actual Google Drive file URLs
        file_url = 'https://docs.google.com/spreadsheets/d/1TvKYsQ5ctKylFlV5KOzqdGkdFpFhDAZK/edit?usp=drive_link&ouid=117384216760735877574&rtpof=true&sd=true'
        model_file_url = 'https://drive.google.com/file/d/1W4UdgxLl7EnjSMvsBUBR4rZtsZQoLjyH/view?usp=drive_link'

        download_from_google_drive(file_url, file_name)
        download_from_google_drive(model_file_url, model_file)

    df = pd.read_excel(file_name)
    df["Combined_Text"] = df["Combined Column"].fillna("")
    embeddings = torch.load(model_file, map_location=device)
    return df, embeddings

# Function to get top N similar trials
def get_similar_trials(query_embedding, embeddings, top_n=10):
    query_embedding_cpu = query_embedding.cpu().detach().numpy()
    embeddings_cpu = embeddings.cpu().detach().numpy()
    similarities = cosine_similarity(query_embedding_cpu, embeddings_cpu)
    similar_indices = similarities.argsort(axis=1)[:, -top_n-1:-1][:, ::-1]
    return similar_indices, similarities

# Main Streamlit app
def main():
    st.title("Clinical Trials Similarity Finder")
    st.write("Find the most similar clinical trials using BioBERT embeddings.")

    # Input option
    option = st.radio(
        "Search by:",
        ("NCT ID", "Outcome or Criteria"),
        index=0,
        help="Choose how you want to search for similar trials."
    )

    # Input fields
    if option == "NCT ID":
        nct_id = st.text_input("Enter NCT ID:", placeholder="e.g., NCT00385736")
    else:
        criteria_text = st.text_area(
            "Enter Outcome or Criteria:",
            placeholder="e.g., A study evaluating the effects of drug X on Y patients..."
        )

    top_n = st.slider("Number of similar trials to retrieve:", min_value=1, max_value=20, value=10)

    # Process input and find similar trials
    if st.button("Find Similar Trials"):
        if option == "NCT ID" and nct_id:
            # Search by NCT ID
            nct_id_to_index = {nct: idx for idx, nct in enumerate(df["nct_id"])}
            if nct_id in nct_id_to_index:
                query_idx = nct_id_to_index[nct_id]
                query_embedding = embeddings[query_idx].unsqueeze(0).to(device)
            else:
                st.error(f"NCT ID {nct_id} not found in the dataset.")
                st.stop()
        elif option == "Outcome or Criteria" and criteria_text:
            # Search by text
            query_embedding = torch.tensor(generate_single_embedding(criteria_text, tokenizer, model)).to(device)
        else:
            st.error("Please provide a valid input.")
            st.stop()

        # Get similar trials
        similar_indices, similarities = get_similar_trials(query_embedding, embeddings, top_n=top_n)
        similar_trials = df.iloc[similar_indices[0]].copy()
        similar_trials["Similarity Score"] = [
            similarities[0, idx] for idx in similar_indices[0]
        ]

        # Display results
        st.write("### Top Similar Clinical Trials:")
        st.dataframe(similar_trials[["nct_id", "Study Title", "Similarity Score"]])

        # Download as Excel
        output_file = "similar_trials_results.xlsx"
        similar_trials.to_excel(output_file, index=False)
        with open(output_file, "rb") as f:
            st.download_button("Download Results as Excel", f, file_name="similar_trials_results.xlsx")

# Load resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_model_and_tokenizer()
df, embeddings = load_data_and_embeddings()

if __name__ == "__main__":
    main()
