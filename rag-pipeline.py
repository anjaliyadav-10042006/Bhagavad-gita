import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import json

# Load Data Function
def load_data():
    # Replace with your actual file paths
    gita_file = ("C:/Users/lovek/OneDrive/Desktop/RAG uning/Bhagwad_Gita.csv")
    pys_file = ("C:/Users/lovek/OneDrive/Desktop/RAG uning/Patanjali_Yoga_Sutras_Verses_English_Questions.csv")
    # Load the data using pandas, 
    gita = pd.read_csv("C:/Users/lovek/OneDrive/Desktop/RAG uning/Bhagwad_Gita.csv")
    pys = pd.read_csv("C:/Users/lovek/OneDrive/Desktop/RAG uning/Patanjali_Yoga_Sutras_Verses_English_Questions.csv")

    # Print columns to verify correctness
    print("Columns in Gita file:", gita.columns)
    print("Columns in PYS file:", pys.columns)

    # Adjust the column names below as per your file
    gita_texts = gita["translation"].tolist()  # Replace 'Text' with your actual Gita column name
    pys_texts = pys["translation"].tolist()    # Replace 'Text' with your actual PYS column name

    return gita_texts, pys_texts

# Create Embeddings and Build FAISS Index
def create_faiss_index(texts, model_name="all-MiniLM-L6-v2"):
    # Load the Sentence Transformer model
    model = SentenceTransformer(model_name)

    # Create embeddings for the texts
    embeddings = model.encode(texts, convert_to_tensor=False)

    # Initialize FAISS Index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return model, index, embeddings

# Retrieve Shlokas Function
def retrieve_shlokas(query, texts, model, index, top_k=3):
    # Generate query embedding
    query_embedding = model.encode([query], convert_to_tensor=False)

    # Search for the top_k most similar texts
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve and return the top results
    results = [{"text": texts[idx], "score": float(distances[0][i])} for i, idx in enumerate(indices[0])]
    return results

# Main Function
def main():
    # Load data
    gita_texts, pys_texts = load_data()

    # Combine both datasets for a unified index
    all_texts = gita_texts + pys_texts

    # Create embeddings and FAISS index
    model, index, embeddings = create_faiss_index(all_texts)

    # Take a user query
    query = input("Enter your query: ")

    # Retrieve the most relevant shlokas
    top_results = retrieve_shlokas(query, all_texts, model, index)

    # Output in JSON format
    output = {
        "query": query,
        "results": top_results
    }

    # Print and save the output as JSON
    print(json.dumps(output, indent=4))
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

# Run the script
if __name__ == "__main__":
    main()
