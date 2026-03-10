import sys
import json
import ollama
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding(text, model_name="embeddinggemma:latest"):
    response = ollama.embeddings(
        model=model_name,
        prompt=text
    )
    return np.array(response["embedding"], dtype=np.float32)


def load_model(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(json.loads(x), dtype=np.float32)
    )
    return df


def find_best(model_df, query):
    query_emb = get_embedding(query)
    all_emb = np.vstack(model_df["embedding"].values)
    sims = cosine_similarity([query_emb], all_emb)[0]
    idx = int(np.argmax(sims))
    return model_df.iloc[idx], sims[idx]


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Usage: python search.py "Show me a bear" model.csv')
        sys.exit(1)

    query = sys.argv[1]
    model_file = sys.argv[2]

    model_df = load_model(model_file)

    result, score = find_best(model_df, query)

    print("\nQuery:", query)
    print("Best image:", result["filename"])
    print("Description:", result["description"])
    print("Similarity:", score)