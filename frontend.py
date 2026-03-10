import json

import ollama

import numpy as np

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity



def get_embedding(text):

    response = ollama.embeddings(

        model="embeddinggemma:latest",

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



# -------- frontend --------



model = load_model("model.csv")



print("Animal Image Search System")

print("--------------------------")



while True:



    query = input("\nAsk for an animal (or type 'exit'): ")



    if query.lower() == "exit":

        break



    result, score = find_best(model, query)



    print("\nBest match:")

    print("Image:", result["filename"])

    print("Description:", result["description"])

    print("Similarity:", score)
