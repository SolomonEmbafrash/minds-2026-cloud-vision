import ollama
import numpy as np

response = ollama.embeddings(
    model="embeddinggemma",
    prompt="show me a bear"
)
emb = np.array(response["embedding"])
print("Embedding works")
print("Shape:", emb.shape)

response = ollama.chat(
    model="ministral-3:3b",
    messages=[
        {"role": "user", "content": "Describe a bear in one short sentence."}
    ]
)
print("Chat works")
print(response["message"]["content"])