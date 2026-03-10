import sys
import os
import json
import ollama
import numpy as np
import pandas as pd


def describe_image(image_path, model_name="ministral-3:3b"):
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Describe this animal image briefly in one sentence for image search.",
                "images": [image_path],
            }
        ],
    )
    return response["message"]["content"]


def get_embedding(text, model_name="embeddinggemma"):
    response = ollama.embeddings(
        model=model_name,
        prompt=text
    )
    return np.array(response["embedding"], dtype=np.float32)


def load_model(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["filename", "description", "embedding"])

    df = pd.read_csv(path)

    if len(df) == 0:
        return pd.DataFrame(columns=["filename", "description", "embedding"])

    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(json.loads(x), dtype=np.float32)
    )
    return df


def save_model(df, path):
    tmp = df.copy()
    tmp["embedding"] = tmp["embedding"].apply(lambda x: json.dumps(x.tolist()))
    tmp.to_csv(path, index=False)


def process_images(image_list_file, model_file):
    with open(image_list_file, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    model_df = load_model(model_file)

    # avoid duplicates if rerun
    existing_files = set(model_df["filename"].tolist()) if len(model_df) > 0 else set()

    new_rows = []

    for image_path in image_paths:
        if image_path in existing_files:
            print(f"Skipping already processed: {image_path}")
            continue

        print(f"Processing: {image_path}")

        try:
            description = describe_image(image_path)
            embedding = get_embedding(description)

            new_rows.append({
                "filename": image_path,
                "description": description,
                "embedding": embedding
            })

            print(f"Done: {image_path}")
            print(f"Description: {description}")

        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        model_df = pd.concat([model_df, new_df], ignore_index=True)

    save_model(model_df, model_file)
    print(f"Saved updated model to: {model_file}")
    print(f"Total images in model: {len(model_df)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py images.txt model.csv")
        sys.exit(1)

    image_list_file = sys.argv[1]
    model_file = sys.argv[2]

    process_images(image_list_file, model_file)