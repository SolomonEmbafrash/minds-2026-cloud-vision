# Cloud Vision – Assignment 3  
## Image Search with Ollama Embeddings

Author: Solomon Kiflom  
Course: Cloud Computing and Data Engineering – MINDS 2026

GitHub repository:  
https://github.com/SolomonEmbafrash/minds-2026-cloud-vision

---

# Project Description

This project implements a simple AI-based image search system using Ollama embedding models.
The goal is to build a pipeline that processes images, generates textual descriptions and embedding vectors, and allows a user to search for images using natural language queries.

The experiment was first developed and tested on a local computer, and then executed on the CSC Puhti supercomputer to demonstrate the same workflow in a cloud/HPC environment.

---

# System Overview

The system works by converting both image descriptions and user queries into numerical vectors called embeddings.
These embeddings capture the semantic meaning of the text.
By comparing embeddings using similarity metrics, the system can retrieve the most relevant image.

Workflow:

1. Load images from the frames folder
2. Generate a short description of each image
3. Convert the description into an embedding vector
4. Store embeddings and metadata in a Pandas dataframe
5. Save the dataframe as model.csv
6. Convert a user query into an embedding
7. Compare query embedding with stored embeddings
8. Return the most similar image

---

# Technologies Used

- Python
- Ollama
- embeddinggemma model
- ministral-3:3b model
- NumPy
- Pandas
- Jupyter Notebook
- Visual Studio Code
- CSC Puhti supercomputer

---

# Project Structure

Assignment3

frames/                     # image dataset  
pipeline.py                 # creates embeddings and builds the model  
pipeline.sh                 # Puhti batch execution script  
search.py                   # script to query the model  
frontend.py                 # simple command line interface  

model.csv                   # stored image embeddings  

test_ollama_local.ipynb  
test_ollama_puhti.ipynb  

README.md  

---

# Running the System Locally

Install required libraries:

pip install ollama pandas numpy

Run the frontend:

python frontend.py

Example query:

Show me a zebra

Example result:

Best match:
Image: frames/frame_0021.jpg
Description: A group of zebras grazing in a grassy savanna.
Similarity: 0.51

---

# Running the System on CSC Puhti

Load the Python environment:

module load pytorch/2.7

Run the search script:

python3 search.py "Show me a bear" model.csv

The script returns the image path, description, and similarity score.

---

# Results

The final system successfully retrieves images that match the meaning of the user query.

Example:

Query: Show me a bear

Best image: frames/frame_0109.jpg
Description: A close-up of a bear partially submerged in water.
Similarity score: 0.57

---

# What I Learned

- How embedding models represent semantic meaning
- How to build a simple vector search pipeline
- How to use Ollama for local AI inference
- How to integrate AI models with Python workflows
- How to run AI experiments on the CSC Puhti HPC environment
- How to move a project from local development to cloud infrastructure

---

# Challenges

Some challenges encountered during the project included:

- Installing and configuring Ollama on Puhti
- Ensuring the Ollama server was running during compute jobs
- Managing Python package dependencies
- Handling large embedding vectors efficiently

---

# Conclusion

This project demonstrates how modern AI models can be used to build an intelligent image retrieval system.
By combining LLM-generated descriptions, embedding models, and vector similarity search, it is possible to retrieve relevant images using natural language queries.

Running the experiment both locally and on CSC Puhti provided valuable experience in deploying AI workflows across different computing environments.

---

# Author

Solomon Kiflom  
MINDS 2026 – Cloud Vision
