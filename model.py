import os
import pickle

import pandas as pd
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>We are using {device} device.<<")


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

def nahidOrg_generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


with open('uiu_website_embeddings.pkl', 'rb') as f:
    article_embeddings = pickle.load(f)

dimension = len(article_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(article_embeddings))

def nahidOrg_retrieve_documents(query_embedding, k):
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

def nahidOrg_processing_context(docs, max_tokens=1536):
    processed_docs = []
    total_tokens = 0
    for doc in docs:
        tokens = len(doc.split())
        if total_tokens + tokens <= max_tokens:
            processed_docs.append(doc)
            total_tokens += tokens
        else:
            break
    return "\n".join(f"- {doc}" for doc in processed_docs)

GOOGLE_API_KEY = os.environ['GEMINI_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

def nahidOrg_answer_by_gemini(temperature, question, contexts):
   
    prompt = f"""Answer the question based on the provided context.
    **Question:** {question}
    **Context:** {contexts}
    """
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=250,
        temperature=temperature
    )
    
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(
        contents = prompt,
        generation_config = generation_config
    )
    
    return response.text
