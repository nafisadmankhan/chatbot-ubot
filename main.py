import json
from flask import Flask, render_template, request
import pandas as pd

from model import nahidOrg_generate_embeddings, nahidOrg_retrieve_documents, nahidOrg_processing_context, nahidOrg_answer_by_gemini

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/profile')
def profile():
    return render_template("profile.html")

@app.route('/grade')
def grade():
    return render_template("grade.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/string_operation', methods=["POST"])
def string_operation():
    if request.method == "POST":
        dataToReceive = request.get_json()
        dataToSend = dataToReceive['inputKey']

        question_embeddings = nahidOrg_generate_embeddings(dataToSend)
        retrived_articles_indices = nahidOrg_retrieve_documents(question_embeddings, 3)
        retrived_articles_series = pd.Series(retrived_articles_indices)
        df_filtered = pd.read_csv('uiu_website.csv')
        retrieved_text_articles = retrived_articles_series.apply(lambda x: df_filtered['Article text'][x]).tolist()
        final_retrived_contexts = nahidOrg_processing_context(retrieved_text_articles)
        answer_gemini_pro = nahidOrg_answer_by_gemini(temperature=0.5, question=dataToSend, contexts=final_retrived_contexts)

    response = {"response": answer_gemini_pro}
    return json.dumps(response)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
