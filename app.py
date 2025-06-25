from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd
import torch

app = Flask(__name__)

# Load dataset
df = pd.read_csv("college_qa_dataset.csv", encoding='latin1')
df.columns = df.columns.str.strip().str.lower()

# Load models once
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = model_embed.encode(df['input'].tolist(), convert_to_tensor=True)
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

THRESHOLD = 0.6  # confidence threshold

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        user_input = request.json.get("queryResult", {}).get("queryText", "")
        user_embedding = model_embed.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(user_embedding, faq_embeddings)
        best_idx = int(scores.argmax())
        best_score = float(scores[0][best_idx])
        matched_answer = df.iloc[best_idx]['output']

        if best_score >= THRESHOLD:
            response = matched_answer
        else:
            prompt = f"Answer like a helpful college counselor: {user_input}"
            response = gen_pipeline(prompt, max_new_tokens=100)[0]['generated_text']

        return jsonify({"fulfillmentText": response.strip()})

    except Exception as e:
        return jsonify({"fulfillmentText": f"Error: {str(e)}"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

            
