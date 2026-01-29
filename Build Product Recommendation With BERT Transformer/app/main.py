from flask import Flask, request , render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

#load pickle 
pkl_path = os.path.join(BASE_DIR, "product_embedding.pkl")
dataframe = pd.read_pickle(pkl_path)
dataframe['imgs'] = dataframe['imgs'].apply(lambda x: eval(x) if isinstance(x, str) else x)


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    # static_folder=os.path.join(BASE_DIR, "static")
)


def recommend_products(product_name, num_recommendations=10):
  query_embedding = model.encode(product_name)
  dataframe['similarity'] = dataframe['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x]).flatten()[0])
  recommendations = dataframe.sort_values(by='similarity', ascending=False).head(num_recommendations)
  return recommendations[['title', 'brand', 'category', 'similarity', 'imgs']]


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        query = request.form['query']
        recommendations = recommend_products(query).to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
