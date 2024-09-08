from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Function to load pickle files with joblib and fallback to pandas if necessary
def load_pickle(filename):
    try:
        data = joblib.load(filename)
        logging.info(f"Successfully loaded {filename}")
        return data
    except Exception as e:
        logging.error(f"Failed to load {filename} with joblib: {e}")
        try:
            data = pd.read_pickle(filename)
            logging.info(f"Successfully loaded {filename} with pandas")
            return data
        except Exception as e:
            logging.error(f"Failed to load {filename} with pandas: {e}")
            return None

# Load data
popular_df = load_pickle('popular.pkl')
pt = load_pickle('pt.pkl')
books = load_pickle('books.pkl')
similarity_scores = load_pickle('similarity_scores.pkl')

# Check if data was loaded successfully
if popular_df is None or pt is None or books is None or similarity_scores is None:
    raise RuntimeError("One or more data files could not be loaded. Check the logs for details.")

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df.get('Book-Title', [])),
                           author=list(popular_df.get('Book-Author', [])),
                           image=list(popular_df.get('Image-URL-M', [])),
                           votes=list(popular_df.get('num_ratings', [])),
                           rating=list(popular_df.get('avg_rating', []))
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    
    if pt.empty or user_input not in pt.index:
        return render_template('recommend.html', data=[], error="Book not found")

    index = np.where(pt.index == user_input)[0][0]
    
    if similarity_scores.size == 0:
        return render_template('recommend.html', data=[], error="No similarity scores available")

    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title').get('Book-Title', [])))
        item.extend(list(temp_df.drop_duplicates('Book-Title').get('Book-Author', [])))
        item.extend(list(temp_df.drop_duplicates('Book-Title').get('Image-URL-M', [])))

        data.append(item)

    print(data)

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
