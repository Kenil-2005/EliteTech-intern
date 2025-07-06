from flask import Flask, render_template, request
from recommender import recommend_movies

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie = request.form['movie']
        recommendations = recommend_movies(movie)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
