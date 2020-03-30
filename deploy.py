import pandas as pd
from flask import Flask, jsonify, request
import pickle
from summary_handler import Summarizer
# load model
model = pickle.load(open('small_saved_model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, y) for x, y in data.items())
    text = data["body"]

    # predictions
    result = model(text,ratio=0.2,algorithm='kmeans',use_first=True,min_length=10,max_length=500)
    # send back to browser
    # return data
    return jsonify(results=result)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)