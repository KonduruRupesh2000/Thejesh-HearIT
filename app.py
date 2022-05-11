from flask import Flask, render_template, request, redirect
from utils import extract_feature
import pickle
import subprocess
import random
import os
app = Flask(__name__)

@app.route("/projectdetails.html", methods=["GET", "POST"])

def project():
    return render_template('projectdetails.html')

@app.route("/", methods=["GET", "POST"])

def home():
    return render_template('index.html')
@app.route("/inner-page.html", methods=["GET", "POST"])

def index():
    result=""
    
    if request.method == "POST":
        print("DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            model = pickle.load(open("result/mlp_classifier.model", "rb"))
            features = extract_feature(file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
            result = model.predict(features)[0]
            if result=='neutral':
                return render_template('neutral.html')
            elif result=='angry':
                return render_template('angry.html')
            elif result=='sad':
                return render_template('sad.html')
            elif result=='happy':
                return render_template('happy.html')
             
    return render_template('inner-page.html', transcript=result)

@app.route('/inner-page.html')
def about():
    return render_template('inner-page.html')




if __name__ == "__main__":
    app.run(debug=True, threaded=True)
