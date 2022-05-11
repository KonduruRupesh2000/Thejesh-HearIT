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
            print(result)   
            try:
                mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                if result == 'angry':
                    randomfile = random.choice(os.listdir("C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/angry/"))
                    print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                    file = ('C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/angry/' + randomfile)
                    subprocess.call([mp, file])
                    print("Playlist Completed, Please Run program to continue...!!")

                if result == 'happy':
                    randomfile = random.choice(os.listdir("C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/happy/"))
                    print('You are smiling :) ,I playing special song for you: ' + randomfile)
                    file = ('C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/happy/' + randomfile)
                    subprocess.call([mp, file])

                if result == 'calm':
                    randomfile = random.choice(os.listdir("C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/calm/"))
                    print('You have fear of something ,I playing song for you: ' + randomfile)
                    file = ('C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/calm/' + randomfile)
                    subprocess.call([mp, file])

                if result == 'sad':
                    randomfile = random.choice(os.listdir("C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/sad/"))
                    print('You are sad,dont worry:) ,I playing song for you: ' + randomfile)
                    file = ('C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/sad/' + randomfile)
                    subprocess.call([mp, file])
                if result == 'neutral':
                    randomfile = random.choice(os.listdir("C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/neutral/"))
                    print('You are normal,I playing song for you: ' + randomfile)
                    file = ('C:/Users/Thejesh/Desktop/Hear it/HEAR-IT(UPDATED)/neutral/' + randomfile)
                    subprocess.call([mp, file])  
            except:
                print('Unable to detect emotion correctly, Please try Again..!') 
    return render_template('inner-page.html', transcript=result)

@app.route('/inner-page.html')
def about():
    return render_template('inner-page.html')




if __name__ == "__main__":
    app.run(debug=True, threaded=True)
