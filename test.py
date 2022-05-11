import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier
import subprocess
import random
from ser import accuracy
from utils import extract_feature

if __name__ == "__main__":
    # load the saved model (after training)
    model = pickle.load(open("result/mlp_classifier.model", "rb"))
    filename = "happy.wav"
    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    # show the result !
    print("Accuracy: {:.2f}%".format(accuracy*100))
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


    