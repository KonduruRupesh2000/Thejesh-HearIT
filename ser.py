from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import pandas as pd

from utils import AVAILABLE_EMOTIONS,int2emotion

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.2)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])

model_params = {
    'alpha': 0.005,
    'batch_size': 256, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")
try:
    pickle.dump(model, open("result/mlp_classifier.model", "wb"))
    print("Model Stored to Pickle File")
except:
    print("Not Stored in pickle file")

    
matrix = confusion_matrix(y_test, y_pred)
index = ['neutral','happy','sad','angry']  
columns = ['neutral','happy','sad','angry']
matrix_df = pd.DataFrame(matrix,columns,index)

print(matrix_df)


