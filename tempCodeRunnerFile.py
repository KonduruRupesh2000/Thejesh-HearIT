from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
index = ['neutral','happy','sad','angry']  
columns = ['neutral','happy','sad','angry']
matrix_df = pd.DataFrame(matrix,columns,index)

print(matrix_df)