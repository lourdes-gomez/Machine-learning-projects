import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
import os
seed = 42


X_train = pd.read_csv(r'C:\Users\lour2\Desktop\LOURDES\data science\Proyecto Machine Learning\data\processed\X_train_data.csv')
y_train = pd.read_csv(r'C:\Users\lour2\Desktop\LOURDES\data science\Proyecto Machine Learning\data\processed\Y_train_data.csv')

X_train.to_numpy()
y_train = y_train.iloc[:,0]

lr = LogisticRegression(C= 0.5, max_iter= 38, random_state=42)

lr.fit(X_train, y_train)


file_path = os.path.join(r'C:\Users\lour2\Desktop\LOURDES\data science\Proyecto Machine Learning\model', 'my_model.pkl')

with open(file_path, 'wb') as archivo_salida:   #wb 'write bytes'. abrir archivo y escribir bytes 
    pickle.dump(lr, archivo_salida)  #guarda el archivo