# ds-datamining
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
!pip install tensorflow
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)
# Encodage des étiquettes en one-hot
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Création du modèle
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test))
# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')
#create a random forest classifier
clf=RandomForestClassifier(n_estimators=10, criterion="gini",random_state=4)

#training the model on the training dataset
clf.fit(X_train,y_train)

#predicting the test set result
y_pred=clf.predict(X_test)

#model accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
