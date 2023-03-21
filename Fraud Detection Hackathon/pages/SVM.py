import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import streamlit as st


def prediction(X_test, model):
    # Predicton on test with giniIndex
    y_pred = model.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))

credit_card_data = pd.read_csv("pages//creditcard.csv")

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# compare the values for both transactions
credit_card_data.groupby('Class').mean()

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset.groupby('Class').mean()

scaler = MinMaxScaler()
print(scaler.fit(new_dataset.drop('Time',axis=1)))
MinMaxScaler()
print(scaler.data_max_)

print(scaler.transform(new_dataset.drop('Time',axis=1)))
new_dataset1 = new_dataset.drop('Time',axis=1)

X = new_dataset1.drop(columns='Class', axis=1)
Y = new_dataset1['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# SVM

clf = SVC(kernel='linear')

# fitting x samples and y classes
clf.fit(X_train, Y_train)

X_train_prediction_svm = clf.predict(X_train)
training_data_accuracy_svm = accuracy_score(X_train_prediction_svm, Y_train)

y_pred_svc = clf.predict(X_test)

st.title("Support Vector Machine")

st.write(classification_report(Y_test , y_pred_svc))

st.write("Accuracy Score on Test Data : ", {accuracy_score(Y_test, y_pred_svc)})