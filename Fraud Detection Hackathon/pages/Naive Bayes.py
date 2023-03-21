import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import streamlit as st
import sklearn
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


rad = st.radio("Types",["Credit Card", "Insurance Claim"])

if rad == "Credit Card":
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
    print(scaler.fit(new_dataset.drop('Time', axis=1)))
    MinMaxScaler()
    print(scaler.data_max_)

    print(scaler.transform(new_dataset.drop('Time', axis=1)))
    new_dataset1 = new_dataset.drop('Time', axis=1)

    X = new_dataset1.drop(columns='Class', axis=1)
    Y = new_dataset1['Class']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # NAIVE BAYES

    gnb = GaussianNB()

    y_pred_nb = gnb.fit(X_train, Y_train).predict(X_test)

    st.title("Naive Bayes")

    st.write(classification_report(Y_test, y_pred_nb))

    st.write("Accuracy Score on Test Data : ", {accuracy_score(Y_test, y_pred_nb)})





if rad =="Insurance Claim":
    data = pd.read_csv("pages//insurance_claims.csv")

    data.drop('_c39', axis=1, inplace=True)

    data.drop('policy_bind_date', axis=1, inplace=True)

    one_hot_encoded_data = pd.get_dummies(data['policy_state'], columns=['OH', 'IN', 'IL'])

    data2 = data.drop(['policy_state', 'auto_model'], axis=1)

    data = pd.concat([data2, one_hot_encoded_data], axis=1)

    data.drop(['auto_make', 'auto_year'], axis=1, inplace=True)

    one_hot_encoded_data1 = pd.get_dummies(data['insured_education_level'],
                                           columns=['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College',
                                                    'JD'])

    data3 = data.drop(['insured_education_level'], axis=1)

    data = pd.concat([data3, one_hot_encoded_data1], axis=1)

    data4 = data.drop(
        ['incident_date', 'incident_type', 'incident_location', 'policy_csl', 'insured_hobbies', 'insured_relationship',
         'insured_occupation', 'insured_relationship', 'collision_type', 'incident_state', 'incident_city',
         'authorities_contacted'], axis=1)

    data4.insured_sex.replace('MALE', 0, inplace=True)
    data4.insured_sex.replace('FEMALE', 1, inplace=True)

    one_hot_encoded_data3 = pd.get_dummies(data4['incident_severity'],
                                           columns=['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'])

    data4 = data4.drop(['incident_severity'], axis=1)

    data4 = pd.concat([data4, one_hot_encoded_data3], axis=1)


    def print_unique_col_values(df):
        for column in df:
            if df[column].dtypes != 'object':
                print(f'(column) : {df[column].unique()}')


    # Function to calculate accuracy
    def cal_accuracy(y_test, y_pred):

        print("Confusion Matrix: ",
              confusion_matrix(y_test, y_pred))

        print("Accuracy : ",
              accuracy_score(y_test, y_pred) * 100)

        print("Report : ",
              classification_report(y_test, y_pred))

    data4.drop(['property_damage'], axis=1, inplace=True)

    data4.drop(['police_report_available'], axis=1, inplace=True)

    data4.fraud_reported.replace('Y', 1, inplace=True)
    data4.fraud_reported.replace('N', 0, inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(data4)
    data5 = scaler.transform(data4)

    df = pd.DataFrame(data5, columns=[columns for columns in data4])

    X = df.drop(columns='fraud_reported', axis=1)
    Y = df['fraud_reported']

    smote = SMOTE(sampling_strategy='minority')
    X_sm, Y_sm = smote.fit_resample(X, Y)
    Y_sm.value_counts()

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_sm, Y_sm, test_size=0.2,
                                                                                stratify=Y_sm, random_state=2)

    gnb = GaussianNB()

    y_pred_nb = gnb.fit(X_train, Y_train).predict(X_test)

    st.title("Insurance Claim")
    st.write({"Accuracy Score on Test Data :", accuracy_score(Y_test, y_pred_nb)*100})