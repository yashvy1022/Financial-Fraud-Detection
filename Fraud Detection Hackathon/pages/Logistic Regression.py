import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import sklearn
from imblearn.over_sampling import SMOTE

rad = st.radio("Fraud Types", ["Credit Card","Insurance Claim"])

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

    # Logistic Regression

    model1 = LogisticRegression()

    model1.fit(X_train, Y_train)

    # accuracy on training data
    X_train_prediction = model1.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    X_test_prediction = model1.predict(X_test)
    testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    st.title("Logistics Regression")

    st.write('Accuracy Score on Test Data : ', testing_data_accuracy)






if rad == "Insurance Claim":
    # Insurance Claim

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


    data4.drop(['property_damage'], axis=1, inplace=True)

    data4.drop(['police_report_available'], axis=1, inplace=True)

    data4.fraud_reported.replace('Y', 1, inplace=True)
    data4.fraud_reported.replace('N', 0, inplace=True)

    scaler = MinMaxScaler()
    print(scaler.fit(data4.drop(['College', 'High School', 'JD', 'MD', 'Masters', 'PhD', 'Major Damage'], axis=1)))
    print("-------------------------")
    data6 = scaler.transform(data4.drop(['College', 'High School', 'JD', 'MD', 'Masters', 'PhD', 'Major Damage'], axis=1))
    data4.drop(['College', 'High School', 'JD', 'MD', 'Masters', 'PhD', 'Major Damage'], axis=1, inplace = True)
    data5 = pd.DataFrame(data6, [columns for columns in data4])


    df = pd.DataFrame(data5, columns=[columns for columns in data4])

    X = df.drop(columns='fraud_reported', axis=1)
    Y = df['fraud_reported']

    smote = SMOTE(sampling_strategy='minority')
    X_sm, Y_sm = smote.fit_resample(X, Y)
    Y_sm.value_counts()

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_sm, Y_sm, test_size=0.2,
                                                                                stratify=Y_sm, random_state=2)

    model1 = LogisticRegression()

    model1.fit(X_train, Y_train)

    X_train_prediction = model1.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    X_test_prediction = model1.predict(X_test)
    testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    st.write("Insurance Claim")

    st.write("Accuracy Score on Test Data : ", testing_data_accuracy)