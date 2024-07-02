# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 00:03:28 2024

@author: prachet
"""
import json
import pickle
#import numpy as np
import streamlit as st
import pandas as pd

#loading. the saved model
with open("Updated_Project/columns.pkl", 'rb') as f:
    all_features = pickle.load(f)
with open("Updated_Project/scaler.pkl", 'rb') as f:
    scalers = pickle.load(f)
with open("Updated_Project/best_features_knn.json", 'r') as file:
    best_features_knn = json.load(file)
with open("Updated_Project/best_features_xgb.json", 'r') as file:
    best_features_xgb = json.load(file)
with open("Updated_Project/best_features_rfc.json", 'r') as file:
    best_features_rfc = json.load(file)
with open("Updated_Project/parkinsons_disease_trained_knn_model.sav", 'rb') as f:
    loaded_model_knn = pickle.load(f)
with open("Updated_Project/parkinsons_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb = pickle.load(f)
with open("Updated_Project/parkinsons_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc = pickle.load(f)

    
def parkinson_disease_prediction(input_data):
    
    
    df = pd.DataFrame([input_data], columns=all_features)

    df[all_features] = scalers.transform(df[all_features])
    
    df_best_features_knn = df[best_features_knn]
    df_best_features_xgb = df[best_features_xgb]
    df_best_features_rfc = df[best_features_rfc]
    
    prediction1 = loaded_model_knn.predict(df_best_features_knn)
    prediction2 = loaded_model_xgb.predict(df_best_features_xgb)
    prediction3 = loaded_model_rfc.predict(df_best_features_rfc)
    
    
    return prediction1 , prediction2, prediction3


def main():
    
    #page title
    st.title('Parkinson Disease Prediction using ML')
    
    #getting input data from user
    #columns for input fields

    col1 , col2 , col3 = st.columns(3)

    with col1:
        Fo = st.number_input("MDVP_Fo(Hz)",format="%.6f")
    with col2:
        Fhi = st.number_input("MDVP_Fhi(Hz)",format="%.6f")
    with col3:
        Flo = st.number_input("MDVP_Flo(Hz)",format="%.6f")
    with col1:
        Jitter_per = st.number_input("MDVP_Jitter(%)",format="%.6f")
    with col2:
        Jitter_Abs = st.number_input("MDVP_Jitter(Abs)",format="%.6f")
    with col3:
        RAP = st.number_input("MDVP_RAP",format="%.6f")
    with col1:
        PPQ = st.number_input("MDVP_PPQ",format="%.6f")
    with col2:
        Jitter_DDP = st.number_input("Jitter_DDP",format="%.6f")
    with col3:
        Shimmer = st.number_input("MDVP_Shimmer",format="%.6f")
    with col1:
        Shimmer_dB = st.number_input("MDVP_Shimmer(dB)",format="%.6f")
    with col2:
        Shimmer_APQ3 = st.number_input("Shimmer_APQ3",format="%.6f")
    with col3:
        Shimmer_APQ5  = st.number_input("Shimmer_APQ5",format="%.6f")
    with col1:
        APQ = st.number_input("MDVP_APQ",format="%.6f")
    with col2:
        Shimmer_DDA  = st.number_input("Shimmer_DDA",format="%.6f")
    with col3:
        NHR = st.number_input("NHR",format="%.6f")
    with col1:
        HNR = st.number_input("HNR",format="%.6f")
    with col2:
        RPDE = st.number_input("RPDE",format="%.6f")
    with col3:
        DFA = st.number_input("DFA",format="%.6f")
    with col1:
        spread1 = st.number_input("spread1",format="%.6f")
    with col2:
        spread2 = st.number_input("spread2",format="%.6f")
    with col3:
        D2 = st.number_input("D2",format="%.6f")
    with col1:
        PPE = st.number_input("PPE",format="%.6f")

    # code for prediction
    parkinson_isease_diagnosis_knn = ''
    parkinson_isease_diagnosis_xgb = ''
    parkinson_isease_diagnosis_rfc = ''
    parkinson_isease_diagnosis_knn,parkinson_isease_diagnosis_xgb,parkinson_isease_diagnosis_rfc = parkinson_disease_prediction([Fo,Fhi,Flo,Jitter_per,Jitter_Abs,RAP,PPQ,Jitter_DDP,Shimmer,Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
    
    #creating a button for Prediction
    if st.button("Predict Parkinson Disease"):
        if(parkinson_isease_diagnosis_knn[0]==0):
            prediction = 'The Person does not have Parkinson Disease' 
        else:
            prediction = 'The Person have Parkinson Disease'
        st.write(f"Prediction: {prediction}")
    
    if st.checkbox("Show Advanced Options"):
        if st.button("Predict Breast Cancer with K Neighbors Classifier"):
            if(parkinson_isease_diagnosis_knn[0]==0):
                prediction = 'The Person does not have Parkinson Disease' 
            else:
                prediction = 'The Person have Parkinson Disease'
            st.write(f"Prediction: {prediction}")
        if st.button("Predict Breast Cancer with Random Forest Classifier"):
            if(parkinson_isease_diagnosis_rfc[0]==0):
                prediction = 'The Person does not have Parkinson Disease' 
            else:
                prediction = 'The Person have Parkinson Disease'
            st.write(f"Prediction: {prediction}")
        if st.button("Predict Breast Cancer with XG Boost Classifier"):
            if(parkinson_isease_diagnosis_xgb[0]==0):
                prediction = 'The Person does not have Parkinson Disease' 
            else:
                prediction = 'The Person have Parkinson Disease'
            st.write(f"Prediction: {prediction}")
        
if __name__ == '__main__':
    main()

