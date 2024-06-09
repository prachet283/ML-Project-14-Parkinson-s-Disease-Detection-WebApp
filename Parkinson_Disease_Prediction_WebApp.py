# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 00:03:28 2024

@author: prachet
"""

import numpy as np
import pickle
import streamlit as st

#loading. the saved model
model = pickle.load(open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-14-Parkinson's Disease Detection/parkinsons_disease_trained_model.sav",'rb'))

with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-14-Parkinson's Disease Detection/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
    
def parkinson_disease_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    #standardize data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    
    return prediction


def main():
    
    #page title
    st.title('Parkinson Disease Prediction using ML')
    
    #getting input data from user
    #columns for input fields

    col1 , col2 , col3 , col4 = st.columns(4)

    with col1:
        Fo = st.text_input("MDVP_Fo(Hz)")
    with col2:
        Fhi = st.text_input("MDVP_Fhi(Hz)")
    with col3:
        Flo = st.text_input("MDVP_Flo(Hz)")
    with col4:
        Jitter_per = st.text_input("MDVP_Jitter(%)")
    with col1:
        Jitter_Abs = st.text_input("MDVP_Jitter(Abs)")
    with col2:
        RAP = st.text_input("MDVP_RAP")
    with col3:
        PPQ = st.text_input("MDVP_PPQ")
    with col4:
        Jitter_DDP = st.text_input("Jitter_DDP")
    with col1:
        Shimmer = st.text_input("MDVP_Shimmer")
    with col2:
        Shimmer_dB = st.text_input("MDVP_Shimmer(dB)")
    with col3:
        Shimmer_APQ3 = st.text_input("Shimmer_APQ3")
    with col4:
        Shimmer_APQ5  = st.text_input("Shimmer_APQ5")
    with col1:
        APQ = st.text_input("MDVP_APQ")
    with col2:
        Shimmer_DDA  = st.text_input("Shimmer_DDA")
    with col3:
        NHR = st.text_input("NHR")
    with col4:
        HNR = st.text_input("HNR")
    with col1:
        RPDE = st.text_input("RPDE")
    with col2:
        DFA = st.text_input("DFA")
    with col3:
        spread1 = st.text_input("spread1")
    with col4:
        spread2 = st.text_input("spread2")
    with col1:
        D2 = st.text_input("D2")
    with col2:
        PPE = st.text_input("PPE")

    # code for prediction
    parkinson_isease_diagnosis = ''

    #creating a button for Prediction
    if st.button('Parkinson Disease Test Result'):
        parkinson_isease_diagnosis=parkinson_disease_prediction([[Fo,Fhi,Flo,
                                             Jitter_per,Jitter_Abs,RAP,
                                             PPQ,Jitter_DDP,Shimmer,
                                             Shimmer_dB,Shimmer_APQ3,
                                             Shimmer_APQ5,APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        if(parkinson_isease_diagnosis[0]==0):
            parkinson_isease_diagnosis = 'The Person does not have Parkinson Disease' 
        else:
            parkinson_isease_diagnosis = 'The Person have Parkinson Disease'
        st.success(parkinson_isease_diagnosis)
        
if __name__ == '__main__':
    main()

