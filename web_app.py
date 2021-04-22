#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import pickle
import shutil
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
import cv2
from PIL import Image, ImageOps
import numpy as np


# In[28]:


import tensorflow as tf
from tensorflow.keras.models import load_model

# pickle_in1 = open("Agri_pesticide.pkl","rb")
# model1=pickle.load(pickle_in1)

# pickle_in2 = open("Agri_pesticide.pkl","rb")
# model2=pickle.load(pickle_in2)

# pickle_in3 = open("Agri_pesticide.pkl","rb")
# model3=pickle.load(pickle_in3)

pickle_in4 = open("crop_recommendation.pkl","rb")
model4=pickle.load(pickle_in4)

model5 = load_model('plant_disease.hdf5')



pickle_in6 = open("Agri_pesticide.pkl","rb")
model6=pickle.load(pickle_in6)
def predict_crop_damage(Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Number_Weeks_Quit,Season):

    prediction=model6.predict([[Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Number_Weeks_Quit,Season]])
    print(prediction)
    return prediction

def import_and_predict(image_data,model):
    img=ImageOps.fit(Image.ANTIALIAS,image_data,target_size=(220,220))
    x = image.img_to_array(img)
    x = x/255
    result = model.predict([np.expand_dims(x, axis=0)])
    
    return result
#@st.cache(allow_output_mutation=True)

def main():

  # st.title('AGRI AI')
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> AGRI AI 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    selection = st.radio("Select Use", ['Crop Disease Detection', 'Weed Detection', 'Yield Prediction', 'Crop Recommendation', 'Price Recommendation','Crop Health'])
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if selection == 'Crop Health':
        st.sidebar.markdown('<p>Graphs are auto-updated on each change.</p>', unsafe_allow_html=True)
        Estimated_Insects_Count = st.sidebar.text_input("Estimated_Insects_Count","Type Here")
        Crop_Type = st.sidebar.text_input("Crop_Type","Type Here")
        Soil_Type = st.sidebar.text_input("Soil_Type","Type Here")
        Pesticide_Use_Category = st.sidebar.text_input("Pesticide_Use_Category","Type Here")
        Number_Doses_Week = st.sidebar.text_input("Number_Doses_Week","Type Here")
        Number_Weeks_Used = st.sidebar.text_input("Number_Weeks_Used","Type Here")
        Number_Weeks_Quit = st.sidebar.text_input("Number_Weeks_Quit","Type Here")
        Season = st.sidebar.text_input("Season","Type Here")

        result=""
        if st.button("Predict"):
            result=predict_crop_damage(Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Number_Weeks_Quit,Season)
        if result==0:
            result = 'Alive'
        elif result==1:
            result = 'Damaged'
        elif result==2:
            result = 'Damaged due to Pesticides'

        st.success('The crop is {}'.format(result))

    elif selection =='Crop Disease Detection':
   
        #st.set_option('depreciation.showfileUploaderEncoding',False)
        
        st.markdown("""Crop Diseases Detection""")
        st.markdown(""" In recent times, drastic climate changes and lack of immunity in crops has caused substantial increase in growth of crop diseases. This causes large scale demolition of crops, decreases cultivation and eventually leads to financial loss of farmers. Due to rapid growth in variety of diseases , identification and treatment of the disease is a major importance.""" )
        file=st.sidebar.file_uploader("Please upload a crop image")
        
        #text_io = io.TextIOWrapper(file)
        if st.button("Detect"):
            if file is None:
                st.sidebar.text("please upload an image file")
                
            else:
                image=Image.open(file)
                st.image(image,use_column_width=True)
                predictions=import_and_predict(image,model5)
                from IPython.display import FileLink
                FileLink(r'class_indices.json')
                classes=list(class_indices.keys())
                classresult=np.argmax(predictions,axis=1)
                st.success("This crop is {}".format(classes[classresult[0]]))
                
    elif selection == 'Crop Recommendation':
        st.write("""#Crop Recommendation""")
        st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.
            """)
        st.sidebar.markdown(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.sidebar.number_input("Nitrogen", 1,10000)
        P = st.sidebar.number_input("Phosporus", 1,10000)
        K = st.sidebar.number_input("Potassium", 1,10000)
        temp = st.sidebar.number_input("Temperature",0.0,100000.0)
        humidity = st.sidebar.number_input("Humidity in %", 0.0,100000.0)
        ph = st.sidebar.number_input("Ph", 0.0,100000.0)
        rainfall = st.sidebar.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = model4.predict(single_pred)
            
            st.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")
            


# In[29]:


if __name__ == '__main__':
    main()


# In[ ]:




