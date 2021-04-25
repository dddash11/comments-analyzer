import streamlit as st
from keras.models import load_model
import tensorflow as tf
import pickle
import pandas as pd


cv_path = "models/countvectorizer.pkl"
with open(cv_path, 'rb') as file:
    cv = pickle.load(file)

ann_path = "models/phone_review.h5"
cla = load_model(ann_path, compile=False)
cla.compile(optimizer='adam', loss='binary_crossentropy')

st.title("Comments Analyser")

st.image("performing-twitter-sentiment-analysis1new.png")

st.subheader("A simple app to analyse comments")

st.write("Your product is out there in the market. Want to know how people are liking it? Curious, if your product is favoured by the people?    \n\n\n\n\n\n\n\n\n\n\n\n\n Try our app !!! \n\n\n\n\n\n\n\n\n\nOur app provides you an overall review of your product based on the feedback provided by your customers. We drive your decisions towards your product's improvement!")

st.markdown("\n\n\n\n\n")

st.header("Try a demo!")
text = st.text_input("Type any comment here")
check = st.button("Submit")
if check:
    y_pred = cla.predict(cv.transform([text]))
    if y_pred > 0.5:
        st.header("Positive 😀")
        # st.image(
        #     "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRiEVTzyN-DCHwbjIH_4-z6a66voZmbMqT0jQ&usqp=CAU")
    else:
        st.header("Negative ☹️")
        # st.image(
        #     "https://www.clipartmax.com/png/middle/64-644516_very-sad-emoji-sad-emoji.png")

st.header("OR")
st.markdown("\n\n\n\n\n")
st.subheader(
    "Upload a csv file to generate the overall sentiments of each review")
csv_file = st.file_uploader("Upload a csv file")
if csv_file:
    df = pd.read_csv(csv_file)
    st.write(df.head())
    btn = st.button("Start")
    my_bar = st.progress(0)
    df.insert(8, "output", "")
    df['review'] = df['title']+df['body']
    if btn:
        for i in range(len(df['review'])):
            pred = cla.predict(cv.transform([df['review'][i]]))
        if pred > 0.5:
            df['output'][i] = "Positive"
        else:
            df['output'][i] = "Negative"
            my_bar.progress(i+1)
        st.write("Complete")
        st.write(df.head())
