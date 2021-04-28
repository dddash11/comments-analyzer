import streamlit as st
from keras.models import load_model
import tensorflow as tf
import pickle
import pandas as pd
import time
import base64


@st.cache
def start_btn_func(df, cla, cv, progress):
    for i in range(len(df['review'])):
        content = df['review'][i]
        pred = cla.predict(cv.transform([content]))
        if pred > 0.5:
            df['output'][i] = "Positive"
        else:
            df['output'][i] = "Negative"
        progress = progress+scale
        if progress >= 1.0:
            progress = 1.0
        my_bar.progress(progress)
    st.success("Complete")
    time.sleep(0.1)
    df.drop(['review'], axis=1, inplace=True)

    st.write(df.head())
    data = pd.DataFrame(df).to_csv(index=False)

    return data


cv_path = "models/countvectorizer.pkl"
with open(cv_path, 'rb') as file:
    cv = pickle.load(file)

ann_path = "models/phone_review.h5"
cla = load_model(ann_path, compile=False)
cla.compile(optimizer='adam', loss='binary_crossentropy')

st.title("Comments Analyser")

rad = st.sidebar.radio("Navigation", ["Home", "CSV upload"])

if rad == "Home":

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

# st.header("OR")


if rad == "CSV upload":

    st.subheader(
        "Upload a csv file to generate the overall sentiment of each review")

    csv_file = st.file_uploader("Upload a csv file")

    if csv_file:

        df = pd.read_csv(csv_file)

        st.write(df.head())
        start_btn = st.button("Start")
        progress = 0.0
        # t = st.empty()
        my_bar = st.progress(progress)

        df.insert(8, "output", "")
        df['review'] = df['title']+df['body']
        df['review'].fillna(df['review'].mode()[0], inplace=True)
        scale = 1/(len(df['review']))

        if start_btn:
            data = start_btn_func(df, cla, cv, progress)
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download Results</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.write(href, unsafe_allow_html=True)

            # anal_btn = st.button("Analyse results")
        if st.sidebar.button("Analyse results"):
            st.write("Output file analysis")

            rating_1 = (df[df['rating'] == 1])
            rating_2 = (df[df['rating'] == 2])
            rating_3 = (df[df['rating'] == 3])
            rating_4 = (df[df['rating'] == 4])
            rating_5 = (df[df['rating'] == 5])

            st.subheader("Review Count")
            st.write("5⭐ :     ", rating_5.shape[0])
            st.write("4⭐ :     ", rating_4.shape[0])
            st.write("3⭐ :     ", rating_3.shape[0])
            st.write("2⭐ :     ", rating_2.shape[0])
            st.write("1⭐ :     ", rating_1.shape[0])

            positives = data[data['output'] == 'Positive']
            negatives = data[data['output'] == 'Negative']

            st.subheader("Sentiment Count")
            st.write("Positive 😀 :     ", positives.shape[0])
            st.write("Negative ☹️ :     ", negatives.shape[0])
