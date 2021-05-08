import base64
import time
import pickle
import tensorflow as tf
from keras.models import load_model
import streamlit as st
from collections import Counter
from matplotlib import rcParams
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import activations, initializers, regularizers, constraints
from keras.utils.conv_utils import conv_output_length
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.initializers import glorot_uniform
from keras.preprocessing import text, sequence
from keras.regularizers import l2
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import os
import plotly.express as px
np.random.seed(0)


# load all models

# lstm
lstm_path = "models/lstm.h5"
lstm = load_model(lstm_path, compile=False)
lstm.compile(optimizer='adam', loss='categorical_crossentropy')
# xgboost
file_name = "models/xgb_reg.pkl"
xgb = pickle.load(open(file_name, "rb"))
# naive bayes
nb_file_name = "models/nb_clf.pkl"
NB_classifier = pickle.load(open(nb_file_name, "rb"))
# decision tree
dt_file_name = "models/dt_clf.pkl"
DT_classifier = pickle.load(open(dt_file_name, "rb"))


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

# convert number to one_hot vector


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# sentences to word indices


def sentences_to_indices(X, word_to_index):
    # number of training examples
    m = X.shape[0]
    X_indices = np.zeros(m)
    X_indices = []
    for i in range(m):                      # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = [word.lower().replace('\t', '')
                          for word in X[i].split(' ') if word.replace('\t', '') != '']
        # sentence_words=X[i].split(' ')
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        li = np.zeros(len(sentence_words))
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                li[j] = word_to_index[w]
            except:
               # print(w)
                li[j] = 0
            # Increment j to j + 1
            j += 1
        X_indices.append(li)
    X_indices = np.array(X_indices)
    return X_indices


def sentiment_category(score):
    if score >= 4:
        return "positive"
    elif score <= 2:
        return "negative"
    else:
        return "neutral"


def getMostCommon(reviews_list, topn=20):
    reviews = " ".join(reviews_list)
    tokenised_reviews = reviews.split(" ")
    freq_counter = Counter(tokenised_reviews)
    return freq_counter.most_common(topn)


def generateNGram(text, n):
    tokens = text.split(" ")
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return ["_".join(ngram) for ngram in ngrams]


# default number of words is given as 30
def plotMostCommonWords(reviews_list, color, topn=30, title="Common Review Words"):
    top_words = getMostCommon(reviews_list, topn=topn)
    data = pd.DataFrame()
    data['words'] = [val[0] for val in top_words]
    data['freq'] = [val[1] for val in top_words]
    data.sort_values('words', ascending=False)

    # if axis != None:
    # st.bar_chart(data = data)
    # sns.barplot(y='words', x='freq', data=data).set_title(
    #     title+" top "+str(topn))
    fig = px.bar(data,  x='freq', y='words')
    fig.update_traces(marker_color=color)

    st.plotly_chart(fig)
    # st.pyplot()
    # else:
    #     # st.bar_chart(data = data)
    #     sns.barplot(y='words', x='freq', data=data,
    #                 color=color).set_title(title+" top "+str(topn))
    #     st.pyplot()


@st.cache
def generate_reviews(df):

    df['star'] -= 1
    df_indices = sentences_to_indices(np.array(df['body']), word_to_index)
    df_indices = sequence.pad_sequences(df_indices, maxlen=100)
    df_preds = lstm.predict(df_indices)
    df_preds = [np.argmax(pred) for pred in df_preds]
    df['lstm_sentiment_score'] = df_preds
    df['lstm_review_category'] = df['lstm_sentiment_score'].apply(
        lambda x: sentiment_category(x))

    lstm_accuracy = accuracy_score(
        df['star'].astype('int'), df_preds)

    # wordcloud = WordCloud(height=4000, width=4000,
    #                       background_color='black')
    # wordcloud = wordcloud.generate(
    #     ' '.join(df.loc[df['lstm_review_category'] == 'positive', 'body'].tolist()))
    # plt.imshow(wordcloud)
    # plt.title("Most common words in positive customer comments")
    # plt.axis('off')
    # st.pyplot()

    df_xgb_preds = xgb.predict(df_indices)
    df['xgb_preds'] = df_xgb_preds
    xgb_accuracy = accuracy_score(df['star'].astype('int'), df_xgb_preds)

    df_nb_preds = NB_classifier.predict(df_indices)
    df['NB_preds'] = df_nb_preds
    NB_accuracy = accuracy_score(df['star'].astype('int'), df_nb_preds)

    df_dt_preds = DT_classifier.predict(df_indices)
    df['DT_preds'] = df_dt_preds
    DT_accuracy = accuracy_score(df['star'].astype('int'), df_dt_preds)

    return df, lstm_accuracy, xgb_accuracy, NB_accuracy, DT_accuracy


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    'glove.6B.50d.txt')

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Comments Analyser")
st.subheader(
    "Classify product reviews and opinions in English as positive or negative according to their sentiment")

rad = st.sidebar.radio("Navigation", ["Demo", "Batch"])

if rad == 'Demo':

    st.markdown("\n\n\n\n\n")
    st.markdown("\n\n\n\n\n")
    st.markdown("\n\n\n\n\n")
    st.markdown("\n\n\n\n\n")
    st.markdown("\n\n\n\n\n")

    st.header("Models Available")
    col1, col2 = st.beta_columns(2)
    col2.subheader("Pre-trained :")
    # m1, m2, m3 = st.beta_columns(3)
    col2.write("TextBlob")
    col2.write("VADER")
    col1.subheader("Custom Trained:")
    # a,b,c = st.beta_columns(3)
    col1.write("LSTM")
    col1.write("Decision Tree")
    col1.write("XGBoost")
    col1.write("Naive Bayes")
    # m3.info("Flair")

    st.header("Input comment")
    text = st.text_input("Enter comment here")

    model_choice = st.selectbox("Choose model", [
        "Text Blob", "VADER"])

    st.header("Comment analysis")

    if model_choice == "Text Blob":
        from textblob import TextBlob
        blob = TextBlob(text)
        st.write(blob.sentiment)

    if model_choice == "VADER":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(text)
        st.write(sentiment_dict)

if rad == "Batch":
    st.subheader(
        "Upload a csv file to generate the overall sentiment of each review")
    csv_file = st.file_uploader("Upload a csv file")
    if csv_file:
        df = pd.read_csv(csv_file)
        st.write(df.head())
        size = st.slider("Choose number of rows to analyse",
                         min_value=200, max_value=1000, value=500, step=100)
        reviewed_df, lstm_accuracy, xgb_accuracy, NB_accuracy, DT_accuracy = generate_reviews(
            df[0:size])
        st.write(reviewed_df.head())
        data = pd.DataFrame(reviewed_df).to_csv(index=False)
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download Results</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.markdown(href, unsafe_allow_html=True)
        st.markdown("\n\n\n\n\n")

        # colA, colB = st.beta_columns([1, 3])
        st.subheader("Accuracy")
        st.write("LSTM: ", lstm_accuracy)
        st.write("XGBoost Regressor: ", xgb_accuracy)
        st.write("Naive Bayes: ", NB_accuracy)
        st.write("Decision Tree: ", DT_accuracy)
        st.subheader("Sentiments")
        # sns.countplot(reviewed_df['lstm_review_category']).set_title(
        #     "Distribution of Reviews Category")
        # # colB.image(figg)

        # st.bar_chart())
        fig = px.bar(
            x=['Negative', 'Positive', 'Neutral'], y=reviewed_df['lstm_review_category'].value_counts())
        fig.update_traces(marker_color=['red', 'yellow', 'green'])

        st.plotly_chart(fig)
        # px.bar(reviewed_df['lstm_review_category'].value_counts(), color=reviewed_df['lstm_review_category'].value_counts(), title='Sentiment Spread')

        positive_reviews = reviewed_df.loc[reviewed_df['lstm_review_category']
                                           == 'positive', 'body'].tolist()
        negative_reviews = reviewed_df.loc[reviewed_df['lstm_review_category']
                                           == 'negative', 'body'].tolist()
        positive_reviews_bigrams = [
            " ".join(generateNGram(review, 2)) for review in positive_reviews]
        negative_reviews_bigrams = [
            " ".join(generateNGram(review, 2)) for review in negative_reviews]

        # rcParams['figure.figsize'] = 15, 20
        # fig, ax = plt.subplots(1, 2)
        # fig.subplots_adjust(wspace=1)
        st.subheader("Most common positive reviews")
        plotMostCommonWords(positive_reviews_bigrams, 'blue', 40,
                            'Positive Review Bigrams')
        # st.pyplot()
        st.subheader("Most common negative reviews")
        plotMostCommonWords(negative_reviews_bigrams, 'red', 40,
                            'Negative Review Bigrams')
        # st.pyplot()
