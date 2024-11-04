import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from string import punctuation
import pickle
import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')
PUNCT = list(punctuation)
SW = stopwords.words("english")
Features_count = 6

def vectorize(w, scaled_position):
    v = np.zeros(Features_count).astype(np.float32)
    title = 0
    allcaps = 0
    sw = 0
    punct = 0

    # If first character in uppercase
    if w[0].isupper():
        title = 1
    # All characters in uppercase
    if w.isupper():
        allcaps = 1

    # Is stopword
    if w.lower() in SW:
        sw = 1
    
    # Is punctuation
    if w in PUNCT:
        punct = 1

    return [title, allcaps, len(w), sw, punct, scaled_position]


# To perform inference
def infer(model, scaler, s): 
    tokens = word_tokenize(s)
    features = []
    l = len(tokens)
    for i in range(l):
        f = vectorize(w = tokens[i], scaled_position = (i/l))
        features.append(f)
    features = np.asarray(features, dtype = np.float32)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    return pred, tokens, features


nei_model = pickle.load(open("nei_model.sav", 'rb'))
scaler_model = pickle.load(open("scaler_model.sav", 'rb'))

st.title("Named-Entity Identification")
st.text("Group: Chetan, Harshvivek, Udhay")

input = st.text_input("Enter input string here: ")

if st.button("Process Text"):
    st.write("Output: ")
    pred, tokens, features = infer(nei_model, scaler_model, input)

    o = ''
    for w, p in zip(tokens, pred):
        o += w + '_' + str(int(p)) + ' '

    st.write(o)



