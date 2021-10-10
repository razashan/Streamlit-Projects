import streamlit as st
import pandas as pd
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier

st.write(
    """
    # Simple Iris Flowe Prediction App

    This app preducts the **Iris flower** type!!
    """
)

# define the sidebar

st.sidebar.header('User Input Parameter')


def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length', 4.4, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)

    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}

    features = pd.DataFrame(data, index=[0])

    return features


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_probs = clf.predict_proba(df)

st.subheader('Class labels and corresponding index number')
st.write(iris.target_names)

st.subheader('Predictions')
st.write(iris.target_names[prediction])

st.subheader("Predictions and Probability")
st.write(prediction_probs)

if st.button('Credit'):
    st.write(' #### Made with streamlit by Bharath')