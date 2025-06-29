import streamlit as st
import joblib as jb
import numpy as np

model = jb.load('iris_random_forest_model.joblib')

st.title('Iris Flower Classification')

# Input fields for features
sepal_length = st.slider('Sepal length (cm)', min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.slider('Sepal width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.slider('Petal length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.slider('Petal width (cm)', min_value=0.0, max_value=10.0, value=0.2)

if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    target_names = ['setosa', 'versicolor', 'virginica']
    st.success(f'Predicted class: {target_names[prediction]}')