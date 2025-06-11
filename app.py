import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
st.title("꽃 분류기(iris classifier)")
st.write('입력값 기반 종류 예측')

sepal_length = st.slider('sepal length(cm)',4.0,8.0,5.1)
sepal_width = st.slider('sepal width(cm)',2.0,4.5,3.5)
petal_length = st.slider('petal length(cm)',1.0,7.0,1.4)
petal_width = st.slider('petal width(cm)',0.1,2.5,0.2)

input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
prediction = model.predict(input_data)

predicted_class=prediction[0]
class_names=['Setosa','Versicolor','Virginica']

prediction = model.predict(input_data)
predicted_class=prediction[0]
class_names=['Setosa','Versicolor','Virginica']

st.subheader('예측결과:')
st.write(f"->{class_names[predicted_class]}")