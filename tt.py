
import streamlit as st
#Widgets
value = st.slider('val')  # this is a widget
st.write(value, 'squared is', value * value)
import pandas as pd
st.title("Welcome to Streamlit! DataFrame")
st.write("Our first DataFrame")
st.write(
  pd.DataFrame({
      'A': [1, 2, 3, 4],
      'B': [5, 6, 7, 8]
    })
)


