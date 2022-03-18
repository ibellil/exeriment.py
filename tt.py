
import streamlit as st
#Widgets
value = st.slider('val')  # this is a widget
st.write(value, 'squared is', value * value)
