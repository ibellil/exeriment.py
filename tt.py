
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
st.title("Welcome to Streamlit! Select Box")

selectbox = st.selectbox(
    "Select yes or no",
    ["Yes", "No"]
)
st.write(f"You selected {selectbox}")
st.title("Welcome to Streamlit! Checkbox")

checkbox_one = st.checkbox("Yes")
checkbox_two = st.checkbox("No")

if checkbox_one:
    value = "Yes"
elif checkbox_two:
    value = "No"
else:
    value = "No value selected"

st.write(f"You selected: {value}")

#Line chart in streamlit
import numpy as np
st.title("Welcome to Streamlit! Line chart")
st.write("Line Chart in Streamlit")
# 10 * 2 dimensional data
chart_data = pd.DataFrame(
    np.random.randn(10, 2),
    columns=[f"Col{i+1}" for i in range(2)]
)

st.line_chart(chart_data)
#Plotly charts in streamlit
#pip install plotly
import plotly.graph_objects as go

st.title("Welcome to Streamlit! Plotly Graphs")

fig = go.Figure(
    data=[go.Pie(
        labels=['A', 'B', 'C'],
        values=[30, 20, 50]
    )]
)
fig = fig.update_traces(
    hoverinfo='label+percent',
    textinfo='value',
    textfont_size=15
)

st.write("Pie chart in Streamlit")
st.plotly_chart(fig)
