import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import *

# set default page config to wide
st.set_page_config(layout="wide")

# Title text of the app
title_txt = "Plotting intersection of curves using gradient descent"
st.markdown(f"<h3 style='text-align: center;'>{title_txt}</h1> <br><br>", unsafe_allow_html=True)

# Next row of app will have 4 columns
#   column-1 : Input for function-1
#   column-2 : Input for function-2
#   column-3 : Input for no. of points to plot
#   column-4 : Display equations of functions
col1,col2,col3,col4 = st.columns([1,2,2,4])
with col1:
    func1 = st.radio("Function-1", ('DoubleCone1', 'Sphere1', 'Cylinder1'))
    func1 = str(func1)
with col2:
    func2 = st.radio("Function-2", ('DoubleCone2', 'Sphere2', 'Cylinder2'))
    func2 = str(func2)
with col3:
    N = st.slider('Number of points to plot. More the slower', 200, 2000, 500, step=100)
with col4:
    st.write(F1_curve(X, func1))
    st.write(F2_curve(X, func2))

# Next row of app will have 2 columns
#   column-1 : Plot for input functions
#   column-2 : Plot for intersection of the functions
col1,col2 = st.columns(2)

with col1:
    data1 = generate_F1_sample(N, func1)
    data2 = generate_F2_sample(N, func2)

    x, y, z = data1[:, 0], data1[:, 1], data1[:, 2]
    trace1 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="F1", marker=dict(size=5,sizemode='diameter'))
    x, y, z = data2[:, 0], data2[:, 1], data2[:, 2]
    trace2 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="F2", marker=dict(size=5,sizemode='diameter'))

    fig = make_subplots()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600, width=1000)
    st.plotly_chart(fig, use_container_width=True, height=600, width=1000)

with col2:
    ## Plot for Intersection of Curves
    data = generate_intersection_sample(N, func1,func2)
    data = np.array(data)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    fig = go.Figure(data=[
            go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(size=5,sizemode='diameter'))]
        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600, width=800)
    st.plotly_chart(fig, use_container_width=True, height=600, width=800)



