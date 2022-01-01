import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import *

st.set_page_config(layout="wide")

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
    # st.write(G(X,func1,func2))

col1,col2 = st.columns(2)

data1 = generate_F1_sample(N, func1)
data2 = generate_F2_sample(N, func2)

x, y, z = data1[:, 0], data1[:, 1], data1[:, 2]
trace1 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="F1", marker=dict(size=5,sizemode='diameter'))
x, y, z = data2[:, 0], data2[:, 1], data2[:, 2]
trace2 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="F2", marker=dict(size=5,sizemode='diameter'))

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout(height=600, width=1000)
with col1:
    st.plotly_chart(fig, use_container_width=True, height=600, width=1000)

## Plot for Intersection of Curves
data = generate_intersection_sample(N, func1,func2)
data = np.array(data)
x, y, z = data[:, 0], data[:, 1], data[:, 2]
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(
            size=5,sizemode='diameter'))])
fig.update_layout(height=600, width=800)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

with col2:
    st.plotly_chart(fig, use_container_width=True, height=600, width=800)


