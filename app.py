import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import *

st.set_page_config(layout="wide")

col1,col2 = st.columns(2)

N = 500
data1 = np.array([list(generate_F1_sample()) for i in range(N)])
data2 = np.array([list(generate_F2_sample()) for i in range(N)])

x, y, z = data1[:, 0], data1[:, 1], data1[:, 2]
trace1 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="F1", marker=dict(
            # color=px.colors.qualitative.D3,
            size=5,sizemode='diameter'))
x, y, z = data2[:, 0], data2[:, 1], data2[:, 2]
trace2 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="F2", marker=dict(
            # color=px.colors.qualitative.D3,
            size=5,sizemode='diameter'))

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout(height=800, width=1000)
with col1:
    st.plotly_chart(fig, use_container_width=True, height = 800, width=1000)

## Plot for Intersection of Curves
data = [list(generate_sample()) for i in range(N)]
data = np.array(data)
x, y, z = data[:, 0], data[:, 1], data[:, 2]
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(
            size=5,sizemode='diameter'))])
fig.update_layout(height=800, width=800)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

with col2:
    st.plotly_chart(fig, use_container_width=True, height = 800, width=800)


st.write(F1_curve(X))
st.write(F2_curve(X))
st.write(G(X))

st.write(y_F1_prime_mat)