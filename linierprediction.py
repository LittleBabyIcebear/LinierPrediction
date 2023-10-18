import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import random
import csv
import streamlit as st
import pandas as pd

st.title("Linier Prediction Signal Program 🔮")
st.sidebar.subheader("Input the value of auto correlation")
waktu = []
x = []
#Reading dataset 
file_path= "Print_14_v2_PCG_RV.txt"
with open(file_path) as file:
    lines = csv.reader(file, delimiter='\t')
    for row in lines:
        waktu.append(float(row[0]))
        x.append(float(row[1]))

plotly_original_signal= go.Figure()

plotly_original_signal.add_trace(go.Scatter(x=waktu, y=x, mode='lines', name='Toe Signal', line=dict(color='orange')))
# Customize the layout
plotly_original_signal.update_layout(
    xaxis_title="Time [s]",
    yaxis_title="Voltage [mV]",
    title="Original Signals",
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)
st.write("Lenght data of the signal: ", len(x))
st.plotly_chart(plotly_original_signal)

# Assuming JumlahTimeLag, X, Ndata, rxx, and Listbox1 are defined appropriately
input_autocorrelation = st.sidebar.number_input("Input Auto Correlation", value=4)

if st.sidebar.button("Run"):
    JumlahTimeLag = input_autocorrelation + 1
    rxx = np.zeros(JumlahTimeLag)
    for L in range(JumlahTimeLag):
        sum_val = 0.0
        for n in range(len(x)):
            if n - L >= 0:
                sum_val += x[n] * x[n - L]
        rxx[L] = sum_val
    st.subheader("Compute all Matrix")
    df = pd.DataFrame({'rxx': rxx})
    

    UkuranMatrik1 = JumlahTimeLag -1
    Matrix1 = np.zeros((UkuranMatrik1, UkuranMatrik1))

    for i in range(UkuranMatrik1):
        for k in range(UkuranMatrik1):
            Matrix1[i, k] = rxx[i - k]

    Matrix1_inv = np.linalg.inv(Matrix1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Auto Correlation rxx[L]")
        st.dataframe(df)
    with col2:
        st.write("Matrix rxx")
        st.write(Matrix1)
    with col3:
        st.write("Inverse Matrix rxx")
        st.write(Matrix1_inv)

    # koefisien prediction
    UkuranMatrik2 = JumlahTimeLag -1
    Matrix2 = np.zeros((1, UkuranMatrik2))
    a = np.zeros(UkuranMatrik2)

    for i in range(UkuranMatrik2):
        HasilKaliMatrik = 0.0
        for k in range(UkuranMatrik2):
            HasilKaliMatrik += Matrix1_inv[k, i] * rxx[k]
        a[i] = HasilKaliMatrik

    # for i, val in enumerate(a):
    #     print(f'a[{i}] = {val:.5f}')
    st.subheader("Predictor Coefficient and Error")

    colom1, colom2 = st.columns(2)
    with colom1:
        st.write("Predictor Coefficient a[k]")
        df_a = pd.DataFrame({"a[k]": a})
        st.write(df_a)

    # invers a
    a_inv = -a

    # e(m) menggunakan Invers Filtering
    e = np.zeros(len(x))

    for i in range(len(x)):
        x_hat = 0
        for k in range(0, JumlahTimeLag-1):
            x_hat += a_inv[k] * x[i - k]
        e[i] = x[i] + x_hat
    with colom2: 
        st.write("Error Inverse Filtering")
        df1 = pd.DataFrame({'error': e})
        st.write(df1)

    #predict sinyal
    x_hat = np.zeros(len(x))

    for i in range(-JumlahTimeLag, 0):
        x_hat[i] = 0

    for i in range(len(x)):
        sum2 = 0.0
        for k in range(0, JumlahTimeLag-1):
            sum2 += a[k] * x[i - k]
        x_hat[i] = sum2



    Ndata = len(x)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=waktu, y=x, mode='lines', name='Original Signal', line=dict(color='orange')))
    # Customize the layout
    fig.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Voltage [mV]",
        title="Original Signal",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    # Show the Plotly figure
    st.plotly_chart(fig)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=waktu, y=x_hat, mode='lines', name='Prediction Signal', line=dict(color='blue')))
    # Customize the layout
    fig1.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Voltage [mV]",
        title="Prediction Signal",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    # Show the Plotly figure
    st.plotly_chart(fig1)

    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(x=waktu, y=x, mode='lines', name='Original Signal', line=dict(color='orange')))
    fig_error.add_trace(go.Scatter(x=waktu, y=e, mode='lines', name='Error Signal', line=dict(color='red')))
    # Customize the layout
    fig_error.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Voltage [mV]",
        title="Error and Original Signal",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    # Show the Plotly figure
    st.plotly_chart(fig_error)

    fig_rekonstruksi = go.Figure()
    fig_rekonstruksi.add_trace(go.Scatter(x=waktu, y=x_hat+e, mode='lines', name='Original Signal', line=dict(color='green')))

    fig_rekonstruksi.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Voltage [mV]",
        title="Linier Prediction Result",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    # Show the Plotly figure
    st.plotly_chart(fig_rekonstruksi)

