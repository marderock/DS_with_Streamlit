import streamlit as st 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title('Previsão inicial de custo de franquia')

# Load the data
dados = pd.read_csv('slr12.csv', sep=';')

# Extract features and target
X = dados[['FrqAnual']]  # Extract column as DataFrame
y = dados['CusInic']     # Extract column as Series

# Convert to numeric (if needed)
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Handle missing or invalid values
dados.dropna(subset=['FrqAnual', 'CusInic'], inplace=True)
X = dados[['FrqAnual']]
y = dados['CusInic']

# Train the model
modelo = LinearRegression().fit(X, y)

# Create Streamlit layout
col1, col2 = st.columns(2)

with col1:
    st.header('Dados')
    st.table(dados.head(10))
    
with col2:
    st.header('Gráfico de Dispersão')
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Dados')
    ax.plot(X, modelo.predict(X), color='red', label='Linha de Regressão')
    ax.set_xlabel('FrqAnual')
    ax.set_ylabel('CusInic')
    ax.legend()
    st.pyplot(fig)

st.header('Valor anual da Franquia')
novo_valor = st.number_input('Insira novo valor', min_value=1.0, max_value=9999.9, value=1500.0, step=0.01)

processar = st.button('processar')

if processar:
    dados_novo_valor = pd.DataFrame([novo_valor],columns=['FrqAnual'])
    prev = modelo.predict(dados_novo_valor)
    st.header(f'Previsão de custo inicial R$: {prev[0]:.2f}')