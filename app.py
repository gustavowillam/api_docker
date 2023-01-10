#carregando as bibliotecas
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model


# loading the trained model.
model = load_model('datalake/curated/model')

# carregando uma amostra dos dados.
dataset = pd.read_csv('datalake/curated/dataset.csv') 


# carregando o cluster k-means
cluster = joblib.load('datalake/curated/cluster.joblib')


#print (dataset.head())

# título
st.title("App for prediction turnover of employee")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de Predição de permanência de funcionários.")

# imprime o conjunto de dados usado
st.dataframe(dataset.head())

# grupos de empregados.
kmeans_colors = ['green' if c == 0 else 'red' if c == 1 else 'blue' for c in cluster.labels_]

st.sidebar.subheader("Defina os atributos do empregado para predição de turnover")

# mapeando dados do usuário para cada atributo
satisfaction = st.sidebar.number_input("satisfaction", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("evaluation", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("averageMonthlyHours", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("yearsAtCompany", value=dataset["yearsAtCompany"].mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Classificação")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["satisfaction"] = [satisfaction]
    data_teste["evaluation"] =	[evaluation]    
    data_teste["averageMonthlyHours"] = [averageMonthlyHours]
    data_teste["yearsAtCompany"] = [yearsAtCompany]
    
    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = predict_model(model, data=data_teste)
    
    st.write(result)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter( x="satisfaction"
                ,y="evaluation"
                ,data=dataset[dataset.turnover==1],
                alpha=0.25,color = kmeans_colors)

    plt.xlabel("Satisfaction")
    plt.ylabel("Evaluation")

    plt.scatter( x=cluster.cluster_centers_[:,0]
                ,y=cluster.cluster_centers_[:,1]
                ,color="black"
                ,marker="X",s=100)
    
    plt.scatter( x=[satisfaction]
                ,y=[evaluation]
                ,color="yellow"
                ,marker="X",s=300)

    plt.title("Grupos de Empregados - Satisfação vs Avaliação.")
    plt.show()
    st.pyplot(fig) 