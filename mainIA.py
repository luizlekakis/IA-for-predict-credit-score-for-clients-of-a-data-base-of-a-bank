# %% [markdown]
# # Projeto Python IA: Inteligência Artificial e Previsões
# 
# ### Case: Score de Crédito dos Clientes
# 
# Você foi contratado por um banco para conseguir definir o score de crédito dos clientes. Você precisa analisar todos os clientes do banco e, com base nessa análise, criar um modelo que consiga ler as informações do cliente e dizer automaticamente o score de crédito dele: Ruim, Ok, Bom
# 
# Arquivos da aula: https://drive.google.com/drive/folders/1FbDqVq4XLvU85VBlVIMJ73p9oOu6u2-J?usp=drive_link

# %%
# Passo a passo
# Passo 0: Entender o desafio e a empresa
# Passo 1: Importar a base de dados


# pacotes de codigos = bibliotecas
# pandas scikit-learn 
 
import pandas as pd

tabela = pd.read_csv("clientes.csv") 

print(tabela)

# %%
# Passo 2: Preparar a base de dados para a inteligência artificial

print(tabela.info())

# int - numero inteiro
# float - numero com casa decimal
# object - textos 

# score de credito

# Good = Bom
# Standart = Medio
# Poor = Ruim

# %%
# Profissao

# mecanico - 1
# professor - 2 
# artista - 3
# advogado - 4
# medico - 5
# bombeiro - 6

# Label Encoder

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])


# mix_credito

codificador2 = LabelEncoder()
tabela["mix_credito"] = codificador2.fit_transform(tabela["mix_credito"])

# comportamento_pagamento

codificador3 = LabelEncoder()
tabela["comportamento_pagamento"] = codificador3.fit_transform(tabela["comportamento_pagamento"])

print(tabela.info())

# %%
# y é quem eu quero prever e x quem eu quero usar de base para a previsão

y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split (x, y, test_size= 0.3) 




# %%
# Passo 3: Criar um modelo de IA -> Nota/Score de Credito: Bom, Ok ou Ruim
# Arvore de decisao -> RandomForest
# Vizinhos Proximos -> (Nearest Neighbors) -> KNN

# importa a IA
  
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# cria a IA

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# treina a IA

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

# %%
# Passo 4: Avaliar qual o melhor modelo de IA

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_teste, previsao_arvoredecisao))
print(accuracy_score(y_teste, previsao_knn))

# %%
# Passo 5: Usar novas previsões
# modelo_arvoredecisao

tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
print(tabela_novos_clientes)

tabela_novos_clientes["profissao"] = codificador.fit_transform(
    tabela_novos_clientes["profissao"])

tabela_novos_clientes["mix_credito"] = codificador2.fit_transform(
    tabela_novos_clientes["mix_credito"])

tabela_novos_clientes["comportamento_pagamento"] = codificador3.fit_transform(
    tabela_novos_clientes["comportamento_pagamento"])

previsao = modelo_arvoredecisao.predict(tabela_novos_clientes)

print(previsao)


