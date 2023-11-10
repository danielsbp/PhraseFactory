import matplotlib.pyplot as plt

# Seu dicionário de dados (substitua com seus próprios valores)
dados = {
    'Categoria A': 30,
    'Categoria B': 20,
    'Categoria C': 25,
    'Categoria D': 15,
    'Categoria E': 10
}

# Extrair as chaves (categorias) e os valores do dicionário
categorias = dados.keys()
valores = dados.values()

# Configurar as cores para as fatias da pizza
# cores = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

# Criar um gráfico de pizza
plt.figure(figsize=(6, 6))
plt.pie(valores, labels=categorias, autopct='%1.1f%%', startangle=140)

# Adicionar um título
plt.title('Gráfico de Pizza')

# Mostrar o gráfico
plt.show()