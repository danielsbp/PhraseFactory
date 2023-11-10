import json

with open("dados_final.json", "r") as json_file:
    dados = json.load(json_file)

    print(len(dados))
    
    contador_frase = 0
    contador_nao_frase = 0

    subject_qtd = {
        "adjectives": 0,
        "animals_and_their_babies": 0,
        "climate_and_seasons": 0,
        "clothing_and_accessories": 0,
        "colors_and_numbers": 0,
        "days_and_months": 0,
        "food_and_drinks": 0,
        "organs_and_parts_of_human_body": 0,
        "places_and_means_of_transport": 0,
        "professions_and_family_members": 0,
        "school_and_study_supplies": 0,
        "signs_and_universe": 0,
        "houses_objects_and_parts": 0,
        "sports_and_games": 0
    }


    for dado in dados:

        subject_qtd[dado["subject"]] = subject_qtd[dado["subject"]] + 1

        if dado["is_phrase"] == True:
            contador_frase = contador_frase + 1
        else:
            contador_nao_frase = contador_nao_frase + 1


    

print("Quantidade de frases: {} | Quantidade de não frases: {}".format(contador_frase, contador_nao_frase))
print("Com relação aos assuntos:")
print(subject_qtd)

import matplotlib.pyplot as plt

# Extrair as chaves (categorias) e os valores do dicionário
subject_labels = subject_qtd.keys()
values = subject_qtd.values()

# Configurar as cores para as fatias da pizza
# cores = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

# Criar um gráfico de pizza
plt.figure(figsize=(10, 2))
plt.pie(values, labels=subject_labels, autopct='%1.1f%%', startangle=140)

# Adicionar um título
plt.title('Porcentagem de frases por assunto')

# Adicione uma legenda embaixo do gráfico
legenda = plt.legend(subject_labels, title="Assuntos", loc="upper center", bbox_to_anchor=(0.5, -0.2))

plt.gca().add_artist(legenda)

# Mostrar o gráfico
plt.show()