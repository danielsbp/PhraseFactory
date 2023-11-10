
## API E BANCO DE DADOS
from flask import Flask, request, jsonify
import mysql.connector

## VALIDAÇÃO DE CORPOS DAS REQUISIÇÕES
import jsonschema
from jsonschema import validate

## PROCESSAMENTO DE LINGUAGEM NATURAL
import re
import nltk
from nltk.corpus import words

nltk.download("punkt")

from nltk import sent_tokenize

## MACHINE LEARNING E UTILITÁRIOS
import csv
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error

import language_tool_python

gramatic_checker = language_tool_python.LanguageTool("en-US")

# -========= PREPARAÇÃO DO MACHINE LEARNING =========-
valid_special_characters = "!.,ºª;-#$%&()’”`´" # caracteres especiais que são válidos em uma frase

# nltk.download('words')

def extract_special_characters(input_string):
    # Encontrar todos os caracteres que não são letras, números ou espaços
    special_chars = re.findall(r'[^a-zA-Z0-9\s]', input_string)
    return ''.join(special_chars)

def separate_words(input_string):
    # Separador de palavras de uma frase/texto

    clean_string = ''.join(char for char in input_string if char.isalnum() or char.isspace())
    return clean_string.split(" ")

def x1(phrase):

  # x1 = Verificar se possui apenas caracteres especiais válidos 
  
    special_characters = extract_special_characters(phrase)

    for char in special_characters:
        if(not char in valid_special_characters):
            return False

    return True

def x2(phrase):
    # Verificar se a quantidade de palavras é maior do que 3
    words = separate_words(phrase)
    quantity = len(words)
    return quantity >= 3


def x3(phrase):
    # x5 = Verifica se existe a situação de digito acompanhado de caracter alfabético
    pattern1 = r'\d+[a-zA-Z]'

    # ou se existe a situação de caracter alfabético acompanhado de digito
    pattern2 = r'[a-zA-Z]+\d'

    return bool(re.search(pattern1, phrase)) or bool(re.search(pattern2, phrase))

def x4(phrase):
    # x6 = Tem apenas palavras em inglês
    word_list = set(words.words())

    words_phrase = separate_words(phrase)

    cont = 0
    for word in words_phrase:
        if not word in word_list:
            cont += 1

    return cont >= 2

def x5(phrase):
    #x7 = Verificar se existe número seguido de caracter especial seguido de letra (Ex: 45ºC)
    
    pattern = r'\d[' + re.escape(valid_special_characters) + r'][a-zA-Z]'
    match = re.search(pattern, phrase)

    if match:
        return True
    else:
        return False

def x6(phrase):
    #x6 = Possuir sentido.
    errors = gramatic_checker.check(phrase)
    return len(errors) == 0
# Função para preparar entradas do perceptron

def transcribe_phrase_to_boolean_array(phrase, result = None):
    phrase_representation = [int(x1(phrase)), int(x2(phrase)), int(x3(phrase)), int(x4(phrase)), int(x5(phrase)), int(x6(phrase))]
    if(result != None):
        phrase_representation.append(int(result))

    return phrase_representation


## ABERTURA DOS DATASETS DE TREINAMENTO
dataset = []

path_valid_phrases = "../datasets/frases.csv"

## FRASES VÁLIDAS
with open(path_valid_phrases) as csv_file:
  phrases = csv.reader(csv_file)

  for phrase in phrases:
    phrase_representation = transcribe_phrase_to_boolean_array(phrase[0], 1)
    dataset.append(phrase_representation)
    
path_invalid_phrases = "../datasets/nao-frases.csv"

## FRASES INVÁLIDAS
with open(path_invalid_phrases) as csv_file:
  phrases = csv.reader(csv_file)

  for phrase in phrases:
    phrase_representation = transcribe_phrase_to_boolean_array(phrase[0], 0)
    dataset.append(phrase_representation)


## -======= COMEÇO DO TREINO: Modelo responsável por validar as frases =======-
df_dataset = pd.DataFrame(dataset)
print(df_dataset)
x = df_dataset.iloc[:,0:6].values
y = df_dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x_train, y_train, batch_size=32, epochs=20)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.7)

pred_array = 1 * y_pred.reshape(len(y_pred), 1)
test_array = y_test.reshape(len(y_test), 1)

np.concatenate([pred_array, test_array], axis=1)

cm = confusion_matrix(y_test,y_pred)
as_ = accuracy_score(y_test,y_pred)
ps = precision_score(y_test,y_pred) 
rs = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print(f"Relatório: \n - Matriz de Confusão: {cm} \n - Acurácia: {as_} \n - Precisão: {ps} \n - Recall: {rs} \n - F1-Score: {f1} \n - MSE: {mse}")

# -======= COMEÇO DO TREINO: Modelo responsável por classificar os assuntos das frases =======-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from rich import print

categories = [
    "adjectives",
    "animals_and_their_babies",
    "climate_and_seasons",
    "clothing_and_accessories",
    "colors_and_numbers",
    "days_and_months",
    "food_and_drinks",
    "houses_objects_and_parts",
    "organs_and_parts_of_human_body",
    "places_and_means_of_transport",
    "professions_and_family_members",
    "school_and_study_supplies",
    "signs_and_universe",
    "sports_and_games"
]

data_classif = {
    "texto": [],
    "categoria": []
}

for category in categories:
  print(f"Passando pela categoria {category}")
  with open("./dataset_treino_classificador/{}.csv".format(category), "r", encoding="utf8") as csv_file:

    text = csv_file.read()

    phrases = text.split("\n");
    data_classif["texto"] = data_classif["texto"] + phrases
    categories_label = [category for x in range(len(phrases))]
    data_classif["categoria"] = data_classif["categoria"] + categories_label

df = pd.DataFrame(data_classif)

X = df['texto']
y = df['categoria']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_bow, y_train)

y_pred = mlp.predict(X_test_bow)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('\nRelatório de classificação:\n', classification_rep)

# -======== COMEÇO DAS PREDIÇÕES ================-
import os
os.system("cls")

print("Começo das predições...")
datasets_input = ["./datasets_final/" + dataset + ".txt" for dataset in ["gpt", "pdfs", "webscraping"]]
to_classify_and_validate = []

for dataset_input in datasets_input:

    if dataset_input == "":
        input(f"Arquivo {dataset_input} não encontrado. Aperte ENTER para continuar.   !!!!")
        continue

    try:
        with open(dataset_input, "r", encoding="utf8") as csv_file:
            
            full_text = csv_file.read()
            texts = full_text.split("\n")
            to_classify_and_validate = to_classify_and_validate + texts

            # cont = len(texts)
            # for text in texts:
            #     to_classify_and_validate = to_classify_and_validate + sent_tokenize(text)

    except:

        input(f"Arquivo {dataset_input} não encontrado. Aperte ENTER para continuar.")
        continue

frases_validas = []
frases_invalidas = []

print("Quantidade de frases: {}".format(len(to_classify_and_validate)))

cont_phrases = 0
cont_not_phrases = 0

final_data = []
for phrase in to_classify_and_validate:
    
    phrase_reg = {
        "phrase": phrase,
        "is_phrase": False,
        "subject": ""
    }

    print(f"Validando \"{phrase}\" ...")
    phrase_bool = transcribe_phrase_to_boolean_array(phrase)        
    pred = ann.predict([phrase_bool])

    subject = mlp.predict(vectorizer.transform([phrase]))

    phrase_reg["subject"] = subject[0]

    if pred > 0.7:
        phrase_reg["is_phrase"] = True
        cont_phrases = cont_phrases + 1
    else:
        cont_not_phrases = cont_not_phrases + 1

    final_data.append(phrase_reg)

with open("dados_final.json", "w", encoding="utf-8") as arquivo:
    import json

    json.dump(final_data, arquivo, indent=4) 

print("Frases Válidas: {} \n Frases Inválidas: {}".format(len(frases_validas), len(frases_invalidas)))

# print(accuracy_score(y_test, y_pred))

# app = Flask(__name__)

# conn = mysql.connector.connect(
#     host=config.DB["host"],
#     user=config.DB["user"],
#     password=config.DB["password"],
#     database=config.DB["name"]
# )

# @app.route('/')
# def index():
#     return jsonify({
#         "name": "PhraseFactory",
#         "version": "1.0.0"
#     })

# @app.route("/categories", methods=['POST', 'PATCH', 'DELETE', 'GET'])
# def categories():

#     if request.method == "GET":

#         name = request.args.get('name')
#         id = request.args.get('id')

#         has_params = name != None or id != None

#         sql_query = "SELECT * FROM category"
    
#         if has_params:
#             sql_query += "WHERE "

#             if id:
#                 sql_query += " id_category = {}".format(id)
#             elif name:
#                 sql_query += " ct_name LIKE '%{}%'".format(name)

         
#         cursor = conn.cursor(dictionary=True)
        
#         cursor.execute(sql_query)

#         categories = cursor.fetchall()
        
#         cursor.close()

#         return jsonify(categories), 200
    
#     if request.method == "POST":
        
#         schema = {
#             "type": "object",
            
#             "properties": {
#                 "name": {"type": "string"},
#                 "description": {"type": "string"},
#                 "image": {"type": "string"},
#                 "icon": {"type": "string"}
#             },
            
#             "required": ["name", "description", "image", "icon"]
#         }

#         try:
#             data = request.get_json()

#             # Valide o corpo da requisição com o esquema definido
#             validate(data, schema)

#             name =  data["name"]
#             description = data["description"]
#             image = data["image"]
#             icon = data["icon"]

#             sql_query = "INSERT INTO category(ct_name, ct_description, ct_image, ct_icon) VALUES ('{}', '{}', '{}', '{}')".format(name, description, image, icon)
            
#             cursor = conn.cursor(dictionary=True)
        
#             cursor.execute(sql_query)

#             conn.commit()

#             success = cursor.rowcount > 0 

#             if success:
#                 return jsonify({
#                             "message": "The category was created with success.",
#                             "query": sql_query
#                         }), 201
#             else:
#                 return jsonify({
#                     "message": "Something went wrong. The category wasn't created."
#                 }), 500

        

#         except jsonschema.exceptions.ValidationError as e:
#             # Se a validação falhar, retorne uma resposta de erro
#             return jsonify({"message": "Invalid request: {}".format(e.message)}), 400
        
#     if request.method == "PATCH":
#         id = request.args.get('id')

#         sql_query = "SELECT * FROM category WHERE id_category = {}".format(id)

#         cursor = conn.cursor(dictionary=True)
#         cursor.execute(sql_query)

#         category = cursor.fetchall()

#         if len(category) == 0:
#             return {"message": "This category doesn't exist."},404
        
#         try:
#             data = request.get_json()

#             schema = {
#                 "type": "object",
#                 "properties": {
#                     "name": {"type": "string"},
#                     "description": {"type": "string"},
#                     "image": {"type": "string"},
#                     "icon": {"type": "string"}
#                 }
#             }
            
#             # Valide o corpo da requisição com o esquema definido
#             validate(data, schema)

#             fields = {
#                 "name": data.get("name"),
#                 "description": data.get("description"),
#                 "image": data.get("image"),
#                 "icon": data.get("icon")
#             }             

#             keys = list(fields.keys())

#             sets_query = ""

#             for key in keys:
#                 prefix = "ct_"

#                 if fields[key] != None:
#                     sets_query += prefix + key + " = '" + fields[key] + "',"

#             if sets_query == "":
#                 return {"message": "You need to inform one field at least."}, 400

#             sets_query = sets_query[:-1]

#             sql_query = "UPDATE category SET " + sets_query + " WHERE id_category = {}".format(id) 

#             cursor = conn.cursor(dictionary=True)
        
#             cursor.execute(sql_query)

#             conn.commit()

#             success = cursor.rowcount > 0 

#             if success:
#                 return {"message": "The category was updated with success."}, 200
#             else:
#                 return {"message": "Something went wrong. The category was not updated."}, 500


#         except jsonschema.exceptions.ValidationError as e:
#             # Se a validação falhar, retorne uma resposta de erro
#             return jsonify({"message": "Invalid request: {}".format(e.message)}), 400
        
#     if request.method == "DELETE":
#         id = request.args.get('id')

#         if id == None:
#             return {"message": "You need to inform the category's id."}, 400
        
#         sql_query = "UPDATE category SET enable = 0 WHERE id_category = {}".format(int(id))

#         cursor = conn.cursor(dictionary=True)
#         cursor.execute(sql_query)
#         conn.commit()

#         success = cursor.rowcount == 1

#         if success:
#             return {"message": "The category was deleted with success"}, 200
#         else:
#             return {"message": "This category doesn't exist."}, 400     
        

# @app.route("/phrases", methods=['POST', 'PATCH', 'DELETE', 'GET'])       
# def phrases():
#     if request.method == "GET":
#         category = request.args.get('category')
#         id = request.args.get('id')
#         text = request.args.get('text')
        
#         has_params = category != None or id != None or text != None

#         sql_query = "SELECT * FROM phrase"
    
#         if has_params:
#             sql_query += " WHERE "

#             if id:
#                 sql_query += " id_phrase = {}".format(int(id))
#             elif text:
#                 sql_query += " ph_phrase LIKE '%{}%'".format(text)
#             elif category:
#                 sql_query += " ph_category = {}".format(int(category))
        
#         cursor = conn.cursor(dictionary=True)
        
#         cursor.execute(sql_query)

#         categories = cursor.fetchall()
        
#         cursor.close()

#         return jsonify(categories), 200
    
#     if request.method == "POST":

#         schema = {
#             "type": "array",    
#             "items": {
#                 "properties": {
#                     "phrase": {"type": "string"},
#                     "category": {"type": "number"}
#                 },
#                 "required": ["phrase", "category"]
#             }

#         }

#         data = request.get_json()

#         try:
#             validate(data, schema)

#             if len(data) == 0:
#                 return {"message": "You need to inform one phrase at least."}
            
#             phrases_predictions = []
#             phrases_transcriptions = []

#             for phrase in data:
#                 phrase_transcription = transcribe_phrase_to_boolean_array(phrase["phrase"])
#                 phrases_transcriptions.append(phrase_transcription)

#             phrases_predictions = ann.predict(phrases_transcriptions)

#             return phrases_predictions, 200

#         except jsonschema.exceptions.ValidationError as e:
#             # Se a validação falhar, retorne uma resposta de erro
#             return jsonify({"message": "Invalid request: {}".format(e.message)}), 400

# @app.route("/split_phrases_from_text", methods = ["POST"])     
# def split_phrases_from_text():

#     data = request.get_json()        
    
#     text = data.get("text")

#     text = text.replace("\n", "")
#     text = text.replace("...", "[[3p]]")

#     # Divide o texto em frases usando expressões regulares
    
#     frases = re.split(r'[.!?]', text)

#     # Remove strings vazias da lista de frases resultante
#     frases = [frase.strip() for frase in frases if frase.strip()]
    
#     frases = [frase.replace("\"", "") for frase in frases]
#     frases = [frase.replace("“" "") for frase in frases]    
#     frases = [frase.replace("”", "") for frase in frases]
#     frases = [re.sub(r'\?([a-z])', r'? \1', frase) for frase in frases]

#     return frases

# app.run(port=5000, host='localhost', debug=True)