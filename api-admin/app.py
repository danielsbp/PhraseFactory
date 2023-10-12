## API E BANCO DE DADOS
from flask import Flask, request, jsonify
import mysql.connector
import config

## VALIDAÇÃO DE CORPOS DAS REQUISIÇÕES
import jsonschema
from jsonschema import validate

## PROCESSAMENTO DE LINGUAGEM NATURAL
import re
import nltk
from nltk.corpus import words

## MACHINE LEARNING E UTILITÁRIOS
import csv
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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


## -======= COMEÇO DO TREINO =======-
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
y_pred = (y_pred > 0.5)

pred_array = 1 * y_pred.reshape(len(y_pred), 1)
test_array = y_test.reshape(len(y_test), 1)

np.concatenate([pred_array, test_array], axis=1)

cm = confusion_matrix(y_test,y_pred)

print(cm)

while True:
    phrase = input("Digite uma frase para ser avaliada: ")
    phrase_bool = transcribe_phrase_to_boolean_array(phrase)
    pred = ann.predict([phrase_bool])
    print(phrase_bool)
    print(pred)

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