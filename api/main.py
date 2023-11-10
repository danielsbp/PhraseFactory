from flask import Flask, request, jsonify
import mysql.connector
import configparser

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

word_list = set(words.words())

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

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('config.ini')

db_config = config['database']

# Configuração do banco de dados
db = mysql.connector.connect(
    host=db_config['host'],
    user=db_config['user'],
    password=db_config['password'],
    database=db_config['database']
)
cursor = db.cursor()

# Rota para obter frases com filtros
@app.route('/phrases', methods=['GET'])
def get_filtered_phrases():
    filter_by_phrase = request.args.get('phrase')
    filter_by_word = request.args.get('word')
    filter_by_subject = request.args.get('subject')

    query = "SELECT p.id_phrase, p.ph_phrase, s.sb_name, w.wd_word FROM phrase AS p"
    query += " JOIN subject AS s ON p.ph_subject = s.id_subject"
    query += " LEFT JOIN phrase_word AS pw ON p.id_phrase = pw.phwd_phrase"
    query += " LEFT JOIN word AS w ON pw.phwd_word = w.id_word"
    where_conditions = []

    if filter_by_phrase:
        where_conditions.append(f"p.ph_phrase LIKE '%{filter_by_phrase}%'")

    if filter_by_word:
        where_conditions.append(f"w.wd_word LIKE '%{filter_by_word}%'")

    if filter_by_subject:
        where_conditions.append(f"s.sb_name LIKE '%{filter_by_subject}%'")

    if where_conditions:
        query += " WHERE " + " AND ".join(where_conditions)

    cursor.execute(query)
    results = cursor.fetchall()

    filtered_phrases = []
    for row in results:
        phrase_dict = {
            "id_phrase": row[0],
            "ph_phrase": row[1],
            "sb_name": row[2],
            "wd_word": row[3]
        }
        filtered_phrases.append(phrase_dict)

    return jsonify({"phrases": filtered_phrases})


def classify_valid_phrase(phrase): 
    phrase_bool = transcribe_phrase_to_boolean_array(phrase)        
    pred = ann.predict([phrase_bool])

    return pred > 0.7

def classify_subject_phrase(phrase):
    predicao = mlp.predict(vectorizer.transform([phrase])) 
    
    return predicao[0]

def add_unique_words(phrase_words):
    unique_words = set(phrase_words)
    for word in unique_words:
        cursor.execute("INSERT IGNORE INTO word (wd_word) VALUES (%s)", (word,))
        db.commit()

# Função para mapear as palavras de uma frase na tabela phrase_word
def map_words_to_phrase(phrase_id, phrase_words):
    for order, word in enumerate(phrase_words, 1):
        cursor.execute("SELECT id_word FROM word WHERE wd_word = %s", (word,))
        word_id = cursor.fetchone()
        if word_id is not None:
            cursor.execute("INSERT INTO phrase_word (phwd_word, phwd_phrase, `order`) VALUES (%s, %s, %s)",
                           (word_id[0], phrase_id, order))
            db.commit()

def is_word_in_list(word):
    return word in word_list

# Função para verificar se uma frase já existe no banco
def does_phrase_exist(phrase):
    cursor.execute("SELECT COUNT(*) FROM phrase WHERE ph_phrase = %s", (phrase,))
    count = cursor.fetchone()[0]
    return count > 0

subjects_index = {
    "adjectives": 1,
    "animals_and_their_babies": 2,
    "climate_and_seasons": 3,
    "clothing_and_accessories": 4,
    "colors_and_numbers": 5,
    "days_and_months": 6,
    "food_and_drinks": 7,
    "houses_objects_and_parts": 8,
    "organs_and_parts_of_human_body": 9,
    "places_and_means_of_transport": 10,
    "professions_and_family_members": 11,
    "school_and_study_supplies": 12,
    "signs_and_universe": 13,
    "sports_and_games":14
}
# Rota para adicionar uma nova frase
@app.route('/phrases', methods=['POST'])
def add_phrases():
    

    ph_subject = 0 # Só pra não bugar.

    data = request.json

    if isinstance(data, list):

        qtt_data = len(data)
        qtt_added = 0

        # Se a entrada for uma lista, adicionamos várias frases
        for item in data:
            ph_phrase = item.get("ph_phrase")
            ph_phrase_clean = re.sub(r'[^\w\s]', '', ph_phrase).lower()

            words = ph_phrase_clean.split()  # Divide a frase em palavras

            print(words)
            is_phrase = classify_valid_phrase(ph_phrase)

            if not is_phrase:
                continue

            ph_subject = subjects_index[classify_subject_phrase(ph_phrase)]
            
            qtt_added = qtt_added + 1

            if not does_phrase_exist(ph_phrase):
                
                cursor.execute("INSERT INTO phrase (ph_subject, ph_phrase) VALUES (%s, %s)", (ph_subject, ph_phrase))
                db.commit()
                
                phrase_id = cursor.lastrowid

                cursor.fetchall()

                for word in words:
                    if is_word_in_list(word):
                        cursor.execute("INSERT IGNORE INTO word (wd_word) VALUES (%s)", (word,))
                        db.commit()
                        cursor.execute("SELECT id_word FROM word WHERE wd_word = %s", (word,))
                        
                        result = cursor.fetchone()

                        if result is not None:
                            word_id = result[0]
                        else:
                            word_id = 0
                    else:
                        word_id = 0

                map_words_to_phrase(phrase_id,words)
        return jsonify({"message": "{} de {} frases adicionadas com sucesso!".format(qtt_added, qtt_data)}), 201
    elif isinstance(data, dict):
        # Se a entrada for um dicionário, adicionamos uma única frase

        
        ph_phrase = data.get("ph_phrase")
        
        ph_phrase_clean = re.sub(r'[^\w\s]', '', ph_phrase).lower()
        words = ph_phrase_clean.split()  # Divide a frase em palavras

        print(words)

        is_phrase = classify_valid_phrase(ph_phrase)

        if not is_phrase:
            return jsonify({"error": "A frase apresentada não foi aceita pelo modelo MLP."}), 400
        
        ph_subject = subjects_index[classify_subject_phrase(ph_phrase)] 

        if not does_phrase_exist(ph_phrase):
            for word in words:
                if is_word_in_list(word):
                    cursor.execute("INSERT IGNORE INTO word (wd_word) VALUES (%s)", (word,))
                    db.commit()
                    cursor.execute("SELECT id_word FROM word WHERE wd_word = %s", (word,))
                    result = cursor.fetchone()
                    if result is not None:
                        word_id = result[0]
                    else:
                        word_id = 0
                        
                else:
                    word_id = 0


            cursor.execute("INSERT INTO phrase (ph_subject, ph_phrase) VALUES (%s, %s)", (ph_subject, ph_phrase))
            db.commit()
            phrase_id = cursor.lastrowid

            cursor.fetchall()

            map_words_to_phrase(phrase_id, words)

            return jsonify({"message": "Frase adicionada com sucesso!"}), 201
    else:
        return jsonify({"error": "Entrada inválida. Deve ser uma única frase ou uma lista de frases"}), 400
    

if __name__ == '__main__':
    app.run(debug=True)
