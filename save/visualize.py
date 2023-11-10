
## API E BANCO DE DADOS
from flask import Flask, request, jsonify
import mysql.connector

## VALIDAÇÃO DE CORPOS DAS REQUISIÇÕES
import jsonschema
from jsonschema import validate

## PROCESSAMENTO DE LINGUAGEM NATURAL
import re
## MACHINE LEARNING E UTILITÁRIOS
import csv
import pandas as pd
import numpy as np
# print(cm)
import nltk
import os

nltk.download("punkt")

from nltk import sent_tokenize

import language_tool_python

gramatic_checker = language_tool_python.LanguageTool("en-US")

datasets_input = []

# PDFs

datasets_input = datasets_input + ["../pdf-collector/pdfs_dataset/" + str(x+1) + ".csv" for x in range(10)]

# Datasets feitos com LLM GPT-3.5


gpt_datasets = ["adjectives", "animals_and_their_babies", "climate_and_seasons", "clothing_and_accessories", "colors_and_numbers", "days_and_months", "food_and_drinks", "houses_objects_and_parts", "places_and_means_of_transport", "professions_and_family_members", "school_and_study_supplies", "signs_and_universe", "sports_and_games"]

datasets_input = datasets_input + ["../datasets/gpt/" + gpt_dataset + ".csv" for gpt_dataset in gpt_datasets]


# Dataset do Perceptron

datasets_input.append("../datasets/frases.csv")

# Datasets dos web scraping

usa_today_dates = [
    "2023-10-10",
    "2023-10-11",
    "2023-10-12",
    "2023-10-13",
    "2023-10-16",
    "2023-10-17",
    "2023-10-19",
    "2023-10-20",
    "2023-10-21",
    "2023-10-22"
]

datasets_input = datasets_input + ["../web-scrapping/usa-today/data/"+usa_today_date + ".csv" for usa_today_date in usa_today_dates] 

phrases_por_dataset = {
    "web-scraping": [],
    "gpt": [],
    "pdfs": []
}


to_insert = []

for dataset_input in datasets_input:

    if dataset_input == "":
        input(f"Arquivo {dataset_input} não encontrado. Aperte ENTER para continuar.   !!!!")
        continue

    try:
        with open(dataset_input, "r", encoding="utf8") as csv_file:

            full_text = csv_file.read()
            texts = full_text.split("\n")
            to_insert = to_insert + texts

            qtd_texts = len(texts)
            cont= 0
            for text in texts:
                to_insert = to_insert + sent_tokenize(text)

                if "gpt" in dataset_input:
                    phrases_por_dataset["gpt"] = phrases_por_dataset["gpt"] + sent_tokenize(text) 
                elif "web-scrapping" in dataset_input:
                    phrases_por_dataset["web-scraping"] = phrases_por_dataset["web-scraping"] + sent_tokenize(text) 
                elif "pdf" in dataset_input:
                    phrases_por_dataset["pdfs"] = phrases_por_dataset["pdfs"] + sent_tokenize(text)
                else:
                    phrases_por_dataset["gpt"] = phrases_por_dataset["gpt"] + sent_tokenize(text) 
                
                cont = cont + 1
                
                # saida = "Arquivo: {} | {} de {} registros...".format(dataset_input, cont, qtd_texts)
                # print(saida)
                
            print(f"Arquivo {dataset_input} finalizado!")
    except:

        input(f"Arquivo {dataset_input} não encontrado. Aperte ENTER para continuar.")
        continue
    
def salvarArquivoCSV(nome, arquivo):
    print("Salvando {}".format(nome))
    csv_string = ""
    
    for phrase in phrases_por_dataset[nome]:

        is_trash = False

        is_trash = ("https://" in phrase) or (len(phrase.split(" ")) <= 2) or (len(gramatic_checker.check(phrase)) > 0)
    
        if not is_trash:        
            csv_string = csv_string + phrase + "\n"

    arquivo.write(csv_string)


with open("./datasets_final/webscraping.txt", "w", encoding="utf8") as csv_file:    
    salvarArquivoCSV("web-scraping", csv_file)

with open("./datasets_final/gpt.txt", "w", encoding="utf8") as csv_file:
    salvarArquivoCSV("gpt", csv_file)

with open("./datasets_final/pdfs.txt", "w", encoding="utf8") as csv_file:
    salvarArquivoCSV("pdfs", csv_file)

print("PDFs: {} | Web Scraping: {} | GPT: {}".format(len(phrases_por_dataset["pdfs"]), len(phrases_por_dataset["web-scraping"]), len(phrases_por_dataset["gpt"])))

