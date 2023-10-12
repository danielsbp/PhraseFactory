from fileinput import filename
import re
import nltk
import json
from rich import print

with open("textos.json", "r", encoding="utf-8") as arquivojson:
    textos = json.loads(arquivojson.read())

    token = []
    for texto in textos:
        
        palavras = re.findall("\w+", texto.replace("’", ''))
        for palavra in palavras:
            token.append(palavra.lower())

    nlp_words=nltk.FreqDist(token)

    filename = "frequenciaPalavras.json"
    with open(filename, "w", encoding="utf-8") as frequencia_file:
        json.dump(nlp_words.most_common(), frequencia_file, ensure_ascii=False, indent=4)


    print("Frequências das palavras disponível no arquivo " + filename)