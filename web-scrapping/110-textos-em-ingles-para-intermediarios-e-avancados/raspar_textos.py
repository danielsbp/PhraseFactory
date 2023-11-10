import requests
from bs4 import BeautifulSoup
import json

dominio = "https://aulasdeinglesgratis.net/"


def start(links):
    dados_final = []

    for link in links:
        print("Passando pelo link {}".format(link))
        resposta = requests.get(dominio + link)
        soup = BeautifulSoup(resposta.content, "html.parser")
        div_table = soup.select_one("table")
        td = div_table.find("td")
        dados_final.append(td.text)


    frases = []
    for texto in dados_final:
        frases_texto = (texto.split("."))
        for frase in frases_texto:
            frase = frase.strip() + "."
            
            palavras = frase.split(" ")

            if len(palavras) < 3:
                continue                

            frases.append(frase)
    # return dados_final

    return frases

def save(dados):
    with open("textos.csv", "w", encoding="utf-8") as arquivo_de_texto:
        # json.dump(dados, arquivo_de_texto, ensure_ascii=False, indent=4)

        csv_string = ""

        for frase in dados:
            csv_string += "{}\n".format(frase)

        arquivo_de_texto.write(csv_string)
