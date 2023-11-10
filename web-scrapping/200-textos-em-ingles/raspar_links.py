import requests
from bs4 import BeautifulSoup
from rich import print

def start():

    qtd_pagina = range(1, 14)
    links = []

    for qtd in qtd_pagina:

        link = "https://aulasdeinglesgratis.net/category/110-textos-em-ingles-para-intermediarios-e-avancados/page/{}".format(qtd)

        print("Capturando links da url: {}".format(link))
        resposta = requests.get(link)
        soup = BeautifulSoup(resposta.content, "html.parser")

        tags_a = soup.find_all("a", {"class": "more-link"})
        
        for tag_a in tags_a:
            
            links.append(tag_a.attrs["href"])

    return links