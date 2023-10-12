import requests
from bs4 import BeautifulSoup
from rich import print

def start():

    qtd_pagina = range(1, 14)
    links = []

    for qtd in qtd_pagina:

        link = "https://aulasdeinglesgratis.net/category/110-textos-em-ingles-para-intermediarios-e-avancados/page/{}".format(qtd)

        print(link)
        # resposta = requests.get(link)
        # soup = BeautifulSoup(resposta.content, "html.parser")


        # tags_a = soup.find_all("a.more-link")

        
        # for tag_a in tags_a:
            
        #     links.append(tag_a.attrs["href"])
            
        #     div_table = soup.select_one("table")
        #     td = div_table.find("td")
            

        # return links