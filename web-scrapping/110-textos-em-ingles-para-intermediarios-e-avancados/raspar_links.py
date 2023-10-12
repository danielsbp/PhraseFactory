import requests
from bs4 import BeautifulSoup
from rich import print

def start():
    link = "https://aulasdeinglesgratis.net/200-textos-em-ingles-com-traducao-e-audio/"

    resposta = requests.get(link)
    soup = BeautifulSoup(resposta.content, "html.parser")

    tag_ul = soup.select_one("#the-post > div.entry-content.entry.clearfix > ul")

    tags_a = tag_ul.find_all("a")

    links = []
    
    for tag_a in tags_a:
        
        links.append(tag_a.attrs["href"])
        
        div_table = soup.select_one("table")
        td = div_table.find("td")
        

    return links