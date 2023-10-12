import requests
from bs4 import BeautifulSoup
from rich import print
from datetime import date

url_domain = "https://www.usatoday.com/"

usatoday = requests.get(url_domain)

html = usatoday.text

soup = BeautifulSoup(html, "html.parser")

links = soup.select("a[data-t-l]")

len(links)

links_validos = []
for link in links:
  href = link.get("href")
  if href != None:
    if "/story/" in href:
      links_validos.append(link.get("href"))



print(links_validos)

today = str(date.today())
with open("./data/" + today + ".csv", "w", encoding="utf-8") as csv_file:
  textos = []
  for link in links_validos:
    print("passando pelo link: {}".format(link))
    link = link.replace(url_domain, "")

    story = requests.get(url_domain + link)

    soup2 = BeautifulSoup(story.text, "html.parser")
    p = soup2.select("p")

    for texto in p:
      textos.append(texto.get_text())

  csv_string = ""
  for texto in textos:
    csv_string += "{}\n".format(texto)
  
  csv_file.write(csv_string)