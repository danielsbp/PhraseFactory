import pdfplumber
from rich import print
import csv
import nltk

nltk.download("punkt")

from nltk import sent_tokenize

# pdf_files_names = [str(x+1) + ".pdf" for x in range(10)]
pdf_files_names = ["hp.pdf"]

path = "./pdfs/"
path_datasets = "./pdfs_dataset/"

for pdf_file_name in pdf_files_names:

    print("Abrindo {}...".format(pdf_file_name))

    # Abra o arquivo PDF em modo de leitura binária
    with pdfplumber.open(path+pdf_file_name) as pdf:
        print("Arquivo aberto...")
        # Inicializa uma string para armazenar o texto extraído
        text = ''

        frases = []
        frases_validas = []

        # Itera através de todas as páginas do PDF
        for page in pdf.pages:
            text += page.extract_text()

        
        text = text.replace("\n", " ")

        print("Texto extraído...")

        frases = frases + sent_tokenize(text)
        
        print("Frases extraidas do texto...")

        dataset_file_path = path_datasets+pdf_file_name.replace(".pdf", ".csv")

        with open(dataset_file_path, "w", encoding="utf8") as csv_file:

            print("Criando/Preenchendo {}".format(dataset_file_path))

            csv_string = ""

            for frase in frases:
                csv_string += frase + "\n"

            csv_file.write(csv_string)
            print("Frases salvas!")
    