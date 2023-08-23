from fastapi import FastAPI
import connect


# Importando as classes
from classes.category import Category


# Criar uma instância do FastAPI
app = FastAPI()

@app.get("/")
def init():
    return {"name": "PhraseFactory Admin", "version": "1.0.0"}

@app.get("/category/")
def getCategory(id: int = None, name: str = None, cursor = None):
    cursor = connect.db.cursor()

    try:
        query = "SELECT * FROM category"

        if(id != None):
            query = query + " id_category = " + id
        elif(name != None):
            query = query + " ct_name LIKE '%"+name+"%'"

        cursor.execute(query)

        result = []
        
        for id, name, description, image, icon in cursor:
            category_obj = Category(id, name, description, image, icon)
            result.append(category_obj)

        return result
    except:
        return []
 