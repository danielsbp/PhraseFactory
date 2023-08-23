import mysql.connector
import config

db = mysql.connector.connect(
    host=config.DB["host"],
    user=config.DB["user"],
    password=config.DB["password"],
    database=config.DB["name"]
)

