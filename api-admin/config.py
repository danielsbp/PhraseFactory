import configparser
config = configparser.ConfigParser()

config.read("config.ini")

ADMIN_KEY = config["KEYS"]["ADMIN_KEY"]

DB = {
    "user": config["DATABASE"]["user"],
    "password": config["DATABASE"]["password"],
    "host": config["DATABASE"]["host"],
    "name": config["DATABASE"]["name"]
}