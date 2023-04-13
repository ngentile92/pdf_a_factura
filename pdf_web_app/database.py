from deta import Deta
import os
from dotenv import load_dotenv

def main():
    load_dotenv(".env")

    DATA_KEY = os.getenv("DATA_KEY")
    print(f"DATA_KEY: {DATA_KEY}")

    deta = Deta(DATA_KEY)

    db = deta.Base("pdf_web_app")

    def insert_data(data):
        db.put(data)

    def fetch_data():
        res = db.fetch()
        return res.items

    def get_file_data(filename):
        res = db.fetch({"nombre_factura": filename})
        return res.items