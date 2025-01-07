import pymongo
from dotenv import load_dotenv
import os
load_dotenv()

dbpath  = os.getenv("DB_PATH")

async def get_db_docs():
    client = pymongo.MongoClient(dbpath)
    # Access a database
    db = client["dataknol-fisec"]
    # Access a collection
    collection = db["paymentfiles"]
    # Find all documents
    documents = collection.find()
    return documents

async def get_db_rules():
    client = pymongo.MongoClient(dbpath)
    db = client["dataknol-fisec"]
    collection = db["stpconfigurations"]
    customer = "NextGen Biotech"
    documents = collection.find()
    return documents

async def get_db_bank_rules():
    print(dbpath)
    client = pymongo.MongoClient(dbpath)
    db = client["dataknol-fisec"]
    collection = db["bankrules"]
    documents = collection.find()
    print(documents)
    return documents