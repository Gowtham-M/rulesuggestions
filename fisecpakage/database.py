import pymongo

async def get_db_docs():
    client = pymongo.MongoClient("mongodb+srv://Sira:pXNYDkq1VvDy3ztf@cluster0.moo8k.mongodb.net/dataknol-fisec?retryWrites=true&w=majoritys")
    # Access a database
    db = client["dataknol-fisec"]
    # Access a collection
    collection = db["paymentfiles"]
    # Find all documents
    documents = collection.find()
    return documents

async def get_db_rules():
    client = pymongo.MongoClient("mongodb+srv://Sira:pXNYDkq1VvDy3ztf@cluster0.moo8k.mongodb.net/dataknol-fisec?retryWrites=true&w=majoritys")
    db = client["dataknol-fisec"]
    collection = db["stpconfigurations"]
    customer = "NextGen Biotech"
    documents = collection.find()
    return documents