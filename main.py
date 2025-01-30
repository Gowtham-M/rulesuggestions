from fastapi import FastAPI, Request
from fisecpakage import database, xmltojson, lavenstein_distance
from typing import Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import os
from fuzzywuzzy import fuzz
from rapidfuzz.distance.Levenshtein import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import copy

load_dotenv()
orcfilepath  = os.getenv("RULES_FILE_PATH")
filerulespath = os.getenv("FILE_RUlES_FILE_PATH")
bankfilepath = os.getenv("BANK_RUlES_FILE_PATH")
tzdir_path = os.getenv("TZDIR")
print(tzdir_path)
 
os.environ["TZDIR"] = tzdir_path

app = FastAPI()

@app.get("/test")
async def test(request: Request):
    print("testing")
    return "success"

@app.get("/trainrules")
async def rules():
    docs  = await database.get_db_rules()
    cnt = 0
    df = pd.DataFrame()
    df["whenvalue"] = "default_value"
    # Convert to Arrow Table and write to ORC
    try:
        os.remove(orcfilepath)
    except:
        print("issue")
    for doc in docs:
        cnt += 1
        customername = doc['customerName']
        key, value = next(iter(doc['then'][0].items()))
        df = pd.concat([df, pd.DataFrame({"id":[str(doc['_id'])],"whenvalue": [doc["when"][0]["ISOWhenValue"]],"whenpath":[doc["when"][0]["ISOWhenField"]],"customerName":[customername], "thenpath":key,"thenvalue":[value],"ruleName":[doc['ruleName']], "ruleDescription":[doc['ruleDescription']]})], ignore_index=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    file_path = orcfilepath
    orc.write_table(table, file_path) 
    read_table = orc.read_table(file_path)
    read_df = read_table.to_pandas()
    print(read_df.whenvalue, read_df.thenvalue)
    print(read_df)

    return {"hello":"something"}


@app.get("/train_payment_rules")
async def rules():
    docs  = await database.get_db_rules()
    cnt = 0
    df = pd.DataFrame()
    df["whenvalue"] = "default_value"
    # Convert to Arrow Table and write to ORC
    try:
        os.remove(filerulespaths)
    except:
        print("issue")
    for doc in docs:
        cnt += 1
        customername = doc['customerName']
        key, value = next(iter(doc['then'][0].items()))
        df = pd.concat([df, pd.DataFrame({"id":[str(doc['_id'])],"whenvalue": [doc["when"][0]["ISOWhenValue"]],"whenpath":[doc["when"][0]["ISOWhenField"]],"customerName":[customername], "thenpath":key,"thenvalue":[value],"ruleName":[doc['ruleName']], "ruleDescription":[doc['ruleDescription']], "customerName":[doc['customerName']]})], ignore_index=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    file_path = filerulespaths
    orc.write_table(table, file_path) 
    read_table = orc.read_table(file_path)
    read_df = read_table.to_pandas()

    return {"hello":"payment rules trained"}

@app.get("/trainbankrules")
async def rules():
    docs  = await database.get_db_bank_rules()
    df = pd.DataFrame()
    df["whenvalue"] = "default_value"
    # Convert to Arrow Table and write to ORC
    try:
        os.remove(bankfilepath+ "\\bankrules.orc")
    except:
        print("issue")
    for doc in docs:
        print(doc,"68s")
        df = pd.concat([df, pd.DataFrame({"whenvalue": [doc["when"][0]["ISOWhenValue"]],"whenpath":[doc["when"][0]["ISOWhenField"]]})], ignore_index=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    file_path = bankfilepath + "bankrules.orc"
    print(file_path)
    orc.write_table(table, file_path) 
    read_table = orc.read_table(file_path)
    read_df = read_table.to_pandas()

    return {"hello":"bank rules savec"}

@app.post("/findrules")
async def findrules(request: Request):
    body = await request.json()
    doc = body.get("regulation")
    typeofreq = body.get("type")
    orc_file_path = orcfilepath
    df_orc = pd.read_orc(orc_file_path)
    key, value = next(iter(doc['then'][0].items()))
    # Example input DataFrame
    df_new = pd.DataFrame({
        'whenvalue': [doc["when"][0]["ISOWhenValue"]],
        'thenvalue': [value]
    })

    # Ensure columns match
    assert all(col in df_orc.columns for col in df_new.columns)
    # df_orc['combined'] = df_orc['whenvalue'] + " " + df_orc['thenvalue'] 
    # df_new['combined'] = df_new['whenvalue'] + " " + df_new['thenvalue'] 
    df_orc['similarity_score'] = df_orc.apply(
    lambda row: fuzz.ratio(
        f"{row['whenvalue']} {row['thenvalue']}", 
        f"{df_new['whenvalue'].iloc[0]} {df_new['thenvalue'].iloc[0]}"), axis=1)
    top_matches = df_orc.sort_values(by='similarity_score', ascending=False)
    if typeofreq == "similar":
        top_matches = top_matches[top_matches["similarity_score"] >= 70]
    if typeofreq == "unique":
        top_matches = top_matches[top_matches["similarity_score"] <= 70]
    top_matches = top_matches.sort_values(by="similarity_score", ascending=False)
    top_matches = top_matches.nlargest(1, 'similarity_score')
    print(top_matches.similarity_score)
    # ---------------------------------------------------------------------------------------

    # Calculate Levenshtein distance for each row in df_orc
    # df_orc['similarity_score'] = df_orc.apply(
    #     lambda row: lavenstein_distance.lavenstein_distance(row['whenvalue'] + " " + row['thenvalue'], df_new['whenvalue'].iloc[0] + " " + df_new['thenvalue'].iloc[0]), axis=1)
    # top_matches = df_orc.sort_values(by='similarity_score', ascending=True)
    # top_matches = top_matches[top_matches["similarity_score"] >= 50]

    # --------------------------------------------------------------------------------
    # Convert text to vectors using TF-IDF
    # vectorizer = TfidfVectorizer()
    # orc_vectors = vectorizer.fit_transform(df_orc['combined'])
    # new_vector = vectorizer.transform(df_new['combined'])

    # Compute cosine similarity
    # similarity_scores = cosine_similarity(orc_vectors, new_vector).flatten()

    # Add similarity scores to the DataFrame
    # df_orc['similarity_score'] = similarity_scores
    # filtered_matches = df_orc[df_orc['similarity_score'] >= 0.2]
    # filtered_matches = df_orc
    # Get top matches
    # print(len(filtered_matches))
    # top_matches = filtered_matches.sort_values(by='similarity_score', ascending=False)
    # top_matches = filtered_matches
    # ruleslist = []
    doc_copy = copy.deepcopy(doc) 
    for index, row in top_matches.iterrows():
        # doc_copy = copy.deepcopy(doc) 
        doc_copy['_id'] = row['id']
        doc_copy["when"][0]["ISOWhenValue"] = row["whenvalue"]
        doc_copy["then"][0] = {key: row["thenvalue"] for key in doc_copy["then"][0]}
        doc_copy["ruleName"] = row["ruleName"]
        doc_copy["ruleDescription"] = row["ruleDescription"]
        doc_copy["score"] = float(row['similarity_score']) 
        # ruleslist.append(doc_copy)
        return doc_copy

@app.post("/findbankrules")
async def findrules(request: Request):
    doc = await request.json()
    orc_file_path = "D:\\rulesML\\rulesuggestions\\orcfiles\\bankrules.orc"
    df_orc = pd.read_orc(orc_file_path)
    key, value = next(iter(doc['then'][0].items()))
    # Example input DataFrame
    df_new = pd.DataFrame({
        'whenvalue': [doc["when"][0]["ISOWhenValue"]],
        'thenvalue': [value]
    })

    # Ensure columns match
    assert all(col in df_orc.columns for col in df_new.columns)
    df_orc['combined'] = df_orc['whenvalue'] + " " + df_orc['thenvalue'] 
    df_new['combined'] = df_new['whenvalue'] + " " + df_new['thenvalue'] 

    # Convert text to vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    orc_vectors = vectorizer.fit_transform(df_orc['combined'])
    new_vector = vectorizer.transform(df_new['combined'])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(orc_vectors, new_vector).flatten()

    # Add similarity scores to the DataFrame
    df_orc['similarity_score'] = similarity_scores
    filtered_matches = df_orc[df_orc['similarity_score'] > 0.8]
    # Get top matches
    top_matches = filtered_matches.sort_values(by='similarity_score', ascending=False)
    print(top_matches[['whenvalue', 'thenvalue', 'similarity_score', 'customerName']])
    RULES
    for index, row in top_matches.iterrows():
        doc_copy = doc
        doc_copy["when"][0]["ISOWhenValue"] = row["whenvalue"]
        doc_copy["then"][0] = {key: row["thenvalue"] for key in doc_copy["then"][0]}
        doc_copy["score"] = float(row['similarity_score']) * 100
        ruleslist.append(doc_copy)
    return ruleslist


@app.post("/findpaymentrules")
async def findrules(request: Request):
    body = await request.json()
    doc = body.get("regulation")
    typeofreq = body.get("type")
    orc_file_path = filerulespaths
    df_orc = pd.read_orc(orc_file_path)
    key, value = next(iter(doc['then'][0].items()))
    # Example input DataFrame
    df_new = pd.DataFrame({
        'whenvalue': [doc["when"][0]["ISOWhenValue"]],
        'thenvalue': [value]
    })

    # Ensure columns match
    assert all(col in df_orc.columns for col in df_new.columns)
    # df_orc['combined'] = df_orc['whenvalue'] + " " + df_orc['thenvalue'] 
    # df_new['combined'] = df_new['whenvalue'] + " " + df_new['thenvalue'] 
    df_orc['similarity_score'] = df_orc.apply(
    lambda row: fuzz.ratio(
        f"{row['whenvalue']} {row['thenvalue']}", 
        f"{df_new['whenvalue'].iloc[0]} {df_new['customer'].iloc[0]}"), axis=1)
    top_matches = df_orc.sort_values(by='similarity_score', ascending=False)
    if typeofreq == "similar":
        top_matches = top_matches[top_matches["similarity_score"] >= 70]
    if typeofreq == "unique":
        top_matches = top_matches[top_matches["similarity_score"] <= 70]
    # ---------------------------------------------------------------------------------------

    # Calculate Levenshtein distance for each row in df_orc
    # df_orc['similarity_score'] = df_orc.apply(
    #     lambda row: lavenstein_distance.lavenstein_distance(row['whenvalue'] + " " + row['thenvalue'], df_new['whenvalue'].iloc[0] + " " + df_new['thenvalue'].iloc[0]), axis=1)
    # top_matches = df_orc.sort_values(by='similarity_score', ascending=True)
    # top_matches = top_matches[top_matches["similarity_score"] >= 50]

    # --------------------------------------------------------------------------------
    # Convert text to vectors using TF-IDF
    # vectorizer = TfidfVectorizer()
    # orc_vectors = vectorizer.fit_transform(df_orc['combined'])
    # new_vector = vectorizer.transform(df_new['combined'])

    # Compute cosine similarity
    # similarity_scores = cosine_similarity(orc_vectors, new_vector).flatten()

    # Add similarity scores to the DataFrame
    # df_orc['similarity_score'] = similarity_scores
    # filtered_matches = df_orc[df_orc['similarity_score'] >= 0.2]
    # filtered_matches = df_orc
    # Get top matches
    # print(len(filtered_matches))
    # top_matches = filtered_matches.sort_values(by='similarity_score', ascending=False)
    # top_matches = filtered_matches
    print(top_matches.columns)
    ruleslist = []
    for index, row in top_matches.iterrows():
        doc_copy = copy.deepcopy(doc) 
        doc_copy['_id'] = row['id']
        doc_copy["when"][0]["ISOWhenValue"] = row["whenvalue"]
        doc_copy["then"][0] = {key: row["thenvalue"] for key in doc_copy["then"][0]}
        doc_copy["ruleName"] = row["ruleName"]
        doc_copy["ruleDescription"] = row["ruleDescription"]
        doc_copy["score"] = float(row['similarity_score']) 
        ruleslist.append(doc_copy)
    print(ruleslist)
    return ruleslist
