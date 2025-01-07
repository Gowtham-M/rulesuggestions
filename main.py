from fastapi import FastAPI, Request
from fisecpakage import database, xmltojson
from typing import Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import os
from rapidfuzz.distance.Levenshtein import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
orcfilepath  = os.getenv("RULES_FILE_PATH")
bankfilepath = os.getenv("BANK_RUlES_FILE_PATH")
tzdir_path = os.getenv("TZDIR")
print(tzdir_path)
 
os.environ["TZDIR"] = tzdir_path

app = FastAPI()

@app.get("/testing")
async def root():
    try:
        docs = await database.get_db_docs()
        listofdocs = []
        for doc in docs:
            jsondoc = xmltojson.xml_string_to_json(str(doc["originalXml"]))
            listofdocs.append(listofdocs)
        return listofdocs
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error fetching data: {str(e)}")
        return {"error": "Failed to fetch data"}

@app.get("/trainrules")
async def rules():
    docs  = await database.get_db_rules()
    df = pd.DataFrame()
    df["whenvalue"] = "default_value"
    # Convert to Arrow Table and write to ORC
    try:
        os.remove(orcfilepath)
    except:
        print("issue")
    for doc in docs:
        customername = doc['customerName']
        key, value = next(iter(doc['then'][0].items()))
        df = pd.concat([df, pd.DataFrame({"whenvalue": [doc["when"][0]["ISOWhenValue"]],"whenpath":[doc["when"][0]["ISOWhenField"]],"customerName":[customername], "thenpath":key,"thenvalue":[value]})], ignore_index=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    file_path = orcfilepath
    orc.write_table(table, file_path) 
    read_table = orc.read_table(file_path)
    read_df = read_table.to_pandas()

    return {"hello":"something"}
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

@app.post("/fetchmatching")
async def fetch_matching(request: Request):
    body = await request.body()
    doc = await request.json()
    key, value = next(iter(doc['then'][0].items()))
    testdf =  pd.DataFrame({"whenvalue": [doc["when"][0]["ISOWhenValue"]],"whenpath":[doc["when"][0]["ISOWhenField"]],"customerName":doc["customerName"], "thenpath":key,"thenvalue":[value]})
    file_path = orcfilepath
    read_table = orc.read_table(file_path)
    read_df = read_table.to_pandas()
    testdf["whenthentest"] = testdf['whenvalue'] + " " + testdf["thenvalue"] + " " + read_df["customerName"]
    read_df["whenthen"] = read_df["whenvalue"] + " " + read_df['thenvalue'] + " " + read_df["customerName"]
    matches = process.extract(
    testdf['whenthentest'].iloc[0],
    read_df['whenthen'].tolist(),
    scorer=fuzz.ratio,
    score_cutoff=60,  # Adjust threshold as needed
    limit=None  # Return all matches above threshold
)
    matches_df = pd.DataFrame(matches, columns=['companyname', 'score', 'index'])
    matches_df = matches_df.sort_values('score', ascending=False)
    return matches_df

@app.post("/findrules")
async def findrules(request: Request):
    doc = await request.json()
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
    filtered_matches = df_orc[df_orc['similarity_score'] > 0.7]
    # Get top matches
    top_matches = filtered_matches.sort_values(by='similarity_score', ascending=False)

    ruleslist = []
    for index, row in top_matches.iterrows():
        doc_copy = doc
        doc_copy["when"][0]["ISOWhenValue"] = row["whenvalue"]
        doc_copy["then"][0] = {key: row["thenvalue"] for key in doc_copy["then"][0]}
        doc_copy["score"] = float(row['similarity_score']) * 100
        ruleslist.append(doc_copy)
    return ruleslist

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
    filtered_matches = df_orc[df_orc['similarity_score'] > 0.7]
    # Get top matches
    top_matches = filtered_matches.sort_values(by='similarity_score', ascending=False)
    print(top_matches[['whenvalue', 'thenvalue', 'similarity_score', 'customerName']])
    ruleslist = []
    for index, row in top_matches.iterrows():
        doc_copy = doc
        doc_copy["when"][0]["ISOWhenValue"] = row["whenvalue"]
        doc_copy["then"][0] = {key: row["thenvalue"] for key in doc_copy["then"][0]}
        doc_copy["score"] = float(row['similarity_score']) * 100
        ruleslist.append(doc_copy)
    return ruleslist