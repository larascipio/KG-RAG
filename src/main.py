import os
import json 
import pandas as pd
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings


def load_data(file_path):
    with open(file_path, 'r') as file:
        jsonData = json.load(file)

    df =  pd.read_json(file_path)

    return df, jsonData

def get_credentials():
    load_dotenv()

    api_token_openai = os.getenv("AZURE_OPENAI_API_KEY")

    if api_token_openai is None:
        raise ValueError("API token not found.")

    password_neo = os.getenv("NEO4J_PASSWORD")
    if password_neo is None:
        raise ValueError("API token not found.")


    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
    # azure_deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")
    # This sets a variable for the current process 
    os.environ['AZURE_OPENAI_API_KEY'] = api_token_openai
    os.environ['NEO4J_PASSWORD'] = password_neo
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint
    os.environ['AZURE_OPENAI_API_VERSION'] = azure_version
    # os.environ['AZURE_DEPLOYMENT_ID'] = azure_deployment_id

    return password_neo, api_token_openai, azure_endpoint, azure_version

def connect_neo4j(url, username, password_neo):

    username = username
    password = password_neo

    graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password)

    return graph

def import_data(jsondata, graph):

    # Loop through each JSON object and add them to the db
    i = 1
    for obj in jsondata:
        print(f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}")
        i+=1
        query = f'''
            MERGE (product:Product {{id: {obj['product_id']}}})
            ON CREATE SET product.name = "{sanitize(obj['product'])}", 
                        product.title = "{sanitize(obj['TITLE'])}", 
                        product.bullet_points = "{sanitize(obj['BULLET_POINTS'])}", 
                        product.size = {sanitize(obj['PRODUCT_LENGTH'])}

            MERGE (entity:{obj['entity_type']} {{value: "{sanitize(obj['entity_value'])}"}})

            MERGE (product)-[:{obj['relationship']}]->(entity)
            '''
        graph.query(query)

    
def sanitize(text):
    text = str(text).replace("'","").replace('"','').replace('{','').replace('}', '')
    return text

def vectorize_data(entity_type, embeddings_model, url, username, password):

    vector_index = Neo4jVector.from_existing_graph(
    AzureOpenAIEmbeddings(model=embeddings_model, chunk_size=1),
    url=url,
    username=username,
    password=password,
    index_name=entity_type,
    node_label=entity_type,
    text_node_properties=['value'],
    embedding_node_property='embedding',
    )




def main():
    pass


if __name__ == "__main__":
    file_path = 'C:/Users/lara.scipio/Documents/KG-RAG/KG-RAG/data/amazon_product_kg.json'
    df, json_data = load_data(file_path)
    password_neo, openai_key, azure_endpoint, azure_version = get_credentials()
    username = 'neo4j'
    url = "neo4j+s://51371699.databases.neo4j.io"
    graph = connect_neo4j(url, username, password_neo)
    # import_data(json_data, graph)
    print("Data imported successfully")

    entities_list = df['entity_type'].unique()
    embeddings_model = "text-embedding-3-large"
    vector_index = Neo4jVector.from_existing_graph(
    AzureOpenAIEmbeddings(model=embeddings_model, chunk_size=1),
    url=url,
    username=username,
    password=password_neo,
    index_name='products',
    node_label="Product",
    text_node_properties=['name', 'title'],
    embedding_node_property='embedding',
)
    for t in entities_list:
        vectorize_data(df, embeddings_model, url, username, password_neo)
