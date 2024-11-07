import os
import json
import pandas as pd
import openai
import numpy as np
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, AzureOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain


def load_data(file_path):
    """Load JSON data from a file and return as a DataFrame and dictionary."""
    with open(file_path, 'r') as file:
        jsonData = json.load(file)
    df = pd.read_json(file_path)
    return df, jsonData

def get_credentials():
    load_dotenv()

    api_token_openai = os.getenv("AZURE_OPENAI_API_KEY")
    password_neo = os.getenv("NEO4J_PASSWORD")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")

    # azure_deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")
    # This sets a variable for the current process 
    os.environ['AZURE_OPENAI_API_KEY'] = api_token_openai
    os.environ['NEO4J_PASSWORD'] = password_neo
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint
    os.environ['AZURE_OPENAI_API_VERSION'] = azure_version

    return password_neo

def connect_neo4j(url, username, password):
    """Establish and return a Neo4j graph connection."""
    return Neo4jGraph(url=url, username=username, password=password)

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
    """Vectorize data from Neo4j and store embeddings back in the database."""
    Neo4jVector.from_existing_graph(
        embeddings=AzureOpenAIEmbeddings(model=embeddings_model),
        url=url,
        username=username,
        password=password,
        index_name=entity_type,
        node_label=entity_type,
        text_node_properties=['value'],
        embedding_node_property='embedding'
    )

# Define the entities to look for
def define_query(prompt, model="gpt-4o"):
    client = openai.AzureOpenAI(api_version='2024-02-15-preview')
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format= {
            "type": "json_object"
        },
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
        ]
    )
    return completion.choices[0].message.content

def create_embedding(text):
    embeddings_model = "text-embedding-3-large"
    client = AzureOpenAIEmbeddings(model=embeddings_model)
    result = client.embed_query(text)
    return result

def cosine_similarity(vec1, vec2):
    # Calculate the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    # Calculate the norm (magnitude) of each vector
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    # Return the cosine similarity
    return dot_product / (norm_a * norm_b)

# The threshold defines how closely related words should be. Adjust the threshold to return more or less results
def create_query(text, threshold=0.81):
    query_data = json.loads(text)
    # Creating embeddings
    embeddings_data = []
    for key, val in query_data.items():
        if key != 'product':
            embeddings_data.append(f"${key}Embedding AS {key}Embedding")
    query = "WITH " + ",\n".join(e for e in embeddings_data)
    # Matching products to each entity
    query += "\nMATCH (p:Product)\nMATCH "
    match_data = []
    for key, val in query_data.items():
        if key != 'product':
            relationship = entity_relationship_match[key]
            match_data.append(f"(p)-[:{relationship}]->({key}Var:{key})")
    query += ",\n".join(e for e in match_data)
    similarity_data = []
    for key, val in query_data.items():
        if key != 'product':
            similarity_data.append(f"cosine_similarity({key}Var.embedding, ${key}Embedding) > {threshold}")
    query += "\nWHERE "
    query += " AND ".join(e for e in similarity_data)
    query += "\nRETURN p"
    return query

def query_graph(response):
    embeddingsParams = {}
    query = create_query(response)
    query_data = json.loads(response)
    for key, val in query_data.items():
        embeddingsParams[f"{key}Embedding"] = create_embedding(val)
    result = graph.query(query, params=embeddingsParams)
    return result



def main():
    pass


if __name__ == "__main__":
    file_path = 'C:/Users/lara.scipio/Documents/KG-RAG/KG-RAG/data/amazon_product_kg.json'
    df, json_data = load_data(file_path)
    password_neo = get_credentials()
    username = 'neo4j'
    url = "neo4j+s://51371699.databases.neo4j.io"
    graph = connect_neo4j(url, username, password_neo)

    # import_data(json_data, graph)
    print("Data imported successfully")

    entities_list = df['entity_type'].unique()
    embeddings_model = "text-embedding-3-large"

    # embeddings_example = "This is an example text"
    # embedding = AzureOpenAIEmbeddings(model=embeddings_model)
    # vector = embedding.embed_query(embeddings_example)
    # print(f"Embedding for '{embeddings_example}': {vector}")
    
# This method takes text from our database, calculates embeddings and stores them back in the database.
#     vector_index = Neo4jVector.from_existing_graph(
#     AzureOpenAIEmbeddings(model=embeddings_model),
#     url=url,
#     username=username,
#     password=password_neo,
#     index_name='products',
#     node_label="Product",
#     text_node_properties=['name', 'title'],
#     embedding_node_property='embedding',
# )
    
    # for t in entities_list:
    #     vectorize_data(t, embeddings_model, url, username, password_neo)

    # print("Data is embedded")
    
    entity_types = {
    "product": "Item detailed type, for example 'high waist pants', 'outdoor plant pot', 'chef kitchen knife'",
    "category": "Item category, for example 'home decoration', 'women clothing', 'office supply'",
    "characteristic": "if present, item characteristics, for example 'waterproof', 'adhesive', 'easy to use'",
    "measurement": "if present, dimensions of the item", 
    "brand": "if present, brand of the item",
    "color": "if present, color of the item",
    "age_group": "target age group for the product, one of 'babies', 'children', 'teenagers', 'adults'. If suitable for multiple age groups, pick the oldest (latter in the list)."
}

    relation_types = {
        "hasCategory": "item is of this category",
        "hasCharacteristic": "item has this characteristic",
        "hasMeasurement": "item is of this measurement",
        "hasBrand": "item is of this brand",
        "hasColor": "item is of this color", 
        "isFor": "item is for this age_group"
    }

    entity_relationship_match = {
        "category": "hasCategory",
        "characteristic": "hasCharacteristic",
        "measurement": "hasMeasurement", 
        "brand": "hasBrand",
        "color": "hasColor",
        "age_group": "isFor"
    }

    system_prompt = f'''
    You are a helpful agent designed to fetch information from a graph database. 
    
    The graph database links products to the following entity types:
    {json.dumps(entity_types)}
    
    Each link has one of the following relationships:
    {json.dumps(relation_types)}

    Depending on the user prompt, determine if it possible to answer with the graph database.
        
    The graph database can match products with multiple relationships to several entities.
    
    Example user input:
    "Which blue clothing items are suitable for adults?"
    
    There are three relationships to analyse:
    1. The mention of the blue color means we will search for a color similar to "blue"
    2. The mention of the clothing items means we will search for a category similar to "clothing"
    3. The mention of adults means we will search for an age_group similar to "adults"
    
    
    Return a json object following the following rules:
    For each relationship to analyse, add a key value pair with the key being an exact match for one of the entity types provided, and the value being the value relevant to the user query.
    
    For the example provided, the expected output would be:
    {{
        "color": "blue",
        "category": "clothing",
        "age_group": "adults"
    }}
    
    If there are no relevant entities in the user prompt, return an empty json object.
'''

    example_queries = [
        "Which pink items are suitable for children?",
        "Help me find gardening gear that is waterproof",
        "I'm looking for a bench with dimensions 100x50 for my living room"
    ]

    # for q in example_queries:
    #     print(f"Q: '{q}'\n{define_query(q)}\n")

    example_response = '''{
    "category": "clothes",
    "color": "blue",
    "age_group": "adults"
    }'''

    result = query_graph(example_response)

    # Result
    print(f"Found {len(result)} matching product(s):\n")
    for r in result:
        print(f"{r['p']['name']} ({r['p']['id']})")
    
