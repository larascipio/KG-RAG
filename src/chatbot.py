import pandas as pd
import json
import os
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_openai import (AzureOpenAIEmbeddings,
                               AzureChatOpenAI)
from langchain.agents import (AgentExecutor, 
                              create_tool_calling_agent)
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Neo4jVector
from langchain import hub

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


# ------------------------- DATA PROCESSING -------------------------
def load_data(file_path: str):
    """
    Load JSON data from a file and return as a DataFrame and dictionary.
    """
 
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.read_json(file_path)
    return df, data

def import_data(jsondata, graph):
    """
    Import JSON data into the Neo4j database.
    """
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

def get_credentials():
    """
    Load environment variables from .env file and return necessary credentials.
    """
    load_dotenv()
 
    credentials = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
 
    if not all(credentials.values()):
        raise ValueError("Missing one or more environment variables in .env file.")
 
    return credentials

def connect_neo4j(url: str, username: str, password: str):
    """
    Establish and return a Neo4j graph connection.
    """
    try:
        return Neo4jGraph(url=url, username=username, password=password)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Neo4j: {e}")
   
def sanitize(text: str) -> str:
    """
    Clean and sanitize a string to prevent injection or query errors.
    """
    return str(text).replace("'", "").replace('"', "").replace("{", "").replace("}", "")

def vectorize_data(entity_type, embeddings_model, url, username, password):
    """
    Vectorize data from Neo4j and store embeddings back in the database.
    """
    Neo4jVector.from_existing_graph(
        AzureOpenAIEmbeddings(model=embeddings_model),
        url=url,
        username=username,
        password=password,
        index_name=entity_type,
        node_label=entity_type,
        text_node_properties=['value'],
        embedding_node_property='embedding'
    )
def create_embedding(text):
    """
    Create an embedding for a given text with Azure Open AI.
    """
    embeddings_model = "text-embedding-3-large"
    client = AzureOpenAIEmbeddings(model=embeddings_model)
    return client.embed_query(text)

# The threshold defines how closely related words should be. Adjust the threshold to return more or less results
def create_query(text, threshold=0.35):
    """
    Create a Cypher query to search for products based on a user prompt.
    """
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
            similarity_data.append(f"gds.similarity.cosine({key}Var.embedding, ${key}Embedding) > {threshold}")
    query += "\nWHERE "
    query += " AND ".join(e for e in similarity_data)
    query += "\nRETURN p"
 
    return query

def query_graph(response):
    """
    Generate and execute a Cypher query based on user input and return results.
    """
    embeddingsParams = {}
    query = create_query(response)
    query_data = json.loads(response)
    for key, val in query_data.items():
        embeddingsParams[f"{key}Embedding"] = create_embedding(val)
    result = graph.query(query, params=embeddingsParams)
    return result

@tool(parse_docstring=True)
def query_db(params):
    """Query a Neo4j database for products based on user input."""
    matches = []
    # Querying the db
    result = query_graph(params)
    for r in result:
        product_id = r['p']['id']
        matches.append({
            "id": product_id,
            "name":r['p']['name']
        })
    return matches

@tool(parse_docstring=True)
def similarity_search(prompt, threshold=0.20):
    """Search for products in the database based on similarity to a user prompt."""
    matches = []
    embedding = create_embedding(prompt)
    query = '''
            WITH $embedding AS inputEmbedding
            MATCH (p:Product)
            WHERE gds.similarity.cosine(inputEmbedding, p.embedding) > $threshold
            RETURN p
            '''
    result = graph.query(query, params={'embedding': embedding, 'threshold': threshold})
    for r in result:
        product_id = r['p']['id']
        matches.append({
            "id": product_id,
            "name":r['p']['name']
        })
    return matches
 
if __name__ ==  "__main__":

    # Load data from json file (only do this once)
    file_path = "../data/amazon_product_kg.json"
    df, json_data = load_data(file_path)
    
    # Get credentials to connect to Neo4j
    credentials = get_credentials()

    # Connect to Neo4j and get the graph object
    graph_config = {
        "url": 'bolt://localhost:7687',
        "username": 'neo4j',
        "password": credentials['neo4j_password'],
    }
    graph = connect_neo4j(**graph_config)

    # Import data into neo4j (only do this once)
    # import_data(json_data, graph)

    # Get all entities that should be embedded 
    entities_list = df['entity_type'].unique()
    embeddings_model = "text-embedding-3-large"

# This method takes text from our database, calculates embeddings and stores them back in the database.
#     vector_index = Neo4jVector.from_existing_graph(
#     AzureOpenAIEmbeddings(model=embeddings_model),
#     url=graph_config['url'],
#     username=graph_config['username'],
#     password=credentials['neo4j_password'],
#     index_name='products',
#     node_label="Product",
#     text_node_properties=['name', 'title'],
#     embedding_node_property='embedding',
# )
 
#     for t in entities_list:
#         vectorize_data(t, embeddings_model, graph_config['url'], graph_config['username'], credentials['neo4j_password'])

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

    # Define tools
    tools_list = {"query_db": query_db,
                    "similarity_search": similarity_search
                    }  

    # Define the model
    model = AzureChatOpenAI(temperature=0, model="gpt-4o", api_version='2024-02-15-preview')

    # Define tools
    tools = [query_db, similarity_search]
    model_with_tools = model.bind_tools(list(tools_list.values()))

    # Define the prompt template
    template = """You are a helpful assistant that uses RAG to get useful information from a graph database that may help answer the question. 
    Your goal is to find a product in the database that best matches the user prompt.
    You have access to these tools:

    {tools}

    Rules to follow:

    1. Start by using the Query tool with the prompt as parameter. If you found results, stop here.
    2. If the result is an empty array, use the similarity search tool with the full initial user prompt. If you found results, stop here.
    3. If you cannot still cannot find the answer with this, probe the user to provide more context on the type of product they are looking for. 

    Keep in mind that we can use entities of the following types to search for products:

    {entity_types}.

    3. Repeat Step 1 and 2. If you found results, stop here.

    4. If you cannot find the final answer, say that you cannot help with the question.

    Never return results if you did not find any results in the array returned by the query tool or the similarity search tool.

    If you didn't find any result, reply: "Sorry, I didn't find any suitable products."

    If you found results from the database, this is your final answer, reply to the user by announcing the number of results and returning results in this format (each new result should be on a new line):

    name_of_the_product (id_of_the_product)"

    Only use exact names and ids of the products returned as results when providing your final answer.

    User prompt:

    {user_prompt}

    Helpful Answer:"""

    
    custom_prompt = PromptTemplate.from_template(template)

    # use langchain to create a chain
    chain = custom_prompt | model_with_tools

    # Example user prompt
    user_prompt = "I am looking for a high waist pants that is waterproof and easy to use."
    response = chain.invoke({"user_prompt": user_prompt, "tools": tools, "entity_types": entity_types})

    