import os
import json
import re
from typing import Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
        
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        #tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        kwargs["entity_types"] = json.dumps(entity_types)
        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
# TODO add type
# ------------------------- UTILITY FUNCTIONS -------------------------
def load_data(file_path: str):
    """
    Load JSON data from a file and return as a DataFrame and dictionary.
    """

    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.read_json(file_path)
    return df, data

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

# ------------------------- DATA PROCESSING -------------------------

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

# TODO write exceptions
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

# ------------------------- QUERY FUNCTIONS -------------------------

# Define the entities to look for
# def define_query(prompt, model="gpt-4o"):
#     client = openai.AzureOpenAI(api_version='2024-02-15-preview')
#     completion = client.chat.completions.create(
#         model=model,
#         temperature=0,
#         response_format= {
#             "type": "json_object"
#         },
#     messages=[
#         {
#             "role": "system",
#             "content": system_prompt
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#         ]
#     )
#     return completion.choices[0].message.content

def create_embedding(text):
    """
    Create an embedding for a given text with Azure Open AI.
    """
    embeddings_model = "text-embedding-3-large"
    client = AzureOpenAIEmbeddings(model=embeddings_model)
    return client.embed_query(text)

# The threshold defines how closely related words should be. Adjust the threshold to return more or less results
def create_query(text, threshold=0.81):
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


def query_similar_items(product_id, relationships_threshold = 3):
    
    similar_items = []
        
    # Fetching items in the same category with at least 1 other entity in common
    query_category = '''
            MATCH (p:Product {id: $product_id})-[:hasCategory]->(c:category)
            MATCH (p)-->(entity)
            WHERE NOT entity:category
            MATCH (n:Product)-[:hasCategory]->(c)
            MATCH (n)-->(commonEntity)
            WHERE commonEntity = entity AND p.id <> n.id
            RETURN DISTINCT n;
        '''
    

    result_category = graph.query(query_category, params={"product_id": int(product_id)})
    #print(f"{len(result_category)} similar items of the same category were found.")
          
    # Fetching items with at least n (= relationships_threshold) entities in common
    query_common_entities = '''
        MATCH (p:Product {id: $product_id})-->(entity),
            (n:Product)-->(entity)
            WHERE p.id <> n.id
            WITH n, COUNT(DISTINCT entity) AS commonEntities
            WHERE commonEntities >= $threshold
            RETURN n;
        '''
    result_common_entities = graph.query(query_common_entities, params={"product_id": int(product_id), "threshold": relationships_threshold})
    #print(f"{len(result_common_entities)} items with at least {relationships_threshold} things in common were found.")

    for i in result_category:
        similar_items.append({
            "id": i['n']['id'],
            "name": i['n']['name']
        })
            
    for i in result_common_entities:
        result_id = i['n']['id']
        if not any(item['id'] == result_id for item in similar_items):
            similar_items.append({
                "id": result_id,
                "name": i['n']['name']
            })
    return similar_items

def query_db(params):
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

def similarity_search(prompt, threshold=0.8):
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

def agent_interaction(agent_executor, user_prompt):
    agent_executor.run(user_prompt)

def main():
    pass


if __name__ == "__main__":
    file_path = 'C:/Users/lara.scipio/Documents/KG-RAG/KG-RAG/data/amazon_product_kg.json'

    # Load data
    # df, json_data = load_data(file_path)
    print("Data loaded successfully")
    # Import data 
    # import_data(json_data, graph)
    print("Data imported successfully")

    # Get all entities that should be embedded
    # entities_list = df['entity_type'].unique()
    # embeddings_model = "text-embedding-3-large"

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
    
#     for t in entities_list:
#         vectorize_data(t, embeddings_model, url, username, password_neo)

    print("Data is embedded")
    # Load credentials and connect to Neo4j
    credentials = get_credentials()
    graph_config = {
        "url": 'bolt://localhost:7687',
        "username": 'neo4j',
        "password": credentials['neo4j_password'],
    }
    graph = connect_neo4j(**graph_config)


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

    # result = query_graph(example_response)

    # # Result
    # print(f"Found {len(result)} matching product(s):\n")
    # for r in result:
    #     print(f"{r['p']['name']} ({r['p']['id']})")
    
    # product_ids = ['1519827', '2763742']

    # for product_id in product_ids:
    #     print(f"Similar items for product #{product_id}:\n")
    #     result = query_similar_items(product_id)
    #     print("\n")
    #     for r in result:
    #         print(f"{r['name']} ({r['id']})")
    #     print("\n\n")

    tools = [
        Tool(
            name="Query",
            func=query_db,
            description="Use this tool to find entities in the user prompt that can be used to generate queries"
        ),
        Tool(
            name="Similarity Search",
            func=similarity_search,
            description="Use this tool to perform a similarity search with the products in the database"
        )
    ]

    tool_names = [f"{tool.name}: {tool.description}" for tool in tools]


    prompt_template = '''Your goal is to find a product in the database that best matches the user prompt.
    You have access to these tools:

    {tools}

    Use the following format:

    Question: the input prompt from the user
    Thought: you should always think about what to do
    Action: the action to take (refer to the rules below)
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

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
    {input}

    {agent_scratchpad}

    '''

    prompt = CustomPromptTemplate(
        template=prompt_template,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )

    output_parser = CustomOutputParser()

    llm = AzureChatOpenAI(temperature=0, model="gpt-4o", api_version='2024-02-15-preview')

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Using tools, the LLM chain and output_parser to make an agent
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\Observation:"], 
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    prompt1 = "I'm searching for pink shirts"
    agent_interaction(agent_executor, prompt1)

    prompt2 = "Can you help me find a toys for my niece, she's 8"
    agent_interaction(agent_executor, prompt2)

    prompt3 = "I'm looking for nice curtains"
    agent_interaction(agent_executor, prompt3)

    
