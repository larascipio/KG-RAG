# KG-RAG project
## Description
This project focuses on developing Knowledge Graphs (KGs) for use in Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs). The implementation leverages Neo4j, a graph database (GD), to create, store, and retrieve graphs as needed. The current codebase serves as an initial step towards achieving this goal. The KG-Creation.py script processes PDF documents, structures the extracted data into a graph database, and computes semantic similarity between text chunks using embeddings. Additionally, the chatbot.py script integrates different components into a functional chatbot, enabling future interaction with the knowledge graph. Further development of the code is required to integrate its components into a cohesive and functional application. 

## Installation

1. **Clone the Repository**  
   Start by cloning the GitHub repository to your local machine:  
   ```bash
   git clone https://github.com/larascipio/KG-RAG
   cd KG-RAG
   ```

2. **Set Up the Python Environment**  
   - Ensure you have Python 3.8 or higher installed on your system.  
   - Create and activate a virtual environment to manage project dependencies:
     ```bash
     python -m venv .venv
     source .venv/bin/activate    # On Linux/Mac
     .venv\Scripts\activate       # On Windows
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Install Neo4j Desktop**  
   Download and install **Neo4j Desktop** from their official website:  
   [Neo4j Desktop Download](https://neo4j.com/download/).

4. **Create a Neo4j Account**  
   If you don’t already have a Neo4j account, register for one. This will allow you to manage and connect to your graph database.

5. **Set Up Your Neo4j Database**  
   - Launch Neo4j Desktop and create a new local database instance.  
   - Note down the following details:  
     - **Database URI** (default: `bolt://localhost:7687`)  
     - **Username** (default: `neo4j`)  
     - **Password** (the password you set during database creation).  

6. **Retrieve an LLM API Key**  
   - Through AzureOpenAI (this is what chatbot.py uses) 
   - Alternatively, you can use a free API key from services like [HuggingFace](https://huggingface.co/inference-api).

7. **Create a `.env` File**  
   In the root directory of the project, create a `.env` file to securely store your sensitive credentials. Add the following information to the file:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_neo4j_password

   LLM_API_KEY=your_llm_api_key
   LLM_API_ENDPOINT=your_llm_endpoint
   LLM_DEPLOYMENT_ID=your_llm_deployment_id
   ```

8. **Run the Application**  
   - Ensure that Neo4j is running locally.  
   - Test your setup by running one of the provided scripts, such as the chatbot:
     ```bash
     python chatbot.py
     ```

---
## Useful Links
KG-Creation_V1:
https://neo4j.com/developer-blog/llamaparse-knowledge-graph-documents/
https://github.com/Joshua-Yu/graph-rag/blob/main/openai%2Bllamaparse/demo_neo4j_vectordb.ipynb

KG-Creation_V2:
https://medium.com/neo4j/building-a-graph-llm-powered-rag-application-from-pdf-documents-24225a5baf01
https://neo4j.com/developer-blog/graph-llm-rag-application-pdf-documents/
https://github.com/Joshua-Yu/graph-rag/tree/main/openai%2Bllmsherpa/genai-stack

RAG Application:
https://cookbook.openai.com/examples/rag_with_graph_db
https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_graph_db.ipynb

General:
https://towardsdatascience.com/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759
Weaviate, Neo4j, Langchain, Langflow, Huggingface, LLMSherpa

## Context on Python files
KG-Creation_V2:

Needs Neo4j desktop for it to work
Currently uses LLMSherpa to read the pdf and HuggingFace for embeddings but can be changed to use other llms and pdf parsers

For LLM Sherpa: Install docker and follow steps under 'Installation Steps' in https://github.com/nlmatics/nlm-ingestor, once set-up, make sure to run the docker container when parsing pdfs

Each part only has to be run once, still has to be changed to functions for easier calling of an individual part
Part 1: Defines functions to initialize Neo4j schema and to ingest a document into the database
Part 2: Parses document(s) and ingests it to neo4j
Part 3: Embeds the chunks and uploads them to database
Part 4: Calculates chunk similarities and creates links between them in neo4j

chatbot.py

This script interacts with a Neo4j database using Langchain and Azure OpenAI. It is based on the example from the OpenAI Cookbook ([link](https://cookbook.openai.com/examples/rag_with_graph_db)), but has been updated to work with the latest version of Langchain. The data utilized in this script is stored in the `/data` directory, matching the dataset from the original cookbook example.