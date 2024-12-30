# KG-RAG

--- Useful links ---

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

--- Context on python files ---

KG-Creation_V2:

Needs Neo4j desktop for it to work
Currently uses LLMSherpa to read the pdf and HuggingFace for embeddings but can be changed to use other llms and pdf parsers

For LLM Sherpa: Install docker and follow steps under 'Installation Steps' in https://github.com/nlmatics/nlm-ingestor, once set-up, make sure to run the docker container when parsing pdfs

Each part only has to be run once, still has to be changed to functions for easier calling of an individual part
Part 1: Defines functions to initialize Neo4j schema and to ingest a document into the database
Part 2: Parses document(s) and ingests it to neo4j
Part 3: Embeds the chunks and uploads them to database
Part 4: Calculates chunk similarities and creates links between them in neo4j
