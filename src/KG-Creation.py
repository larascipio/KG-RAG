from llmsherpa.readers import LayoutPDFReader
from neo4j import GraphDatabase
import uuid
import hashlib
import os
import glob
from datetime import datetime
import csv
from sklearn.metrics.pairwise import cosine_similarity 
import matplotlib.pyplot as plt
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

'''
Loading and setting the necessary credentials from a .env file
'''

load_dotenv()

NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION_EMBEDDING = os.getenv('AZURE_OPENAI_API_VERSION_EMBEDDING')
AZURE_OPENAI_API_VERSION_QUERY = os.getenv('AZURE_OPENAI_API_VERSION_QUERY')
AZURE_DEPLOYMENT_ID_EMBEDDING = os.getenv('AZURE_DEPLOYMENT_ID_EMBEDDING')
AZURE_DEPLOYMENT_ID_QUERY = os.getenv('AZURE_DEPLOYMENT_ID_QUERY')

# Manually setting the other necessary credentials
NEO4J_DATABASE = "neo4j"

'''
Part 1: Define functions to initialise Neo4j schema and to ingest a document into the database
'''

def initialiseNeo4j():
    cypher_schema = [
        "CREATE CONSTRAINT sectionKey IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT chunkKey IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT documentKey IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
        "CREATE CONSTRAINT tableKey IF NOT EXISTS FOR (c:Table) REQUIRE (c.key) IS UNIQUE;",
        "CALL db.index.vector.createNodeIndex('chunkVectorIndex', 'Embedding', 'value', 1536, 'COSINE');"
    ]

    driver = GraphDatabase.driver(NEO4J_URI, database=NEO4J_DATABASE, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        for cypher in cypher_schema:
            session.run(cypher)
    driver.close()

def ingestDocumentNeo4j(doc, doc_location):

    cypher_pool = [
        # 0 - Document
        "MERGE (d:Document {url_hash: $doc_url_hash_val}) ON CREATE SET d.url = $doc_url_val RETURN d;",  
        # 1 - Section
        "MERGE (p:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) ON CREATE SET p.page_idx = $page_idx_val, p.title_hash = $title_hash_val, p.block_idx = $block_idx_val, p.title = $title_val, p.tag = $tag_val, p.level = $level_val RETURN p;",
        # 2 - Link Section with the Document
        "MATCH (d:Document {url_hash: $doc_url_hash_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (d)<-[:HAS_DOCUMENT]-(s);",
        # 3 - Link Section with a parent section
        "MATCH (s1:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_title_hash_val}) MATCH (s2:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (s1)<-[:UNDER_SECTION]-(s2);",
        # 4 - Chunk
        "MERGE (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) ON CREATE SET c.sentences = $sentences_val, c.sentences_hash = $sentences_hash_val, c.block_idx = $block_idx_val, c.page_idx = $page_idx_val, c.tag = $tag_val, c.level = $level_val RETURN c;",
        # 5 - Link Chunk to Section
        "MATCH (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) MATCH (s:Section {key:$doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(c);",
        # 6 - Table
        "MERGE (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) ON CREATE SET t.name = $name_val, t.doc_url_hash = $doc_url_hash_val, t.block_idx = $block_idx_val, t.page_idx = $page_idx_val, t.html = $html_val, t.rows = $rows_val RETURN t;",
        # 7 - Link Table to Section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);",
        # 8 - Link Table to Document if no parent section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Document {url_hash: $doc_url_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);"
    ]

    driver = GraphDatabase.driver(NEO4J_URI, database=NEO4J_DATABASE, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        cypher = ""

        # 1 - Create Document node
        doc_url_val = doc_location
        doc_url_hash_val = hashlib.md5(doc_url_val.encode("utf-8")).hexdigest()

        cypher = cypher_pool[0]
        session.run(cypher, doc_url_hash_val=doc_url_hash_val, doc_url_val=doc_url_val)

        # 2 - Create Section nodes
        countSection = 0
        for sec in doc.sections():
            sec_title_val = sec.title
            sec_title_hash_val = hashlib.md5(sec_title_val.encode("utf-8")).hexdigest()
            sec_tag_val = sec.tag
            sec_level_val = sec.level
            sec_page_idx_val = sec.page_idx
            sec_block_idx_val = sec.block_idx

            # MERGE section node
            if not sec_tag_val == 'table':
                cypher = cypher_pool[1]
                session.run(cypher, page_idx_val=sec_page_idx_val
                                , title_hash_val=sec_title_hash_val
                                , title_val=sec_title_val
                                , tag_val=sec_tag_val
                                , level_val=sec_level_val
                                , block_idx_val=sec_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

                # Link Section with a parent section or Document
                sec_parent_val = str(sec.parent.to_text())

                if sec_parent_val == "None":    # use Document as parent

                    cypher = cypher_pool[2]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , doc_url_hash_val=doc_url_hash_val
                                    , block_idx_val=sec_block_idx_val
                                )

                else:   # use parent section
                    sec_parent_title_hash_val = hashlib.md5(sec_parent_val.encode("utf-8")).hexdigest()
                    sec_parent_page_idx_val = sec.parent.page_idx
                    sec_parent_block_idx_val = sec.parent.block_idx

                    cypher = cypher_pool[3]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , block_idx_val=sec_block_idx_val
                                    , parent_page_idx_val=sec_parent_page_idx_val
                                    , parent_title_hash_val=sec_parent_title_hash_val
                                    , parent_block_idx_val=sec_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
            # **** if sec_parent_val == "None":    

            countSection += 1
        # **** for sec in doc.sections():
    
        # ------- Continue within the blocks -------
        # 3 - Create Chunk nodes from chunks
        
        countChunk = 0
        for chk in doc.chunks():
            
            chunk_block_idx_val = chk.block_idx
            chunk_page_idx_val = chk.page_idx
            chunk_tag_val = chk.tag
            chunk_level_val = chk.level
            chunk_sentences = "\n".join(chk.sentences)

            # MERGE Chunk node
            if not chunk_tag_val == 'table':
                chunk_sentences_hash_val = hashlib.md5(chunk_sentences.encode("utf-8")).hexdigest()

                # MERGE chunk node
                cypher = cypher_pool[4]
                session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                , sentences_val=chunk_sentences
                                , block_idx_val=chunk_block_idx_val
                                , page_idx_val=chunk_page_idx_val
                                , tag_val=chunk_tag_val
                                , level_val=chunk_level_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            
                # Link chunk with a section
                # Chunk always has a parent section 

                chk_parent_val = str(chk.parent.to_text())
                
                if not chk_parent_val == "None":
                    chk_parent_hash_val = hashlib.md5(chk_parent_val.encode("utf-8")).hexdigest()
                    chk_parent_page_idx_val = chk.parent.page_idx
                    chk_parent_block_idx_val = chk.parent.block_idx

                    cypher = cypher_pool[5]
                    session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                    , block_idx_val=chunk_block_idx_val
                                    , parent_hash_val=chk_parent_hash_val
                                    , parent_block_idx_val=chk_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
                    
                # Link sentence 
                #   >> TO DO for smaller token length

                countChunk += 1
        # **** for chk in doc.chunks(): 

        # 4 - Create Table nodes

        countTable = 0
        for tb in doc.tables():
            page_idx_val = tb.page_idx
            block_idx_val = tb.block_idx
            name_val = 'block#' + str(block_idx_val) + '_' + tb.name
            html_val = tb.to_html()
            rows_val = len(tb.rows)

            # MERGE table node

            cypher = cypher_pool[6]
            session.run(cypher, block_idx_val=block_idx_val
                            , page_idx_val=page_idx_val
                            , name_val=name_val
                            , html_val=html_val
                            , rows_val=rows_val
                            , doc_url_hash_val=doc_url_hash_val
                        )
            
            # Link table with a section
            # Table always has a parent section 

            table_parent_val = str(tb.parent.to_text())
            
            if not table_parent_val == "None":
                table_parent_hash_val = hashlib.md5(table_parent_val.encode("utf-8")).hexdigest()
                table_parent_page_idx_val = tb.parent.page_idx
                table_parent_block_idx_val = tb.parent.block_idx

                cypher = cypher_pool[7]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , parent_page_idx_val=table_parent_page_idx_val
                                , parent_hash_val=table_parent_hash_val
                                , parent_block_idx_val=table_parent_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

            else:   # link table to Document
                cypher = cypher_pool[8]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            countTable += 1

        # **** for tb in doc.tables():
        
        print(f'\'{doc_url_val}\' Done! Summary: ')
        print('#Sections: ' + str(countSection))
        print('#Chunks: ' + str(countChunk))
        print('#Tables: ' + str(countTable))

    driver.close()

# Initialize NEO4J schema
initialiseNeo4j()

'''
Part 2: Parse document and ingest it to neo4j
Only has to be run once
'''

llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
file_location = 'Advanced Business Law and the Legal Environment.pdf'

# get all documents under the folder
pdf_files = glob.glob(file_location)

print(f'#PDF files found: {len(pdf_files)}!')
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

# parse documents and create graph
startTime = datetime.now()

for pdf_file in pdf_files:
    doc = pdf_reader.read_pdf(pdf_file)
    
    # Write the chunks to a csv file
    chunk_text = []
    for chk in doc.chunks():
            chk_text = ''
            
            for sentence in chk.sentences:
                chk_text += sentence
                chk_text += ' '

            chunk_text.append([chk_text])
    print(chunk_text)

    with open(file_location + '.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(chunk_text)

    # find the first / in pdf_file from right
    idx = pdf_file.rfind('/')
    pdf_file_name = pdf_file[idx+1:]
    print(pdf_file_name)

    # open a local file to write the JSON
    with open(pdf_file_name + '.json', 'w', encoding="utf-8") as f:
        # convert doc.json from a list to string
        f.write(str(doc.json))

    # ingest the document to neo4j
    ingestDocumentNeo4j(doc, pdf_file)

print(f'Total time: {datetime.now() - startTime}')

'''
Part 3: Embed the chunks and upload to database
Only has to be run once
'''

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def LoadEmbedding(label, property):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    # openai_client = AzureOpenAIEmbedding(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION_EMBEDDING, azure_endpoint=AZURE_OPENAI_ENDPOINT, azure_deployment=AZURE_DEPLOYMENT_ID_EMBEDDING)
    # embed_model = AzureOpenAIEmbeddings(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION_EMBEDDING, azure_endpoint=AZURE_OPENAI_ENDPOINT, azure_deployment=AZURE_DEPLOYMENT_ID_EMBEDDING)
    # client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION_EMBEDDING, azure_endpoint=AZURE_OPENAI_ENDPOINT, azure_deployment=AZURE_DEPLOYMENT_ID_EMBEDDING)

    with driver.session() as session:
        # get chunks in document, together with their section titles
        result = session.run(f"MATCH (ch:{label}) RETURN id(ch) AS id, ch.{property} AS text")
        # call OpenAI embedding API to generate embeddings for each proporty of node
        # for each node, update the embedding property
        count = 0
        embeddings = []
        for record in result:
            id = record["id"]
            text = record["text"]
            
            # For better performance, text can be batched
            embedding = embed_model.embed_query(text)
            embeddings.append(embedding)

            # key property of Embedding node differentiates different embeddings
            cypher = "CREATE (e:Embedding) SET e.key=$key, e.value=$embedding, e.model=$model"
            cypher = cypher + " WITH e MATCH (n) WHERE id(n) = $id CREATE (n) -[:HAS_EMBEDDING]-> (e)"
            session.run(cypher,key=property, embedding=embedding, id=id, model='all-MiniLM-L6-v2') 
            count = count + 1

        session.close()
        
        print("Processed " + str(count) + " " + label + " nodes for property @" + property + ".")
        return count
# For smaller amount (<2000) of text data to embed
LoadEmbedding("Chunk", "sentences")

'''
Part 4: Calculate chunk similarities and create links between them 
Can be skipped if other ways of linking chunks are used
Can be updated to use results from entities and relations extraction to link chunks
'''

# Cosine similarity of embeddings
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)

with driver.session() as session:
    result = session.run("MATCH (ch:Embedding) RETURN ch.value AS text, ID(ch) AS id")
    result_chunk = session.run("MATCH (ch:Chunk) RETURN ID(ch) AS id")
    
    N = 6229 # Can be set to ChunkCount if entire script is executed
    Embeddings = []
    IDs_Embeddings = []
    IDs_Chunks = []

    results = result.fetch(N)
    results_chunk = result_chunk.fetch(N)

session.close()

for i in range(N):
    Embeddings.append(results[i][0])
    IDs_Embeddings.append(results[i][1])
    IDs_Chunks.append(results_chunk[i][0])

Matrix = cosine_similarity(Embeddings)
print(IDs_Chunks)
print(IDs_Embeddings)
#print(Matrix)

with driver.session() as session:

    for i in range(N):
        for j in range(N):
            if Matrix[i, j] >= 0.75:
                id_1 = IDs_Chunks[i]
                id_2 = IDs_Chunks[j]
                cypher = f'MATCH (c:Chunk WHERE ID(c) = {id_1}), (s:Chunk WHERE ID(s) = {id_2}) CREATE (c)<-[:SIMILAR_TO]-(s)'
                session.run(cypher)

session.close()

plt.imshow(Matrix)
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()
print(Matrix)
