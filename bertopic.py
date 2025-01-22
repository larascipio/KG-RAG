#!/usr/bin/env python
# coding: utf-8

# This file runs estimates a Bertopic model for a single document to estimate topics based on a collection of texts. This demonstration used a single text divided into chunks to estimate a topic for each chunk obtained from the overall text. The result obtained is that the model divided the chunks based on the chapter topic.  

# In[ ]:


# Import necessary packages
from bertopic import BERTopic
import os


# In[92]:


# Set working directory to source file location
cwd = os.getcwd()


# In[93]:


# Define file name and filetype
file = "Advanced Business Law and the Legal Environment"
filetype = "csv"


# In[94]:


# Set path to data source
path = (f"{cwd}\\{file}.{filetype}")


# In[95]:


path


# In[96]:


# from datasets import load_dataset
# dataset = load_dataset("csv", data_files=path)["train"]


# In[97]:


import pandas as pd
# Load data and convert to df
df = pd.read_csv(path, header=None, on_bad_lines='skip',
                 delimiter='\t')
df.head()


# In[98]:


# Remove all empty chunks and convert columns to string
df = df.dropna()
df[0] = df.astype('string')


# In[99]:


# Convert to list
data = df[0].to_list()


# In[101]:


len(data)


# ### Pre-calculate Embeddings
# 
# BERTopic works by converting documents into numerical values, called embeddings. This process can be very costly, especially if we want to iterate over parameters. Instead, we can calculate those embeddings once and feed them to BERTopic to skip calculating embeddings each time.

# In[102]:


from sentence_transformers import SentenceTransformer

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(data, show_progress_bar=True)


# In[103]:


embeddings


# ### Preventing Stochastic Behavior
# 
# 
# In BERTopic, we generally use a dimensionality reduction algorithm to reduce the size of the embeddings. This is done to prevent the curse of dimensionality to a certain degree.
# 
# As a default, this is done with UMAP which is an incredible algorithm for reducing dimensional space. However, by default, it shows stochastic behavior which creates different results each time you run it. To prevent that, we will need to set a random_state of the model before passing it to BERTopic.
# 
# As a result, we can now fully reproduce the results each time we run the model.

# In[104]:


from umap.umap_ import UMAP
#from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)


# ### Controlling Number of Topics
# 
# There is a parameter to control the number of topics, namely nr_topics. This parameter, however, merges topics after they have been created. It is a parameter that supports creating a fixed number of topics.
# 
# However, it is advised to control the number of topics through the cluster model which is by default HDBSCAN. HDBSCAN has a parameter, namely min_topic_size that indirectly controls the number of topics that will be created.
# 
# A higher min_topic_size will generate fewer topics and a lower min_topic_size will generate more topics.

# In[105]:


from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


# ### Improving Default Representation
# 
# The default representation of topics is calculated through c-TF-IDF. However, c-TF-IDF is powered by the CountVectorizer which converts text into tokens. Using the CountVectorizer, we can do a number of things:
# 
# * Remove stopwords
# * Ignore infrequent words
# * Increase
# 
# In other words, we can preprocess the topic representations after documents are assigned to topics. This will not influence the clustering process in any way.
# 
# Here, we will ignore English stopwords and infrequent words. Moreover, by increasing the n-gram range we will consider topic representations that are made up of one or two words.

# In[106]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 3))


# ### Additional Representations
# 
# Previously, we have tuned the default representation but there are quite a number of other topic representations in BERTopic that we can choose from. From KeyBERTInspired and PartOfSpeech, to OpenAI's ChatGPT and open-source alternatives, many representations are possible.
# 
# In BERTopic, you can model many different topic representations simultanously to test them out and get different perspectives of topic descriptions. This is called multi-aspect topic modeling.
# 
# Here, we will demonstrate a number of interesting and useful representations in BERTopic:
# 
# * KeyBERTInspired
#     * A method that derives inspiration from how KeyBERT works
# * PartOfSpeech
#     * Using SpaCy's POS tagging to extract words
# * MaximalMarginalRelevance
#     * Diversify the topic words
# * OpenAI
#     * Use ChatGPT to label our topics

# In[107]:


import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

# KeyBERT
keybert_model = KeyBERTInspired()

# Part-of-Speech
pos_model = PartOfSpeech("en_core_web_sm")

# MMR
mmr_model = MaximalMarginalRelevance(diversity=0.3)

# GPT-3.5
prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
topic: <topic label>
"""
client = openai.OpenAI(api_key="sk-...")
openai_model = OpenAI(client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
    # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
    "MMR": mmr_model,
    "POS": pos_model
}


# Training

# In[108]:


from bertopic import BERTopic

topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

topics, probs = topic_model.fit_transform(data, embeddings)


# In[109]:


topic_model.get_topic_info()


# In[110]:


topic_model.get_topic(1, full=True)


# (Custom) Labels

# In[112]:


# # Label the topics yourself
# topic_model.set_topic_labels({1: "Space Travel", 7: "Religion"})

# # or use one of the other topic representations, like KeyBERTInspired
# keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
# topic_model.set_topic_labels(keybert_topic_labels)

# or ChatGPT's labels
#chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in topic_model.topic_aspects_["OpenAI"].items()}
#chatgpt_topic_labels[-1] = "Outlier Topic"
#topic_model.set_topic_labels(chatgpt_topic_labels)


# In[113]:


topic_model.get_topic_info()


# Topic-Document Distribution

# In[114]:


# `topic_distr` contains the distribution of topics in each document
topic_distr, _ = topic_model.approximate_distribution(data, window=8, stride=4)


# In[115]:


# # Visualize the topic-document distribution for a single document
# topic_model.visualize_distribution(topic_distr[0])


# In[116]:


# # Visualize the topic-document distribution for a single document
# topic_model.visualize_distribution(topic_distr[abstract_id], custom_labels=True)


# In[117]:


# # Calculate the topic distributions on a token-level
# topic_distr, topic_token_distr = topic_model.approximate_distribution(abstracts[abstract_id], calculate_tokens=True)

# # Visualize the token-level distributions
# df = topic_model.visualize_approximate_distribution(abstracts[abstract_id], topic_token_distr[0])
# df


# Outlier Reduction

# In[118]:


# Reduce outliers
new_topics = topic_model.reduce_outliers(data, topics)

# Reduce outliers with pre-calculate embeddings instead
new_topics = topic_model.reduce_outliers(data, topics, strategy="embeddings", embeddings=embeddings)


# Visualize Topics

# In[119]:


topic_model.visualize_topics(custom_labels=True)


# In[120]:


topic_model.visualize_hierarchy(custom_labels=True)


# Visualize Documents

# In[121]:


# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=4, min_dist=0.0, metric='cosine').fit_transform(embeddings)


# In[122]:


reduced_embeddings[0]
len(reduced_embeddings)


# In[123]:


# Visualize the documents in 2-dimensional space and show the titles on hover instead of the abstracts
# NOTE: You can hide the hover with `hide_document_hover=True` which is especially helpful if you have a large dataset
topic_model.visualize_documents(data, reduced_embeddings=reduced_embeddings, custom_labels=True)


# In[124]:


# We can also hide the annotation to have a more clear overview of the topics
topic_model.visualize_documents(data, reduced_embeddings=reduced_embeddings, custom_labels=True, hide_annotations=True)


# Model Inference

# In[125]:


topic_model.get_topic_info()


# In[126]:


topic_model.get_topic_info()[['Topic', 'Name']]


# In[127]:


topic_model.get_params()


# In[137]:


topic_model._get_param_names()


# Save Topics: Topic Label + Numerical Label

# In[138]:


chunk_topic = {
    "Data":data,
    "Topic": topics
    }


# In[139]:


df_chunk_topic = pd.DataFrame(chunk_topic)
# df_chunk_topic['Topic'] = df_chunk_topic['Topic']+1


# In[140]:


df_chunk_topic.loc[df_chunk_topic['Topic'] > -1]


# In[141]:


topic_model.get_topic_info()[['Topic', 'Name']]


# In[142]:


topic_labels = topic_model.get_topic_info()[['Topic', 'Name']]


# In[143]:


df_chunk_topic = df_chunk_topic.merge(topic_labels, on='Topic', how='left')


# In[144]:


# df_chunk_topic = df_chunk_topic['Name'].rename('Label')


# In[147]:


df_chunk_topic = df_chunk_topic.rename(columns={'Name': 'Label'})


# In[152]:


df_chunk_topic = df_chunk_topic.drop('Topic', axis=1)


# In[154]:


df_chunk_topic


# In[153]:


df_chunk_topic.to_csv('Business-Law-Legal-Chunks-Topics.csv')

