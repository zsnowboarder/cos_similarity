#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util




# In[6]:


# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#model = SentenceTransformer('all-distilroberta-v1')
#model = SentenceTransformer('all-MiniLM-L12-v2')
#model = SentenceTransformer('paraphrase-mpnet-base-v2')
#model = SentenceTransformer('nli-mpnet-base-v2') # this can handle negation better.


# In[9]:


def replace_abbreviations(text):
    abbreviations = {
        "blk":"black",
        "brn":"brn",
        "blu":"blue"
    }
    for abbr, word in abbreviations.items():
        text = re.sub(r"\b" + abbr + r"\b", word, text)
    return text

# this function is for embedding models. it will make the embedding treat NOT as a distinct token.
def handle_negations(text):
    # Define common negation patterns
    negation_words = ["not", "never", "no", "neither", "none", "cannot"]
    for word in negation_words:
        text = text.replace(f" {word} ", f" {word}_(NEG) ")
    return text


# In[11]:


# Create a DataFrame with descriptions
data = {
    'report': [
        "The car turned left and the kept going. The description of the car was white honda civic, 4 door. The victim didn't see the suspects clearly.",
        "Victim described the make was honda and model was civic. colour is white. 4 door with large windows",
        "The description provided was honda accord, 4 door.",
        "It was a black pickup truck. unknown make and model."
    ]
}
df = pd.DataFrame(data)
df["report"] = df["report"].apply(replace_abbreviations)
df


# In[ ]:





# # Text to compare against
# query_text = "white honda civic with 4 doors."
# 
# # Encode the query text
# query_embedding = model.encode(query_text, convert_to_tensor=True)
# 
# # Calculate cosine similarity for each description
# def calculate_similarity(description):
#     description_embedding = model.encode(description, convert_to_tensor=True)
#     similarity_score = util.pytorch_cos_sim(query_embedding, description_embedding)
#     return similarity_score.item()
# 
# # Apply similarity calculation and add as a new column
# df['Similarity Score'] = df['report'].apply(calculate_similarity)
# 
# print(df)

# ## Streamlit ##

# In[84]:


st.title("COS Similarity")
st.write("""This demo shows the application of cos simialarity to scan through the reports and identify a match in description, MO or simiarity of the texts.
            The model searches through the database and returns a similarity score.""")

query_text = st.text_area(label="Enter a vehicle description or a sex offender MO:", value="a 4-dr white honda civic was driving east and hit another car. the honda didn't stop. Witness saw the car and reported the hit and run to police.")
query_text = handle_negations(query_text)

# Encode the query text
query_embedding = model.encode(query_text, convert_to_tensor=True)

# Calculate cosine similarity for each description
def calculate_similarity(description):
    description_embedding = model.encode(description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(query_embedding, description_embedding)
    return similarity_score.item()

if st.button(label="Search"):
    # Apply similarity calculation and add as a new column
    df['Similarity Score'] = df['report'].apply(calculate_similarity)
    st.table(df)

st.write("This demo only shows examples of vehicle and MO search. It can do any similarity search and the possibilities are endless.")


# In[ ]:




