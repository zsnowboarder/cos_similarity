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
        "It was a black pickup truck. unknown make and model.",
        "Victim was walking to home. Suspect jumped out from the bush and groped victim from behind and ran a way. Victim reported the incident to police.",
        "Victim was on the way to school. Suspect followed victim in a vehicle. Suspect approached victim and asked if victim wanted candies. Victim ran away.",
        "Victim was walking on a street. Suspect suddenly groouped victim and pushed victim to the ground. Victim was able to escape."
    ]
}

# Load excel data
#df = pd.read_excel("/mount/src/cos_similarity/data.xlsx")
df = pd.read_excel("data.xlsx")
#df = pd.DataFrame(data)
df["report"] = df["report"].apply(replace_abbreviations)

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
st.write("""This demo shows the application of cosine simialarity to scan through the reports and identify a match in description, MO or simiarity of the texts.
            The model analyzes the database and returns a similarity score.
            The model understands the context. For example, it recognizes that youth, teen, and young male have similar meanings.""")

# query_text = st.text_area(label="Enter a vehicle description or a sex offender MO:", value="Female student was walking home from school. Suspect jumped out of the bush and groped the student from behind.")
query_text = st.text_area(label="Enter a vehicle description or a sex offender MO:", value="Victim walked from the BMO Bank of Montreal located at Champlain Square to her mechanic's residence. As she was walking North in the 6800 block of Main St., an unknown male snook up directly behind her and grabbed both of her buttocks with two of his bare hands. Victim yelled out loud and the male continued to sprint northbound along the East side of Main St. Police conducted an area search with negative results. Witness, a resident in the 6700 block of Main St observed the suspect running northbound toward Pandora Ave while another resident in the block, provided CCTV which corroborates the event. PC's reviewed the video which showed the side profile of a u/k race, young adult male, wearing a black hoody, dark track suit, and black sneakers. CCTV to be tagged at the property office for safekeeping.")
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
    df = df.sort_values(by="Similarity Score", ascending=False)
    st.table(df.style.hide(axis="index"))

st.write("This demo only shows examples of vehicle and MO search. It can do any similarity search and the possibilities are endless.")


# In[ ]:


