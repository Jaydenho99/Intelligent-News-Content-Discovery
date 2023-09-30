
# Python File To Update Topic Modelling Visualization for each Month

# Install libraries
get_ipython().system('pip install pymongo')
get_ipython().system('pip install pytz')
get_ipython().system('pip install tweepy')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install contractions')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install scikit-learn')


# Import libraries
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime,timedelta,date
from pymongo import MongoClient,collection
import re
from pytz import timezone
import pytz
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import contractions
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Obtain a list of stopwords
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines()) 
additional_stopwords={"cent"}
stopwords_final=stopwords | additional_stopwords

# Lemmatizer library
lemmatizer = WordNetLemmatizer()

# Credentials for mongoDB database
client = MongoClient("mongodb+srv://jayden:Leonho31@atlascluster.hrjoukd.mongodb.net/?retryWrites=true&w=majority")
db_news=client['news-content']
db_news_articles=db_news['news-articles']

# Data pre-processing to clean text data
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text= re.sub('[,\.!?]','',text)
    text = re.sub(r'http\S+', '', text)
    text=contractions.fix(text)
    text = text.lower().split()
    text = [word for word in text if not word in stopwords_final]
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

# Import the optimal LDA model
lda_model_optimal = LdaModel.load("C:/ContentNewsFYP/flask-server/static/LDA_model/lda_model_file_25_5_2023")
lda_model_optimal


# Get the current month and year
current_month = datetime.now().month
current_year = datetime.now().year

# Calculate the previous month
previous_month = current_month - 1
if previous_month == 0:  # If the current month is January
    previous_month = 12
    current_year -= 1

# User-specified month
target_month = 5  # May

# Get the current year and previous month
current_year = datetime.now().year
previous_month = target_month - 1 if target_month > 1 else 12

# Match the documents in the database and pre-processed the documents
preprocessed_text = []
for doc in db_news['news-articles'].find({
    "newsType": "Tweets",
    "newsTimeDatePublished": {
        "$gte": datetime(current_year, target_month, 1),
        "$lt": datetime(current_year, target_month + 1, 1)
    }
}):
    text = preprocess_text(doc["newsTitle"])
    preprocessed_text.append(text)

preprocessed_text = []
for doc in db_news['news-articles'].find({"newsType":"Tweets","newsTimeDatePublished": {"$gte": datetime(current_year, previous_month, 1), "$lt": datetime(current_year, previous_month + 1, 1)}}):
    text = preprocess_text(doc["newsTitle"])
    preprocessed_text.append(text)


# Import corpus from the optimal LDA model
bow_corpus = [lda_model_optimal.id2word.doc2bow(doc) for doc in preprocessed_text]

# Perform the aggregation query to retrieve distinct months
pipeline = [
    {
        "$project": {
            "month": { "$month": "$newsTimeDatePublished" }
        }
    },
    {
        "$group": {
            "_id": "$month"
        }
    },
    {
        "$sort": {
            "_id": 1
        }
    }
]
distinct_months = list(db_news['news-articles'].aggregate(pipeline))

# Visualize the LDA model using pyLDAvis library
vis_data = gensimvis.prepare(lda_model_optimal, bow_corpus, lda_model_optimal.id2word,sort_topics=False)
pyLDAvis.display(vis_data)

# Save the visualization in html in the local folder
save_path = "C:/ContentNewsFYP/flask-server/templates/"
pyLDAvis.save_html(vis_data, save_path + "lda.html")

