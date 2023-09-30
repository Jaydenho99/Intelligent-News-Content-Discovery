
# Install all relevant requirements
get_ipython().system('pip install selenium')
get_ipython().system('pip install bs4')
get_ipython().system('pip install kora')
get_ipython().system('pip install pymongo')
get_ipython().system('pip install pytz')
get_ipython().system('pip install rake-nltk')
get_ipython().system('pip install newsdataapi')
get_ipython().system('pip install tweepy')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install bokeh_wordcloud2')
get_ipython().system('pip install wikipedia-api')
get_ipython().system('pip install timeago')


# Import all Python libraries
from kora.selenium import wd
import string
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException,TimeoutException
import requests
import timeago
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from datetime import datetime,timedelta,date
from pymongo import MongoClient,collection,UpdateOne
from pytz import timezone
import re
import nltk
import json
from rake_nltk import Rake
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.ldamodel import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
import contractions
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Credentials access for mongodb database 
client = MongoClient("mongodb+srv://jayden:Leonho31@atlascluster.hrjoukd.mongodb.net/?retryWrites=true&w=majority")
db_news=client['news-content']
db_news_articles=db_news['news-articles']
db_tweets=db_news['tweets']


# Import optimal LDA model
lda_model_optimal = LdaModel.load("C:/ContentNewsFYP/flask-server/static/LDA_model/lda_model_file_25_5_2023")

# create an empty dataframe with columns for topic and keywords
df_topic = pd.DataFrame(columns=['Topic', 'Keywords','Topic Name'])

# get the top 5 keywords for each topic from the LDA model
topics_new = lda_model_optimal.show_topics(num_topics=22, num_words=15, formatted=False)

# Manually assign the topic name for each topic found in LDA model based on the 3 associated keywords
for topic in topics_new:
    keywords = [word[0] for word in topic[1]]
    if all(keyword in keywords for keyword in ['school', 'student', 'education']):
        topic_name = 'Education'
    elif all(keyword in keywords for keyword in ['datuk', 'party', 'minister']):
        topic_name = 'Politics'
    elif all(keyword in keywords for keyword in ['tax', 'law', 'court']):
        topic_name = 'Law'
    elif all(keyword in keywords for keyword in ['police', 'investigation', 'suspect','officer']):
        topic_name = 'Crime'
    elif all(keyword in keywords for keyword in ['city', 'country']):
        topic_name = 'Global Affairs'
    elif all(keyword in keywords for keyword in ['government', 'minister','malaysia']):
        topic_name = 'Government'
    elif all(keyword in keywords for keyword in ['company', 'payment','business']):
        topic_name = 'Business'
    elif all(keyword in keywords for keyword in ['south', 'korea','president','nuclear','yoon']):
        topic_name = 'Geopolitics'
    elif all(keyword in keywords for keyword in ['market', 'bank','price']):
        topic_name = 'Finance'
    elif all(keyword in keywords for keyword in ['water', 'operation','project']):
        topic_name = 'Infrastructure and Environment'
    elif all(keyword in keywords for keyword in ['game', 'league','team']):
        topic_name = 'Sports'
    elif all(keyword in keywords for keyword in ['time', 'japan','tokyo']):
        topic_name = 'Trend and Social Media'
    elif all(keyword in keywords for keyword in ['family', 'time','people']):
        topic_name = 'Lifestyle and Entertainment'
    elif all(keyword in keywords for keyword in ['price', 'good','supply']):
        topic_name = 'Economy'
    elif all(keyword in keywords for keyword in ['korea', 'north','yonhap','korean','seoul']):
        topic_name = 'Korea'
    elif all(keyword in keywords for keyword in ['court', 'pardon','appeal']):
        topic_name = 'Court'
    elif all(keyword in keywords for keyword in ['malaysia', 'kuala','lumpur']):
        topic_name = 'Malaysia'
    elif all(keyword in keywords for keyword in ['road', 'traffic','vehicle']):
        topic_name = 'Transportation'
    elif all(keyword in keywords for keyword in ['child', 'labor']):
        topic_name = 'Social Issues'
    elif all(keyword in keywords for keyword in ['medium', 'digital','news']):
        topic_name = 'Media and Journalism'
    elif all(keyword in keywords for keyword in ['health', 'covid']):
        topic_name = 'Health'
    elif all(keyword in keywords for keyword in ['macc', 'corruption', 'minister']):
        topic_name = 'Investigation'
    else:
        topic_name='Others'
        
    # Create a DataFrame for the current row
    row_df = pd.DataFrame({'Topic': [topic[0]], 'Keywords': [keywords], 'Topic Name': [topic_name]})
    
    # Concatenate the row DataFrame with df_topic
    df_topic = pd.concat([df_topic, row_df], ignore_index=True)


# Convert 'Topic' and 'Topic Name' fields to dictionary format
topic_dict = df_topic.set_index('Topic')['Topic Name'].to_dict()


# define a function to get the top keyword for each document based on LDA topic modeling
def get_top_keyword(doc):
    doc_bow = lda_model_optimal.id2word.doc2bow(doc)
    topic_dists = lda_model_optimal.get_document_topics(doc_bow)
    top_topic = max(topic_dists, key=lambda x:x[1])[0]
    top_keyword = topic_dict[top_topic]
    # get the keywords for the top topic
    keywords = [word[0] for word in lda_model_optimal.show_topic(top_topic, topn=10)]
    return top_keyword,keywords


# Function to pre-process and clean data
def clean_content(content):
    # Remove non-alphanumeric characters except punctuation
    content = ''.join(char for char in content if char.isalnum() or char in string.punctuation or char.isspace())
    
    # Remove extra whitespace
    content = ' '.join(content.split())
    
    # Remove empty paragraphs
    content = '\n'.join(paragraph for paragraph in content.split('\n') if paragraph.strip())
    
    return content

# Define a regular expression to match unwanted punctuation marks
regex = r'[^\w\s\.,!?]'
pattern = r"STARPICKS\s.*?[A-Z]\."


# Define a function to remove unwanted punctuation marks from a string
def remove_punctuation(text):
    return re.sub(regex, '', text)

def remove_advertisement(text):
    return re.sub(pattern,'',text)


# Define a function to pre-process and clean the text data
def preprocess_text(text):
    # convert text to lowercase
    text = text.lower()
    
    # tokenize text into words
    words = word_tokenize(text)
    
    # remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    
    # join words back into text
    text = ' '.join(words)
    
    return text


# Obtain a list of stopwords
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines()) 
additional_stopwords={"cent"}
stopwords_final=stopwords | additional_stopwords

# Import lemmatizer library
lemmatizer = WordNetLemmatizer()

# Pre-process text data
def preprocess_text_details(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text= re.sub('[,\.!?]','',text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[#*&@()]', '', text)
    text=contractions.fix(text)
    text = text.lower().split()
    text = [word for word in text if not word in stopwords_final]
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


# To calculate the aggregation of compound score obtained from the VADER for news contents
def aggregate_compound_score_sentiment(sentences):
    analyzer = SentimentIntensityAnalyzer()
    # Calculate the compound score for each sentence
    scores = []
    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)['compound']
        scores.append(score)

    # Calculate the average compound score for the entire document
    final_score = sum(scores) / len(scores)
    return final_score


# Define a function to extract the keywords from news contents using RAKE keywords extraction
r2=Rake(min_length=1, max_length=3,include_repeated_phrases=False)
def extract_keywords(text):
    text=text.lower()
    text=remove_punctuation(text)
    r2.extract_keywords_from_text(text)
    phrases = r2.get_ranked_phrases()
    keywords = []
    for phrase in phrases:
        if len(keywords) >= 10:
            break
        freq=text.count(phrase)
        keywords.append((phrase, freq))
    
    return keywords


# Identify the condition to specify whether the document is positive, neutral or negative based on the aggregate compound score
def sentiment(x):
    if x>=0.35:
        return 'Positive'
    elif x<=-0.05:
        return 'Negative'
    else:
        return 'Neutral'

def check_duplicate_data(df):
    for row in df.to_dict(orient="records"):
        # Check for duplicates
        existing_article = db_news_articles.find_one({"newsTitle": row["newsTitle"]})

        # Insert the row data if it does not exist
        if not existing_article:
            db_news_articles.insert_one(row)
            print(f"Article {row['newsTitle']} inserted successfully!")
        else:
            print(f"Article {row['newsTitle']} already exists!")


# Determine the current date
today=datetime.now()
year=today.strftime("%Y")
month=today.strftime("%m")
day=today.strftime("%d")
current_day = datetime.combine(date.today(), datetime.min.time())
yesterday = datetime.combine(date.today() - timedelta(days=1), datetime.min.time())


# Increase the width column to get full text
pd.options.display.max_colwidth = 999999999999999999999999999999999999999


# Web Scraping for TheStar News Articles

# Get the link for TheStar news articles
wd.get("https://www.thestar.com.my/")
html_thestar = BeautifulSoup(wd.page_source,'html.parser')


# Get all the html elements from the respective class
cards=html_thestar.find_all('div',{"class":"col-sm-3 in-sec-story"})


# Scrape data for news articles using beautifulSoup - faster data extraction
def scrape_data(tag):
    try:
        heading=tag
        title=heading.a.get("data-content-title").strip()
        url=heading.a.get("href")
        image_url=heading.img.get("data-source")
    except:
        pass
    try:
        content_category=heading.a.get("data-content-category")
    except NoSuchElementException as e:
        if(e):
            content_category='None'
    try:
        content_author=heading.a.get("data-content-author")
    except NoSuchElementException as e:
        if(e):
            content_author='None'
  
    data={'title':title,'url':url, 'category':content_category,'author':content_author,'image_url':image_url}

    return data


# Extract data using beautifulSoup
def extract_data():
    ads_data=[]
    for card in cards:
        data=scrape_data(card)
        ads_data.append(data)
    
    return ads_data


dict_data=extract_data()

# Convert the data into dataframe for further processing
df=pd.DataFrame.from_dict(dict_data)
df=df[(df['category']=="News/Nation")]

# Obtain the top 15 news articles
iterator = iter(dict_data)
articles_TS=[]
for i in range(15):
    try:
        articles=next(iterator)
        articles_TS.append(articles)
    except:
        pass

# Declare a webdriver to extract dynamic data
DRIVER_PATH = "C:/Users/Leon/chromedriver.exe"
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
driver.get('https://www.thestar.com.my/')


# Extract relevant data using webdriver (Selenium)
content_values=[]
for i in articles_TS:
    if((i['author'] is None and i['category'] is None) or (i['url'].startswith("https://www.thestartv.com")) or (i['url'].startswith("https://www.carsifu.my")) or (i['category']=='AseanPlus/Aseanplus News')):
        continue
    else:
        try:
            driver.get(i['url'])
            title=driver.find_element(By.TAG_NAME,'h1').text.strip()
            story_body=driver.find_element(By.ID,'story-body')
            content=story_body.text
        except NoSuchElementException:
            image_url = np.nan
            pass
        try:
            time_date=driver.find_element(By.ID,'sideNote')
            date=time_date.find_element(By.CLASS_NAME,'date').text
        except NoSuchElementException as e:
            if(e):
                date='None'
        try:
            timestamp=time_date.find_element(By.CLASS_NAME,'timestamp').text
        except NoSuchElementException as e:
            if(e):
                timestamp='None'
    
    data={'title':title,'content':content,'timedate':f"{date} {timestamp}"}
    content_values.append(data)
    time.sleep(5)
driver.close()


# Convert dictionary to dataframe
df_content=pd.DataFrame.from_dict(content_values)
df_content


# Apply the clean_content function to clean the documents retrieved
df_content['title']=df_content['title'].apply(clean_content)
df['title']=df['title'].apply(clean_content)


# Merge all data 
final_df_TS=df_content.merge(df,on='title',how='inner').drop_duplicates(subset='title').reset_index(drop=True)
final_df_TS=final_df_TS[(final_df_TS['title'].notnull()) & (final_df_TS['content'].notnull())]
final_df_TS


# Further pre-process data
final_df_TS['content']=final_df_TS['content'].apply(clean_content)
final_df_TS['content']=final_df_TS['content'].str.replace(r'STARPICKS AUSMAT a gateway to the world','',regex=False)
final_df_TS['content']=final_df_TS['content'].str.replace(r'STARPICKS EMPLOYEE RECOGNITION A FORMULA FOR SUCCESS','',regex=False)
final_df_TS['content']=final_df_TS['content'].str.replace(r'STARPICKS SOCIAL PROTECTION FOR THE SELF-EMPLOYED','',regex=False)
final_df_TS['author']=final_df_TS['author'].str.split('\n').str[0]
final_df_TS['author']=final_df_TS['author'].str.replace(r'By','',regex=False)
final_df_TS['content']=final_df_TS['content'].apply(clean_content)
final_df_TS['title']=final_df_TS['title'].apply(clean_content)
final_df_TS.loc[(final_df_TS['author']=="NA") | (final_df_TS['author']==None),'author']="TheStar"
final_df_TS['author']=final_df_TS['author'].apply(str.upper)
final_df_TS['category']=final_df_TS['category'].apply(str.upper)



# Convert timedate column to datetime format
final_df_TS['timedate']=pd.to_datetime(final_df_TS['timedate'])


# Add new column to add the news source and news type
final_df_TS['newsSource']="TheStar"
final_df_TS['newsType']="Local"

# Extract top 10 news articles
final_df_TS=final_df_TS.head(10)

# Convert the first letter of each words to uppercase in the news title
final_df_TS['title']=final_df_TS['title'].apply(string.capwords)

# Extract year
final_df_TS['newsYearPublished']=final_df_TS['timedate'].dt.strftime("%Y")


# Apply RAKE keywords extraction algorithm
final_df_TS['keywords']=final_df_TS['content'].apply(extract_keywords)


# Calculate the frequency of the top 10 keywords appear in the news contents
final_df_TS['keyword_dict'] = final_df_TS['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])


# Rename the columns to appropriate name
final_df_TS.rename(columns={'title':'newsTitle','content':'newsContent','timedate':'newsTimeDatePublished','image':'newsImageURL','url':'newsURL','category':'newsCategory','author':'newsAuthor','image_url':'newsImageURL'},inplace=True)


# Remove all punctuations from the text documents extracted
final_df_TS['cleaned_text']=final_df_TS['newsContent'].apply(remove_punctuation)

# Tokenize the sentences
final_df_TS['tokenize_sentences']=final_df_TS['cleaned_text'].apply(sent_tokenize)


# Calculate the compound score by using VADER algorithm
final_df_TS['aggregate_compound_score']=final_df_TS['tokenize_sentences'].apply(aggregate_compound_score_sentiment)

# To determine the sentiment of the text document
final_df_TS['sentiment']=final_df_TS['aggregate_compound_score'].apply(sentiment)

# Drop unnecessary columns
final_df_TS.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)

# Clean and preprocess the news contents 
preprocessed_docs = final_df_TS['newsContent'].apply(preprocess_text_details)


final_df_TS['preprocessed_docs']=preprocessed_docs

# Obtain a list of topics and associated terms using LDA model 
top_keywords = final_df_TS['preprocessed_docs'].apply(get_top_keyword)
final_df_TS['topic'],final_df_TS['keywords_topic']=zip(*top_keywords)


# Remove unnecessary data
final_df_TS.drop(columns={'preprocessed_docs'},inplace=True)


# Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
# news content is found
check_duplicate_data(final_df_TS)


# Web Scraping for Bernama News Articles

# Get the link for Bernama news articles
wd.get("https://www.bernama.com/en/")
html2 = BeautifulSoup(wd.page_source,'html.parser')

# Get all the html elements from the respective class
cards_bernama=html2.find_all('div',{"class":"carousel-item"})


# Scrape data for news articles using beautifulSoup - faster data extraction
def scrape_data_bernama(tag):
    try:
        heading=tag
        title=heading.a.getText().strip()
        url=heading.a.get("href")
        category=html2.find("div",{"id":"topstory"})
        news_category=category.get("id")
        author=html2.find("div",{"class":"carousel-caption"})
    except:
        pass
    
    data={'title':title,'url':url,'category':news_category}

    return data





# Scrape data for news articles using beautifulSoup - faster data extraction
def extract_data_bernama():
    ads_data=[]
    for card in cards_bernama:
        data=scrape_data_bernama(card.h1)
        ads_data.append(data)
    
    return ads_data


dict_data_bernama=extract_data_bernama()

# Convert the data into dataframe for further processing
df_bernama=pd.DataFrame.from_dict(dict_data_bernama)


# Declare a webdriver to extract dynamic data
DRIVER_PATH = "C:/Users/Leon/chromedriver.exe"
driver2 = webdriver.Chrome(executable_path=DRIVER_PATH)
driver2.get('https://www.bernama.com/en/')


# Obtain the top 15 news articles
iterator = iter(dict_data_bernama)
articles_BN=[]
for i in range(15):
    try:
        articles=next(iterator)
        articles_BN.append(articles)
    except:
        pass


# Extract relevant data using webdriver (Selenium)
content_values_bernama=[]
for i in articles_BN:#Change variable
    if(i['url'].startswith("world")):
        continue
    else:
        try:
            if(i['url'].startswith("https")):
                driver2.get(i['url'])
            else:
                driver2.get(f"https://www.bernama.com/en/{i['url']}")
            story_body=driver2.find_element(By.CLASS_NAME,'col-lg-8')
            title=story_body.find_element(By.CLASS_NAME,'h2').text.strip()
            image_url=story_body.find_element(By.TAG_NAME,'img').get_attribute("src")
            content=story_body.find_element(By.CLASS_NAME,'row').text
        except:
            pass
        try:
            timedate = story_body.find_element(By.CLASS_NAME,'text-right').text
        except NoSuchElementException as e:
            if(e):
                timedate='None' 

        data={'title':title,'content':content,'timedate':timedate,'image':image_url}
        content_values_bernama.append(data)
        time.sleep(5)
driver2.close()


# Convert dictionary to dataframe
df_content_bernama=pd.DataFrame.from_dict(content_values_bernama)
df_content_bernama

# Apply the clean_content function to clean the documents retrieved
df_content_bernama['title']=df_content_bernama['title'].apply(clean_content)
df_bernama['title']=df_bernama['title'].apply(clean_content)

# Merge all data 
final_df_bernama=df_content_bernama.merge(df_bernama,on='title',how='inner')


# Further pre-process data
final_df_bernama['content']=final_df_bernama['content'].str.split('-- BERNAMA').str[0]
final_df_bernama['content']=final_df_bernama['content'].apply(clean_content)
final_df_bernama['title']=final_df_bernama['title'].apply(clean_content)
final_df_bernama['content']=final_df_bernama['content'].str.replace(r'ADVERTISEMENT','',regex=False)
final_df_bernama['content']=final_df_bernama['content'].str.replace(r'From Nik Nurfaqih Nik Wil','',regex=False)
final_df_bernama['author']='BERNAMA'
final_df_bernama['author']=final_df_bernama['author'].apply(str.upper)
final_df_bernama['category']=final_df_bernama['category'].apply(str.upper)


# Check if the timedate column is in the correct format 
final_df_bernama['formatted_timedate'] = ''
for i, date_str in enumerate(final_df_bernama['timedate']):
    try:
        final_df_bernama.loc[i, 'timedate'] = pd.to_datetime(date_str, format='%d/%m/%Y %I:%M %p')
        final_df_bernama.loc[i, 'formatted_timedate'] = final_df_bernama.loc[i, 'timedate'].strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        final_df_bernama.loc[i, 'timedate'] = pd.to_datetime(date_str, errors='coerce')
        final_df_bernama.loc[i, 'formatted_timedate'] = final_df_bernama.loc[i, 'timedate'].strftime('%Y-%m-%d %H:%M:%S')


# Convert timedate column to datetime format
final_df_bernama['formatted_timedate'] = pd.to_datetime(final_df_bernama['formatted_timedate'])

# Extract year
final_df_bernama['newsYearPublished']=final_df_bernama['formatted_timedate'].dt.strftime("%Y")

# Add new column to add the news source and news type
final_df_bernama['newsSource']="Bernama"
final_df_bernama['newsType']="Local"

# Convert the first letter of each words to uppercase in the news title
final_df_bernama['title']=final_df_bernama['title'].apply(string.capwords)

# Apply RAKE keywords extraction algorithm
final_df_bernama['keywords']=final_df_bernama['content'].apply(extract_keywords)

# Rename the columns to appropriate name
final_df_bernama.rename(columns={'title':'newsTitle','content':'newsContent','formatted_timedate':'newsTimeDatePublished','image':'newsImageURL','url':'newsURL','category':'newsCategory','author':'newsAuthor'},inplace=True)


# Calculate the frequency of the top 10 keywords appear in the news contents
final_df_bernama['keyword_dict'] = final_df_bernama['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])



# Remove all punctuations from the text documents extracted
final_df_bernama['cleaned_text']=final_df_bernama['newsContent'].apply(remove_punctuation)

# Tokenize the sentences
final_df_bernama['tokenize_sentences']=final_df_bernama['cleaned_text'].apply(sent_tokenize)



# Calculate the compound score by using VADER algorithm
final_df_bernama['aggregate_compound_score']=final_df_bernama['tokenize_sentences'].apply(aggregate_compound_score_sentiment)

# To determine the sentiment of the text document
final_df_bernama['sentiment']=final_df_bernama['aggregate_compound_score'].apply(sentiment)



# Drop unnecessary columns
final_df_bernama.drop(columns={'keywords','cleaned_text','tokenize_sentences','timedate'},inplace=True)

# Clean and preprocess the news contents 
preprocessed_docs = final_df_bernama['newsContent'].apply(preprocess_text_details)
final_df_bernama['preprocessed_docs']=preprocessed_docs

# Obtain a list of topics and associated terms using LDA model 
top_keywords = final_df_bernama['preprocessed_docs'].apply(get_top_keyword)
final_df_bernama['topic'],final_df_bernama['keywords_topic']=zip(*top_keywords)

# Remove unnecessary data
final_df_bernama.drop(columns={'preprocessed_docs'},inplace=True)


# Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
# news content is found
check_duplicate_data(final_df_bernama)


# Web Scraping for MalayMail Articles

# Get the link for MalayMail news articles
wd.get("https://www.malaymail.com/")
html3 = BeautifulSoup(wd.page_source,'html.parser')


# Get all the html elements from the respective class
cards_MN=html3.find_all("div",{"class":"col-md-4 article-item"})


# Scrape data for news articles using beautifulSoup - faster data extraction
def scrape_data_MN(tag):
    try:
        heading=tag
        title=heading.a.text.strip()
        url=heading.a.get("href")
    except:
        pass
    data={'title':title,'url':url}

    return data


# Extract data using beautifulSoup
def extract_data_MN():
    ads_data=[]
    for card in cards_MN:
        data=scrape_data_MN(card.h2)
        ads_data.append(data)
    
    return ads_data


dict_data_MN=extract_data_MN()

# Convert the data into dataframe for further processing
df_MN=pd.DataFrame.from_dict(dict_data_MN)

# Obtain the top 15 news articles
iterator = iter(dict_data_MN)
articles_MN=[]
for i in range(15):
    try:
        articles=next(iterator)
        articles_MN.append(articles)
    except:
        pass

# Declare a webdriver to extract dynamic data
DRIVER_PATH = "C:/Users/Leon/chromedriver.exe"
driver3 = webdriver.Chrome(executable_path=DRIVER_PATH)
driver3.get('https://www.malaymail.com/')


# Extract relevant data using webdriver (Selenium)
content_values_MN=[]
for i in articles_MN:#change variable
    if((i['url'].startswith("https://www.malaymail.com/news/singapore")) | (i['url'].startswith("https://www.malaymail.com/news/world"))):
        continue
    else:
        try:
            driver3.get(i['url'])
            story_body=driver3.find_element(By.CLASS_NAME,'malaymail-article-details')
            title=story_body.find_element(By.CLASS_NAME,'article-title').text.strip()
            image_body=story_body.find_element(By.TAG_NAME,'picture')
            image_url=image_body.find_element(By.TAG_NAME,'img').get_attribute('data-src')
            content=story_body.find_element(By.CLASS_NAME,'article-body').text
            article_info=story_body.find_element(By.CLASS_NAME,'article-info')
            timedate=article_info.find_element(By.CLASS_NAME,'article-date').text
        except Exception as e:
            pass
        try:
            article_info=story_body.find_element(By.CLASS_NAME,'article-info')
            timedate=article_info.find_element(By.CLASS_NAME,'article-date').text
        except Exception as e:
            if(e):
                timedate='None'
        try:
            author=article_info.find_element(By.CLASS_NAME,'article-byline').text
        except Exception as e:
            if(e):
                author='MalayMail'
        try:
            category=story_body.find_element(By.CLASS_NAME,'article-section').text
        except Exception as e:
            if(e):
                category='None'

        data={'title':title,'content':content,'timedate':timedate,'author':author,'category':category,'image':image_url}
        content_values_MN.append(data)
        time.sleep(5)
driver3.close()

# Convert dictionary to dataframe
df_content_MN=pd.DataFrame.from_dict(content_values_MN)
df_content_MN


try:
    # Merge all data 
    final_df_MN=df_content_MN.merge(df_MN,on='title',how='inner')
    
    # Apply the clean_content function to clean the documents retrieved
    final_df_MN['content']=final_df_MN['content'].apply(clean_content)
    final_df_MN['title']=final_df_MN['title'].apply(clean_content)
    final_df_MN['author']=final_df_MN['author'].str.replace(r'By','',regex=False)
    final_df_MN['author']=final_df_MN['author'].apply(clean_content)
    final_df_MN.loc[(final_df_MN['author']=="NA") | (final_df_MN['author']==None),'author']="MalayMail"
    final_df_MN['author']=final_df_MN['author'].apply(str.upper)
    final_df_MN['category']=final_df_MN['category'].apply(str.upper)
    
    # Convert timedate column to datetime format
    final_df_MN['timedate']=pd.to_datetime(final_df_MN['timedate'])
    
    # Add new column to add the news source and news type
    final_df_MN['newsSource']="MalayMail"
    final_df_MN['newsType']="Local"
    
    # Convert the first letter of each words to uppercase in the news title
    final_df_MN['title']=final_df_MN['title'].apply(string.capwords)
    
    # Extract year
    final_df_MN['newsYearPublished']=final_df_MN['timedate'].dt.strftime("%Y")
    
    # Apply RAKE keywords extraction algorithm
    final_df_MN['keywords']=final_df_MN['content'].apply(extract_keywords)
    
    # Extract top 10 news articles
    final_df_MN=final_df_MN.head(10)
    
    # Rename the columns to appropriate name
    final_df_MN.rename(columns={'title':'newsTitle','content':'newsContent','timedate':'newsTimeDatePublished','image':'newsImageURL','url':'newsURL','category':'newsCategory','author':'newsAuthor'},inplace=True)
    
    # Calculate the frequency of top 10 keywords extracted appearing in the news content
    final_df_MN['keyword_dict'] = final_df_MN['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])
    
    # Remove all punctuations from the text documents extracted
    final_df_MN['cleaned_text']=final_df_MN['newsContent'].apply(remove_punctuation)
    
    # Tokenize the sentences
    final_df_MN['tokenize_sentences']=final_df_MN['cleaned_text'].apply(sent_tokenize)
    
    # Calculate the compound score by using VADER algorithm
    final_df_MN['aggregate_compound_score']=final_df_MN['tokenize_sentences'].apply(aggregate_compound_score_sentiment)
    
    # To determine the sentiment of the text document
    final_df_MN['sentiment']=final_df_MN['aggregate_compound_score'].apply(sentiment)
    
    # Drop unnecessary columns
    final_df_MN.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)
    
    # Clean and preprocess the news contents 
    preprocessed_docs = final_df_MN['newsContent'].apply(preprocess_text_details)
    final_df_MN['preprocessed_docs']=preprocessed_docs
    
    # Obtain a list of topics and associated terms using LDA model 
    top_keywords = final_df_MN['preprocessed_docs'].apply(get_top_keyword)
    final_df_MN['topic'],final_df_MN['keywords_topic']=zip(*top_keywords)
    
    # Remove unnecessary data
    final_df_MN.drop(columns={'preprocessed_docs'},inplace=True)
    
    # Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
    # news content is found
    check_duplicate_data(final_df_MN)
except:
    pass


# Data Extraction for International News

from newsdataapi import NewsDataApiClient

# API key authorization, Initialize the client with your API key
api = NewsDataApiClient(apikey="pub_19443c07a684dd3473a5c3c5689742adb3fd9")

# You can pass empty or with request parameters {ex. (country = "us")}
response_us = api.news_api(language="en",country="us",category="top")
response_jp = api.news_api(language="en",country="jp",category="top")
response_kr = api.news_api(language="en",country="kr",category="top")
response_cn = api.news_api(language="en",country="cn",category="top")
response_eg = api.news_api(language="en",country="eg",category="top")


# Convert to json format
pretty_json_output = json.dumps(response_us, indent=4)


# Convert to dataframe format
def change_to_dataframe(response):
    response_json_string=json.dumps(response)
    response_dict=json.loads(response_json_string)
    articles_list=response_dict['results']
    articles=pd.DataFrame(articles_list)
    return articles


articles_us=change_to_dataframe(response_us)
articles_jp=change_to_dataframe(response_jp)
articles_kr=change_to_dataframe(response_kr)
articles_cn=change_to_dataframe(response_cn)
articles_eg=change_to_dataframe(response_eg)


# Extract top 15 news articles
articles_us=articles_us.head(15)
articles_jp=articles_jp.head(15)
articles_kr=articles_kr.head(15)
articles_cn=articles_cn.head(15)
articles_eg=articles_eg.head(15)


# Combine the data
df_international=pd.concat([articles_us,articles_jp,articles_kr,articles_cn,articles_eg],ignore_index=0).reset_index(drop=True)


# Convert to date to datetime
df_international['pubDate']=pd.to_datetime(df_international['pubDate'])

# Filter the news articles
df_international_filtered=df_international[(df_international['title'].notnull()) & (df_international['content'].notnull())  & (df_international['pubDate']>=yesterday)].reset_index(drop=True)


# Further pre-process data
df_international_filtered['title']=df_international_filtered['title'].apply(clean_content)
df_international_filtered['content']=df_international_filtered['content'].apply(clean_content)
df_international_filtered['creator']=df_international_filtered['creator'].str[0]
df_international_filtered['category']=df_international_filtered['category'].str[0]
df_international_filtered['country']=df_international_filtered['country'].str[0]
df_international_filtered.loc[(df_international_filtered['creator']=="NA") | (df_international_filtered['creator']==None),'creator']="N/A"
df_international_filtered

# Drop unnecessary columns
df_international_filtered.drop(columns=['keywords','video_url','description','language'],inplace=True)


# Create a new column to determine the news source
df_international_filtered['newsType']="International"

# Convert the first letter of each words to uppercase in the news title
df_international_filtered['title']=df_international_filtered['title'].apply(string.capwords)

# Extract year
df_international_filtered['newsYearPublished']=df_international_filtered['pubDate'].dt.strftime("%Y")

# Apply RAKE keywords extraction algorithm
df_international_filtered['keywords']=df_international_filtered['content'].apply(extract_keywords)

# Extract top 30 news articles
df_international_filtered=df_international_filtered.head(30)

# Rename the columns to appropriate name
df_international_filtered.rename(columns={'title':'newsTitle','content':'newsContent','pubDate':'newsTimeDatePublished','image_url':'newsImageURL','link':'newsURL','category':'newsCategory','creator':'newsAuthor','source_id':'newsSource','country':'newsCountry'},inplace=True)

# Calculate the frequency of the top 10 keywords appear in the news contents
df_international_filtered['keyword_dict'] = df_international_filtered['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])
df_international_filtered

# Remove all punctuations from the text documents extracted
df_international_filtered['cleaned_text']=df_international_filtered['newsContent'].apply(remove_punctuation)

# Tokenize the sentences
df_international_filtered['tokenize_sentences']=df_international_filtered['cleaned_text'].apply(sent_tokenize)
df_international_filtered

# Calculate the compound score by using VADER algorithm
df_international_filtered['aggregate_compound_score']=df_international_filtered['tokenize_sentences'].apply(aggregate_compound_score_sentiment)

# To determine the sentiment of the text document
df_international_filtered['sentiment']=df_international_filtered['aggregate_compound_score'].apply(sentiment)
df_international_filtered

# Drop unnecessary columns
df_international_filtered.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)

# Clean and preprocess the news contents
preprocessed_docs = df_international_filtered['newsContent'].apply(preprocess_text_details)
df_international_filtered['preprocessed_docs']=preprocessed_docs

# Obtain a list of topics and associated terms using LDA model 
top_keywords = df_international_filtered['preprocessed_docs'].apply(get_top_keyword)
df_international_filtered['topic'],df_international_filtered['keywords_topic']=zip(*top_keywords)

# Remove unnecessary data
df_international_filtered.drop(columns={'preprocessed_docs'},inplace=True)

# Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
# news content is found
check_duplicate_data(df_international_filtered)


# To update the duration published(time ago) for every news articles

# Function to calculate the duration published for news contents using timeago library
def calculate_time_ago():
    batch_size = 1000 # adjust batch size as needed
    news_collection = db_news['news-articles'].find()
    updates = []
    count = 0
    for news in news_collection:
        duration_published = timeago.format(news['newsTimeDatePublished'], datetime.now())
        updates.append(UpdateOne({'_id': news['_id']}, {'$set': {'time_ago': duration_published}}))
        count += 1
        if count % batch_size == 0:
            db_news['news-articles'].bulk_write(updates)
            updates = []
            print(f'{count} news duration published updated successfully')
    if updates:
        db_news['news-articles'].bulk_write(updates)
        print(f'{len(updates)} news duration published updated successfully')
    else:
        print('No news duration published updated')


calculate_time_ago()


# Twitter

# Twitter API credential
api_key='bO0CmIAth46lLl7OmDZUwlmZh'
api_secrets='9hczk8HQZvkNr9HU00MkPjOIJftVQJhkwjt9ZrXV9jUA9xiUKx'
access_token = '1456185524893335553-bswgElyLsYB3X21xJEzvauEUxjOuMV'
access_secret= 's6ljSZsQKurdtiaIKv8MsiZ748N3ESkXxr58Qx6mKiDzP'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGLLowEAAAAAfvKvF5hY1hDDsajHS9AD6r%2BiHSo%3DlkseDVzw7JCGdMfVL8MA83C5qF0G86kklnt7C6ofXLuRmLCdvU'
auth = tweepy.OAuthHandler(api_key,api_secrets)
auth.set_access_token(access_token,access_secret)
client = tweepy.Client(
    consumer_key=api_key,
    consumer_secret=api_secrets,
    access_token=access_token,
    access_token_secret=access_secret,
    bearer_token=bearer_token, 
    return_type = requests.Response,
    wait_on_rate_limit=True
)

api=tweepy.API(auth,wait_on_rate_limit=True)


# Check connection to API
if(api.verify_credentials()):
    print('Successful Authentication')
else:
    print('Failed Authentication')


# Twitter Extraction for Bernama

# Obtain tweets from Bernama twitter account
tweets = api.user_timeline(id='bernamadotcom', count=10,tweet_mode='extended')

# Extract all tweets related to news articles
tweets_data=[]
for tweet in tweets:
    id=tweet.id
    date_published=tweet.created_at
    tweet_headlines=tweet.full_text
    retweet_count=tweet.retweet_count
    favorite_count=tweet.favorite_count
    language=tweet.lang
    tweet_url = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
    
    media = tweet.entities.get("media", [])
    if(media):
        if len(media) > 0:
            image_url = media[0]["media_url"]
    else:
        urls = tweet.entities.get('urls', [])

        for url in urls:
            official_url=url['url']

            # Scrape the news URL
            response = requests.get(official_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the image
            image = soup.find('meta', property='og:image')
            if image:
                image_url = image['content']
            else:
                image_url = None
        
    retweets=api.get_retweets(tweet.id)
    retweeters_tag=[retweet.user.screen_name for retweet in retweets]
    retweeters_username=[retweet.user.name for retweet in retweets]

    if(language=='en'):
        tweets_data.append({"tweet_id":id,"created_at":date_published,"tweet_headlines":tweet_headlines,"retweet_count":retweet_count,"favorite_count":favorite_count,
                        "language":language,"tweet_url":tweet_url,"image_url":image_url,"retweeters_tag_name":retweeters_tag,"retweeters_username":retweeters_username, "Source":"Bernama"})
    else:
        continue

try:
    
    # Convert to dataframe
    df_dataframe = pd.DataFrame(tweets_data)
    
    # Apply the clean_content function to clean the documents retrieved
    df_dataframe['tweet_headlines']=df_dataframe['tweet_headlines'].apply(clean_content)
    
    # Drop unnecessary column
    df_dataframe.drop(columns=['language','retweeters_username'],inplace=True)
    
    # Convert timedate column to datetime format
    df_dataframe['created_at']=pd.to_datetime(df_dataframe['created_at'])
    
    # Extract year
    df_dataframe['newsYearPublished']=df_dataframe['created_at'].dt.strftime("%Y")
    
    # Add new column to add the news category and news type
    df_dataframe['newsCategory']="Twitter"
    df_dataframe['newsType']="Tweets"
    
    # Remove all urls
    df_dataframe['tweet_headlines']=df_dataframe['tweet_headlines'].replace(r'http\S+','',regex=True)
    
    # Convert the first letter of each words to uppercase in the news title
    df_dataframe['tweet_headlines']=df_dataframe['tweet_headlines'].apply(string.capwords)
    
    # Apply RAKE keywords extraction algorithm
    df_dataframe['keywords']=df_dataframe['tweet_headlines'].apply(extract_keywords)
    
    # Rename the columns to appropriate name
    df_dataframe.rename(columns={'tweet_headlines':'newsTitle','created_at':'newsTimeDatePublished','image_url':'newsImageURL','tweet_url':'newsURL','Source':'newsSource'},inplace=True)
    
    # Calculate the frequency of the top 10 keywords appear in the news contents
    df_dataframe['keyword_dict'] = df_dataframe['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])
    
    # Remove all punctuations from the text documents extracted
    df_dataframe['cleaned_text']=df_dataframe['newsTitle'].apply(remove_punctuation)
    
    # Tokenize the sentences
    df_dataframe['tokenize_sentences']=df_dataframe['cleaned_text'].apply(sent_tokenize)
    
    # Calculate the compound score by using VADER algorithm
    df_dataframe['aggregate_compound_score']=df_dataframe['tokenize_sentences'].apply(aggregate_compound_score_sentiment)
    
    # To determine the sentiment of the text document
    df_dataframe['sentiment']=df_dataframe['aggregate_compound_score'].apply(sentiment)
    
    # Drop unnecessary columns
    df_dataframe.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)
    
    # Clean and preprocess the news contents 
    preprocessed_docs = df_dataframe['newsTitle'].apply(preprocess_text_details)
    df_dataframe['preprocessed_docs']=preprocessed_docs
    
    # Obtain a list of topics and associated terms using LDA model 
    top_keywords = df_dataframe['preprocessed_docs'].apply(get_top_keyword)
    df_dataframe['topic'],df_dataframe['keywords_topic']=zip(*top_keywords)
    
    # Remove unnecessary data
    df_dataframe.drop(columns={'preprocessed_docs'},inplace=True)
    
    # Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
    # news content is found
    check_duplicate_data(df_dataframe)
except:
    pass


# Twitter Extraction for TheStar

# Obtain tweets from TheStar twitter account
tweets_TheStar = api.user_timeline(id='TheStar_news', count=10,tweet_mode='extended')

# Extract all tweets related to news articles
tweets_data_Star=[]
for tweet in tweets_TheStar:
    if(tweet.full_text.startswith("Watch")):
            continue
    id=tweet.id
    date_published=tweet.created_at
    tweet_headlines=tweet.full_text
    retweet_count=tweet.retweet_count
    favorite_count=tweet.favorite_count
    language=tweet.lang
    tweet_url = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
    
    media = tweet.entities.get("media", [])
    if(media):
        if len(media) > 0:
            image_url = media[0]["media_url"]
    else:
        urls = tweet.entities.get('urls', [])

        for url in urls:
            official_url=url['url']

            # Scrape the news URL
            response = requests.get(official_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the image
            image = soup.find('meta', property='og:image')
            if image:
                image_url = image['content']
            else:
                image_url = None

        
    retweets=api.get_retweets(tweet.id)
    retweeters_tag=[retweet.user.screen_name for retweet in retweets]
    retweeters_username=[retweet.user.name for retweet in retweets]

    if(language=='en'):
        tweets_data_Star.append({"tweet_id":id,"created_at":date_published,"tweet_headlines":tweet_headlines,"retweet_count":retweet_count,"favorite_count":favorite_count,
                        "language":language,"tweet_url":tweet_url,"image_url":image_url,"retweeters_tag_name":retweeters_tag,"retweeters_username":retweeters_username, "Source":"TheStar"})
    else:
        continue

# Convert to dataframe
df_dataframe_TheStar = pd.DataFrame(tweets_data_Star)

# Further pre-process data
df_dataframe_TheStar['tweet_headlines']=df_dataframe_TheStar['tweet_headlines'].apply(clean_content)
df_dataframe_TheStar.drop(columns=['language','retweeters_username'],inplace=True)

# Convert timedate column to datetime format
df_dataframe_TheStar['created_at']=pd.to_datetime(df_dataframe_TheStar['created_at'])

# Extract year
df_dataframe_TheStar['newsYearPublished']=df_dataframe_TheStar['created_at'].dt.strftime("%Y")

# Add new column to add the news source and news type
df_dataframe_TheStar['newsType']="Tweets"
df_dataframe_TheStar['newsCategory']="Twitter"

# Remove URLs
df_dataframe_TheStar['tweet_headlines']=df_dataframe_TheStar['tweet_headlines'].replace(r'http\S+','',regex=True)

# Convert the first letter of each words to uppercase in the news title
df_dataframe_TheStar['tweet_headlines']=df_dataframe_TheStar['tweet_headlines'].apply(string.capwords)

# Apply RAKE keywords extraction algorithm
df_dataframe_TheStar['keywords']=df_dataframe_TheStar['tweet_headlines'].apply(extract_keywords)

# Rename the columns to appropriate name
df_dataframe_TheStar.rename(columns={'tweet_headlines':'newsTitle','created_at':'newsTimeDatePublished','image_url':'newsImageURL','tweet_url':'newsURL','Source':'newsSource'},inplace=True)

# Calculate the frequency of the top 10 keywords appear in the news contents
df_dataframe_TheStar['keyword_dict'] = df_dataframe_TheStar['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])

df_dataframe_TheStar=df_dataframe_TheStar.head(10)

# Remove all punctuations from the text documents extracted
df_dataframe_TheStar['cleaned_text']=df_dataframe_TheStar['newsTitle'].apply(remove_punctuation)

# Tokenize the sentences
df_dataframe_TheStar['tokenize_sentences']=df_dataframe_TheStar['cleaned_text'].apply(sent_tokenize)

# Calculate the compound score by using VADER algorithm
df_dataframe_TheStar['aggregate_compound_score']=df_dataframe_TheStar['tokenize_sentences'].apply(aggregate_compound_score_sentiment)

# To determine the sentiment of the text document
df_dataframe_TheStar['sentiment']=df_dataframe_TheStar['aggregate_compound_score'].apply(sentiment)
df_dataframe_TheStar['sentiment']=df_dataframe_TheStar['aggregate_compound_score'].apply(sentiment)

# Drop unnecessary columns
df_dataframe_TheStar.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)

# Clean and preprocess the news contents 
preprocessed_docs = df_dataframe_TheStar['newsTitle'].apply(preprocess_text_details)
df_dataframe_TheStar['preprocessed_docs']=preprocessed_docs

# Obtain a list of topics and associated terms using LDA model
top_keywords = df_dataframe_TheStar['preprocessed_docs'].apply(get_top_keyword)
df_dataframe_TheStar['topic'],df_dataframe_TheStar['keywords_topic']=zip(*top_keywords)

# Remove unnecessary data
df_dataframe_TheStar.drop(columns={'preprocessed_docs'},inplace=True)

# Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
# news content is found
check_duplicate_data(df_dataframe_TheStar)


# Twitter Extraction for MalayMail

# Obtain tweets from MalayMail twitter account
tweets_MM = api.user_timeline(id='malaymail', count=10,tweet_mode='extended')

# Extract all tweets related to news articles
tweets_data_MM=[]
for tweet in tweets_MM:
    id=tweet.id
    date_published=tweet.created_at
    tweet_headlines=tweet.full_text
    retweet_count=tweet.retweet_count
    favorite_count=tweet.favorite_count
    language=tweet.lang
    tweet_url = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
    
    media = tweet.entities.get("media", [])
    if(media):
        if len(media) > 0:
            image_url = media[0]["media_url"]
        
    else:
        urls = tweet.entities.get('urls', [])

        for url in urls:
            official_url=url['url']

            # Scrape the news URL
            response = requests.get(official_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the image
            image = soup.find('meta', property='og:image')
            if image:
                image_url = image['content']
            else:
                image_url = None
    
    retweets=api.get_retweets(tweet.id)
    retweeters_tag=[retweet.user.screen_name for retweet in retweets]
    retweeters_username=[retweet.user.name for retweet in retweets]

    if(language=='en'):
        tweets_data_MM.append({"tweet_id":id,"created_at":date_published,"tweet_headlines":tweet_headlines,"retweet_count":retweet_count,"favorite_count":favorite_count,
                        "language":language,"tweet_url":tweet_url,"image_url":image_url,"retweeters_tag_name":retweeters_tag,"retweeters_username":retweeters_username, "Source":"MalayMail"})
    else:
        continue

# Convert to dataframe
df_dataframe_MM = pd.DataFrame(tweets_data_MM)

# Extract image URLs
for index, row in df_dataframe_MM.iterrows():
    image_url = row['image_url']
    if isinstance(image_url, str) and image_url.startswith('https://www.malaymail.com/malaymail/uploads/images'):
        df_dataframe_MM.loc[index, 'image_url'] = "https://cdn4.premiumread.com/?url=" + image_url
    elif image_url:
        df_dataframe_MM.loc[index, 'image_url']=image_url
    else:
        # Perform validation or handle the case when the image_url does not start with the expected string
        df_dataframe_MM.loc[index, 'image_url'] = None  # Set it to None or any appropriate value


# Further pre-process data
df_dataframe_MM['tweet_headlines']=df_dataframe_MM['tweet_headlines'].apply(clean_content)
df_dataframe_MM.drop(columns=['language','retweeters_username'],inplace=True)

# Convert timedate column to datetime format
df_dataframe_MM['created_at']=pd.to_datetime(df_dataframe_MM['created_at'])

# Extract year
df_dataframe_MM['newsYearPublished']=df_dataframe_MM['created_at'].dt.strftime("%Y")

# Add new column to add the news source and news type
df_dataframe_MM['newsType']="Tweets"
df_dataframe_MM['newsCategory']="Twitter"

# Remove URLs
df_dataframe_MM['tweet_headlines']=df_dataframe_MM['tweet_headlines'].replace(r'http\S+','',regex=True)

# Convert the first letter of each words to uppercase in the news title
df_dataframe_MM['tweet_headlines']=df_dataframe_MM['tweet_headlines'].apply(string.capwords)

# Apply RAKE keywords extraction algorithm
df_dataframe_MM['keywords']=df_dataframe_MM['tweet_headlines'].apply(extract_keywords)

# Rename the columns to appropriate name
df_dataframe_MM.rename(columns={'tweet_headlines':'newsTitle','created_at':'newsTimeDatePublished','image_url':'newsImageURL','tweet_url':'newsURL','Source':'newsSource'},inplace=True)

# Calculate the frequency of the top 10 keywords appear in the news contents
df_dataframe_MM['keyword_dict'] = df_dataframe_MM['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])

# Extract top 10 tweets
df_dataframe_MM=df_dataframe_MM.head(10)

# Remove punctuations
df_dataframe_MM['cleaned_text']=df_dataframe_MM['newsTitle'].apply(remove_punctuation)

# Tokenize the sentences
df_dataframe_MM['tokenize_sentences']=df_dataframe_MM['cleaned_text'].apply(sent_tokenize)

# Calculate the compound score by using VADER algorithm
df_dataframe_MM['aggregate_compound_score']=df_dataframe_MM['tokenize_sentences'].apply(aggregate_compound_score_sentiment)

# To determine the sentiment of the text document
df_dataframe_MM['sentiment']=df_dataframe_MM['aggregate_compound_score'].apply(sentiment)

# Drop unnecessary columns
df_dataframe_MM.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)

# Clean and preprocess the news contents 
preprocessed_docs = df_dataframe_MM['newsTitle'].apply(preprocess_text_details)
df_dataframe_MM['preprocessed_docs']=preprocessed_docs

# Obtain a list of topics and associated terms using LDA model 
top_keywords = df_dataframe_MM['preprocessed_docs'].apply(get_top_keyword)
df_dataframe_MM['topic'],df_dataframe_MM['keywords_topic']=zip(*top_keywords)

# Remove unnecessary data
df_dataframe_MM.drop(columns={'preprocessed_docs'},inplace=True)

# Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
# news content is found
check_duplicate_data(df_dataframe_MM)


# KoreaHerald

# Obtain tweets from KoreaHerald twitter account
tweets_KoreaHerald = api.user_timeline(id='TheKoreaHerald', count=10,tweet_mode='extended')

# Extract all tweets related to news articles
tweets_data_KoreaHerald=[]
for tweet in tweets_KoreaHerald:
    id=tweet.id
    date_published=tweet.created_at
    tweet_headlines=tweet.full_text
    retweet_count=tweet.retweet_count
    favorite_count=tweet.favorite_count
    language=tweet.lang
    tweet_url = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
    
    media = tweet.entities.get("media", [])
    if(media):
        if len(media) > 0:
            image_url = media[0]["media_url"]
        
    else:
        urls = tweet.entities.get('urls', [])

        for url in urls:
            official_url=url['url']

            # Scrape the news URL
            response = requests.get(official_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the image
            image = soup.find('meta', property='og:image')
            if image:
                image_url = image['content']
            else:
                image_url = None
    
    retweets=api.get_retweets(tweet.id)
    retweeters_tag=[retweet.user.screen_name for retweet in retweets]
    retweeters_username=[retweet.user.name for retweet in retweets]

    if(language=='en'):
        tweets_data_KoreaHerald.append({"tweet_id":id,"created_at":date_published,"tweet_headlines":tweet_headlines,"retweet_count":retweet_count,"favorite_count":favorite_count,
                        "language":language,"tweet_url":tweet_url,"image_url":image_url,"retweeters_tag_name":retweeters_tag,"retweeters_username":retweeters_username, "Source":"KoreaHerald"})
    else:
        continue

# Convert to dataframe
df_dataframe_KoreaHerald = pd.DataFrame(tweets_data_KoreaHerald)

# Further pre-process data
df_dataframe_KoreaHerald['tweet_headlines']=df_dataframe_KoreaHerald ['tweet_headlines'].apply(clean_content)
df_dataframe_KoreaHerald.drop(columns=['language','retweeters_username'],inplace=True)

# Convert timedate column to datetime format
df_dataframe_KoreaHerald['created_at']=pd.to_datetime(df_dataframe_KoreaHerald ['created_at'])

# Extract year
df_dataframe_KoreaHerald['newsYearPublished']=df_dataframe_KoreaHerald ['created_at'].dt.strftime("%Y")

# Add new column to add the news source and news type
df_dataframe_KoreaHerald['newsType']="Tweets"
df_dataframe_KoreaHerald['newsCategory']="Twitter"

# Remove URLs
df_dataframe_KoreaHerald['tweet_headlines']=df_dataframe_KoreaHerald['tweet_headlines'].replace(r'http\S+','',regex=True)

# Convert the first letter of each words to uppercase in the news title
df_dataframe_KoreaHerald['tweet_headlines']=df_dataframe_KoreaHerald['tweet_headlines'].apply(string.capwords)

# Apply RAKE keywords extraction algorithm
df_dataframe_KoreaHerald['keywords']=df_dataframe_KoreaHerald['tweet_headlines'].apply(extract_keywords)

# Rename the columns to appropriate name
df_dataframe_KoreaHerald.rename(columns={'tweet_headlines':'newsTitle','created_at':'newsTimeDatePublished','image_url':'newsImageURL','tweet_url':'newsURL','Source':'newsSource'},inplace=True)
 
# Calculate the frequency of the top 10 keywords appear in the news contents
df_dataframe_KoreaHerald['keyword_dict'] = df_dataframe_KoreaHerald['keywords'].apply(lambda keywords: [{'keyword': keyword[0], 'frequency': keyword[1]} for keyword in keywords])

# Remove all punctuations from the text documents extracted
df_dataframe_KoreaHerald['cleaned_text']=df_dataframe_KoreaHerald['newsTitle'].apply(remove_punctuation)

# Tokenize the sentences
df_dataframe_KoreaHerald['tokenize_sentences']=df_dataframe_KoreaHerald['cleaned_text'].apply(sent_tokenize)
df_dataframe_KoreaHerald

# Calculate the compound score by using VADER algorithm
df_dataframe_KoreaHerald['aggregate_compound_score']=df_dataframe_KoreaHerald['tokenize_sentences'].apply(aggregate_compound_score_sentiment)

# To determine the sentiment of the text document
df_dataframe_KoreaHerald['sentiment']=df_dataframe_KoreaHerald['aggregate_compound_score'].apply(sentiment)

# Drop unnecessary columns
df_dataframe_KoreaHerald.drop(columns={'keywords','cleaned_text','tokenize_sentences'},inplace=True)
df_dataframe_KoreaHerald

# Clean and preprocess the news contents 
preprocessed_docs = df_dataframe_KoreaHerald['newsTitle'].apply(preprocess_text_details)
df_dataframe_KoreaHerald['preprocessed_docs']=preprocessed_docs

# Obtain a list of topics and associated terms using LDA model 
top_keywords = df_dataframe_KoreaHerald['preprocessed_docs'].apply(get_top_keyword)
df_dataframe_KoreaHerald['topic'],df_dataframe_KoreaHerald['keywords_topic']=zip(*top_keywords)

# Remove unnecessary data
df_dataframe_KoreaHerald.drop(columns={'preprocessed_docs'},inplace=True)


# Check for any duplicate news title in database, the data and related information is inserted into the database if no duplicate
# news content is found
check_duplicate_data(df_dataframe_KoreaHerald)


# To update the duration published(time ago) for every news articles

# Function to calculate the duration published for news contents using timeago library
def calculate_time_ago():
    batch_size = 1000 # adjust batch size as needed
    news_collection = db_news['news-articles'].find()
    updates = []
    count = 0
    for news in news_collection:
        duration_published = timeago.format(news['newsTimeDatePublished'], datetime.now())
        updates.append(UpdateOne({'_id': news['_id']}, {'$set': {'time_ago': duration_published}}))
        count += 1
        if count % batch_size == 0:
            db_news['news-articles'].bulk_write(updates)
            updates = []
            print(f'{count} news duration published updated successfully')
    if updates:
        db_news['news-articles'].bulk_write(updates)
        print(f'{len(updates)} news duration published updated successfully')
    else:
        print('No news duration published updated')

calculate_time_ago()







