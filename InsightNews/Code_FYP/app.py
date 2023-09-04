# Initialize the libraries
from flask import Flask, render_template, request, jsonify, make_response, session, url_for, redirect, flash, Markup
from pymongo import MongoClient, UpdateOne
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import datetime, timedelta, date
import time as t
import timeago
import numpy as np
from flask_paginate import Pagination, get_page_args
from wordcloud import WordCloud
import base64
from io import BytesIO
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import requests
import os

# Initialize flask library
app = Flask(__name__)
app.secret_key = "super secret key"
CORS(app)

# Mongo configuration
client = MongoClient(
    "*****")
db_news = client['news-content']
db_users_collection = db_news['user']

# Get current date
today = datetime.combine(date.today(), datetime.min.time())
now = datetime.now()

# Get yesterday date
yesterday = datetime.combine(
    date.today() - timedelta(days=1), datetime.min.time())

# Get the day before yesterday
yesterday_yesterday = datetime.combine(
    date.today() - timedelta(days=2), datetime.max.time())


# Get the current month and year
current_month = datetime.now().month
current_year = datetime.now().year

# Calculate the previous month
previous_month = current_month - 1
if previous_month == 0:  # If the current month is January
    previous_month = 12
    current_year -= 1

# Call the LDA model
lda_model_optimal = LdaModel.load('static/LDA_model/lda_model_file_25_5_2023')


# Password Validation
def is_valid_password(password):
    """
    Check if a password meets the following criteria:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one special character
    """
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True

# Calculate the duration published for every news articles in database


def calculate_time_ago():
    batch_size = 1000  # adjust batch size as needed
    news_collection = db_news['news-articles'].find()
    updates = []
    count = 0
    for news in news_collection:
        duration_published = timeago.format(news['newsTimeDatePublished'], now)
        updates.append(UpdateOne({'_id': news['_id']}, {
                       '$set': {'time_ago': duration_published}}))
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

# Calculate trending score based on the initial keyword frequency found and the current aggregate keyword frequency


def calculate_trending_score(current_frequency, previous_frequency):
    return 10 * (1 - math.exp(-((current_frequency - previous_frequency) / previous_frequency)))

# Determine sentiment based on sentiment polarity found


def find_sentiment(x):
    if x >= 0.35:
        sentiment = 'Positive'
    elif x <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment


# List of contractions for text pre-processing
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}

# Obtain a list of stopwords from internet
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines())
additional_stopwords = {"cent", "starpicks"}
# List of stopwords that will be removed in text
stopwords_final = stopwords | additional_stopwords
lemmatizer = WordNetLemmatizer()  # Initialize wordnet lemmatizer


def expand(x):
    """Some of the words like 'i'll', are expanded to 'i will' for better text processing
    The list of contractions is taken from the internet

    param x(str): the sentence in which contractions are to be found and expansions are to be done

    return x(str): the expanded sentence"""
    if type(x) == str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x

# Perform a series of text preprocessing steps


def preprocess_text(text):
    text = text.lower()  # convert text to lowercase
    text = expand(text)  # expand contractions
    # removes all characters that are not letters, digits, spaces, or hyphens from the text
    text = re.sub('[^A-Z a-z 0-9-]+', '', text)
    # removes punctuation marks (commas, periods, exclamation marks, and question marks) from the text
    text = re.sub('[,\.!?]', '', text)
    text = re.sub(r'http\S+', '', text)  # remove URL
    text = text.lower().split()  # Split into individual words and converted to lowercase
    # Remove stopwords
    text = [word for word in text if not word in stopwords_final]
    text = [lemmatizer.lemmatize(word)
            for word in text]  # Perform lemmatization
    return text

# Convert datetime to format 'dd/mm/YYYY HH:MM'


@app.template_filter('format_datetime')
def format_datetime(value, format='%d %b %Y %I:%M %p'):
    if isinstance(value, str):
        value = datetime.strptime(value, '%d/%m/%Y %H:%M')
    return value.strftime(format)

# Convert date to format 'YYYY-mm-dd'


@app.template_filter('format_date')
def format_date(value, format='%d %b %Y'):
    if isinstance(value, str):
        value = datetime.strptime(value, '%Y-%m-%d')
    return value.strftime(format)

# Convert numerical month value to full month name


@app.template_filter('format_month')
def format_month(value, format='%B'):
    return datetime.strptime(str(value), '%m').strftime(format)

# Authentication page


@app.route('/authenticate', methods=['POST', 'GET'])
def authenticate():

    # If the Sign In button is clicked, check the entered email and password in database for verification
    if request.method == 'POST':
        if 'login' in request.form:
            body = request.form
            email = body['email']  # Obtain entered email
            password = body['password']  # Obtain entered password
            # Search for the entered email in the database
            search_user = db_users_collection.find_one({"email": email})

            # If email is found, check the entered password with the password in the database. Otherwise, prompt user to enter the correct email address
            if search_user:
                if check_password_hash(str(search_user['password']), str(password)):
                    session['username'] = search_user['username']
                    session['email'] = search_user['email']
                    # Redirect to the home page if the authentication successful
                    return redirect(url_for('home'))
                else:
                    flash('Wrong Password', category='warning')
                    error = 'Wrong Password'
                    # Prompt user to re-enter password due to incorrect password
                    return render_template('login-register.html', emailAddr=email, error_login=error)
            else:
                flash('Email not found', category='warning')
                error = 'Email not found'
                # Prompt user to re-enter email due to incorrect email address
                return render_template('login-register.html', error_login=error)

        # If the register button is clicked, save the entered username, email and password in database
        if 'register' in request.form:

            # Send input to MongoDB
            body = request.form
            username = body['username']
            email = body['email']
            password = body['password']
            confirmed_password = body['confirmed_password']

            # Check for any existing email address in database
            existing_user = db_users_collection.find_one({'email': email})
            if existing_user:
                flash(email + ' is already exists')
                error = email + ' is already exists'
                return render_template("login-register.html", name=username, error_register=error)

            # Prompt user to enter valid password
            elif not is_valid_password(password):
                flash('Invalid password. Passwords must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character', category='warning')
                error = 'Invalid password. Passwords must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character'
                return render_template("login-register.html", name=username, error_register=error)

            # Prompt user to re-enter password for confirmation
            elif password != confirmed_password:
                flash('Password do not match', category='warning')
                error = 'Password do not match'
                return render_template("login-register.html", name=username, emailAddr=email, error_register=error)
            else:
                hashed_password = generate_password_hash(
                    str(password))  # Encrypt the password
                db_users_collection.insert_one({
                    "username": username,
                    "email": email,
                    "password": hashed_password,
                    "savedContent": []
                })
                session['username'] = username
                session['email'] = email
                # Redirect user to home page after registration is successful
                return redirect(url_for('home'))
    else:
        return render_template("login-register.html")

# Forgot Password page


@app.route('/forgot', methods=['POST', 'GET'])
def forgot_password():

    # Email verification
    if request.method == 'POST':
        if 'forgot' in request.form:
            body = request.form
            email = body['email']  # Obtain entered email address

            # Match the entered email address to the email address in database for verification
            update_user = db_users_collection.find_one({"email": email})
            if update_user:
                session['email_captured'] = email
                status = True
                # Redirect user to update password page if email is found
                return render_template('forgot-password.html', status=status)
            else:
                flash('Email not found', category='warning')
                error = 'Email not found'
                # Prompt user to re-enter email due to incorrect email
                return render_template('forgot-password.html', error_forgot=error)

        # Password Update
        if 'update_pass' in request.form:
            body = request.form
            password = body['password']
            confirmed_password = body['confirmed_password']

            # Password validation
            if not is_valid_password(password):
                flash('Invalid password. Passwords must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character', category='warning')
                error = 'Invalid password. Passwords must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character'
                return render_template("forgot-password.html", error_forgot=error)
            elif password != confirmed_password:
                flash('Password do not match', category='warning')
                error = 'Password do not match'
                return render_template("forgot-password.html", error_forgot=error)
            else:
                hashed_password = generate_password_hash(str(password))
                query = {"email": session['email_captured']}
                new_values = {"$set": {"password": hashed_password}}
                db_users_collection.update_one(query, new_values)
                flash('Password has been successfully updated', category='success')
                return redirect(url_for('authenticate'))
    else:
        return render_template('forgot-password.html')


# Features in home page
@app.route('/home', methods=['POST', 'GET'])
def home():

    # Get the latest and current local news articles
    current_news_local = db_news['news-articles'].find({
        "newsTimeDatePublished": {"$gte": yesterday},
        "newsType": "Local"}).sort("newsTimeDatePublished", -1)

    # Get the latest and current international news articles
    current_news_international = db_news['news-articles'].find({
        "newsTimeDatePublished": {"$gte": yesterday},
        "newsType": "International"}).sort("newsTimeDatePublished", -1)

    # Get the latest and current tweets news articles
    current_news_tweets = db_news['news-articles'].find({
        "newsTimeDatePublished": {"$gte": yesterday},
        "newsType": "Tweets"}).sort("newsTimeDatePublished", -1)

    # Obtains non-duplicate and unique tweets news
    unique_results = []
    for tweet_results in current_news_tweets:
        if tweet_results['newsTitle'] not in [n['newsTitle'] for n in unique_results]:
            unique_results.append(tweet_results)

    # Query the top 3 most recent news contents from Money, Malaysia, Sports and Business categories
    result_category = {}
    money_news = db_news['news-articles'].find(
        {'newsType': 'Local', 'newsCategory': 'MONEY'}).sort('newsTimeDatePublished', -1).limit(3)
    money_news = list(money_news)

    malaysia_news = db_news['news-articles'].find(
        {'newsType': 'Local', 'newsCategory': 'MALAYSIA'}).sort('newsTimeDatePublished', -1).limit(3)
    malaysia_news = list(malaysia_news)

    sports_news = db_news['news-articles'].find(
        {'newsType': 'Local', 'newsCategory': 'SPORTS'}).sort('newsTimeDatePublished', -1).limit(3)
    sports_news = list(sports_news)

    business_news = db_news['news-articles'].find(
        {'newsType': 'International', 'newsCategory': 'business'}).sort('newsTimeDatePublished', -1).limit(3)
    business_news = list(business_news)

    # Combine the results for the category
    result_category = {
        'Money': money_news,
        'Malaysia': malaysia_news,
        'Sports': sports_news,
        'Business': business_news
    }

    # Get the popular news contents from database where the trending score > 3
    popular_news = list(db_news['news-articles'].aggregate([
        {"$match": {"newsTrendingScore": {"$gt": 3},
                    "newsTimeDatePublished": {"$lte": yesterday_yesterday}}},
        {"$sort": {"newsTimeDatePublished": -1, "newsTrendingScore": -1}}
    ]))

    # Get the top 2 highest trending score and latest news contents for Local, International and Tweets news. This is for image slider carousel
    highlights_news = []
    news_types = ['Local', 'International', 'Tweets']
    for news_type in news_types:
        # Query the latest 2 documents for each newsType field
        news_docs = db_news['news-articles'].find({'newsType': news_type, 'newsTimeDatePublished': {'$gte': yesterday}, "newsTrendingScore": {
                                                  "$gt": 3}}, {'_id': 0}).sort([('newsTimeDatePublished', -1), ('newsTrendingScore', -1)]).limit(2)
        # Convert the pymongo Cursor to a list of dicts
        news_docs = list(news_docs)
        highlights_news.extend(news_docs)

    # Show the 4 recommended news contents randomly
    recommended_news = db_news['news-articles'].aggregate(
        [{"$sample": {"size": 4}}])

    # Bookmark news articles or tweets, save in database and display in library if bookmark button is clicked
    if request.method == "POST":
        if 'bookmark-btn' in request.form:
            title = request.form.get('bookmark-main')
            check_bookmark = db_users_collection.find_one(
                {"username": session['username'], "savedContent": title})
            if check_bookmark:
                flash(
                    "Unable to save the selected article because the article has been saved in library", category='warning')
            else:
                db_users_collection.update_one(
                    {'username': session['username']},
                    {'$push': {'savedContent': title}}
                )
                flash("Article has been successfully saved in your library",
                      category='success')

    return render_template('main_page.html', current_news_local=current_news_local, current_news_international=current_news_international, current_news_tweets=current_news_tweets, recommend_news_info=recommended_news, popular_news_info=popular_news, unique_results=unique_results, highlights_news=highlights_news, result_category=result_category)

# Features in search results page


@app.route('/search', methods=['POST', 'GET'])
def search():

    # If the search button is clicked, the relevant news articles or tweets related to user query are displayed based on user query
    if request.args.get('filter-search'):
        search_string = request.args.get('search-filter')  # Get the user query
        # Perform case insensitive search
        regex_pattern = re.compile(
            r".*?{}.*?".format(search_string), re.IGNORECASE)
        all_news_matching = db_news['news-articles'].find({"newsTitle": regex_pattern}).sort(
            "newsTimeDatePublished", -1)  # Matching news contents are received

        per_page = 12  # Maximum news contents displayed in a single page
        page = int(request.args.get('page', 1))

        # Retrieve total number of items in collection
        total_search = db_news['news-articles'].count_documents(
            {"newsTitle": regex_pattern})

        # Retrieve items for the current page using offset and limit
        items_matching = all_news_matching.skip(
            (page-1)*per_page).limit(per_page)

        # Create pagination object using Flask-Paginate
        pagination_search = Pagination(page=page, per_page=per_page,
                                       total=total_search, css_framework='bootstrap4')

        # Bookmark news articles and save in library if bookmark button is clicked
        if 'bookmark-btn' in request.form:
            title = request.form.get('bookmark-main')
            check_bookmark = db_users_collection.find_one(
                {"username": session['username'], "savedContent": title})

            # Check for duplicate news content saved in library
            if check_bookmark:
                flash(
                    "Unable to save the selected article because the article has been saved in library", category='warning')
            else:
                db_users_collection.update_one(
                    {'username': session['username']},
                    {'$push': {'savedContent': title}}
                )
                flash("Article has been successfully saved in your library",
                      category='success')

        return render_template('search_result.html', matching_results=items_matching, search_string=search_string, pagination=pagination_search)

    # If the filter button is clicked, the relevant news articles or tweets related to user query are displayed based on date and categories
    if request.args.get('filter_category_date'):
        date_selection = request.args.get(
            'selection')  # Obtain the selected date
        category_selection = request.args.getlist(
            'selection_box')  # Obtain a list of categories

        if date_selection or category_selection:
            # Filter your data based on the selected option
            if date_selection == 'Last 24 hours':
                # Filter data for the last 24 hours
                filter = {"time_ago": {"$regex": "hours|hour"}}
            elif date_selection == 'By Weeks':
                # Filter data for the last week
                filter = {"time_ago": {"$regex": "week|weeks"}}
            elif date_selection == 'By Months':
                # Filter data for the last month
                filter = {"time_ago": {"$regex": "month|months"}}
            elif category_selection:
                # If categories are selected but no date range is specified, filter only by categories
                filter = {'newsCategory': {'$in': category_selection}}

            # If categories were selected, add them to the filter
            if category_selection:
                filter['newsCategory'] = {"$in": category_selection}

            # Query the database based on the filter
            results = db_news['news-articles'].find(
                filter).sort("newsTimeDatePublished", -1)

            per_page = 12  # Maximum news contents displayed in a single page
            page = int(request.args.get('page', 1))

            # Retrieve total number of items in collection
            total = db_news['news-articles'].count_documents(filter)

            # Retrieve items for the current page using offset and limit
            items = results.skip((page-1)*per_page).limit(per_page)

            # Create pagination object using Flask-Paginate
            pagination = Pagination(page=page, per_page=per_page,
                                    total=total, css_framework='bootstrap4')

            # Bookmark news articles and save in library if bookmark button is clicked
            if 'bookmark-btn' in request.form:
                title = request.form.get('bookmark-main')
                check_bookmark = db_users_collection.find_one(
                    {"username": session['username'], "savedContent": title})
                if check_bookmark:
                    flash(
                        "Unable to save the selected article because the article has been saved in library", category='warning')
                else:
                    db_users_collection.update_one(
                        {'username': session['username']},
                        {'$push': {'savedContent': title}}
                    )
                    flash("Article has been successfully saved in your library",
                          category='success')

            return render_template('search_result.html', results=items, date_selection=date_selection, category_selection=category_selection, pagination=pagination)
        else:
            flash("No filter is clicked. Please choose a filter", category='warning')

    return render_template('search_result.html')

# Library page


@app.route('/library', methods=['POST', 'GET'])
def library():
    if 'username' in session:

        # Obtain all saved news contents from database based on the current username
        library = db_users_collection.find_one(
            {"username": session['username']})
        saved_contents = library['savedContent']
        matching_docs = db_news['news-articles'].find(
            {"newsTitle": {"$in": saved_contents}}).sort("newsTimeDatePublished", -1)

        # Delete news articles in library
        if request.method == "POST":
            if 'delete-article' in request.form:
                # If delete button is clicked
                title = request.form.get('delete')
                db_users_collection.update_one(
                    {'username': session['username']},
                    {'$pull': {'savedContent': title}}
                )
                flash("The selected article has been successfully deleted",
                      category='success')
                return redirect(url_for('library'))

        return render_template('library.html', news_info=matching_docs)
    else:
        # Redirect user to authenticate page if user is not in session or found
        return redirect(url_for('authenticate'))

# News contents details page


@app.route('/main_details', methods=['POST', 'GET'])
def news_details():
    if 'username' in session:

        # Display additional information about a particular news contents
        # Get the news title of the selected news contents
        title = request.args.get('name')
        if title:

            # Calculate the number of clicks for each news article
            result = db_news['news-articles'].aggregate([
                {'$group': {'_id': None, 'total_clicks': {'$max': '$click_count'}}}
            ])
            total_clicks = result.next()['total_clicks']

            # Incrementing the number of clicks count of the selected news article
            db_news['news-articles'].update_one(
                {'newsTitle': title}, {'$inc': {'click_count': 1}}, upsert=True)

            # Calculate the trending score based on number of clicks and recent news contents
            articles = db_news['news-articles'].find()
            for article in articles:

                # Ensures that only articles with at least one click are considered for the trending score calculation.
                clicks = article.get('click_count', 0)
                if clicks == 0:
                    continue

                publication_date = article['newsTimeDatePublished']
                days_since_publication = (datetime.now() - publication_date).days
                clicks_score = np.log10(clicks + 1) * (10 / np.log10(total_clicks + 1))
                # decay factor of 1/7 (corresponding to 1 week) to calculate how much impact the recency of publication has on the score
                time_score = np.exp(-days_since_publication / 7) * 10
                # weighted combination of clicks and time
                trending_score = round(
                    (0.7 * clicks_score) + (0.3 * time_score))
                db_news['news-articles'].update_one(
                    {'newsTitle': article['newsTitle']}, {'$set': {'newsTrendingScore': trending_score}})  # Update the trending score that ranges from 0-10

            # Provide 4 news contents randomly
            news_home = db_news['news-articles'].find_one({"newsTitle": title})
            recommended_news = db_news['news-articles'].aggregate([
                {"$match": {
                    "newsCategory": news_home['newsCategory'], "newsTitle":{"$ne": title}}},
                {"$sample": {"size": 4}}
            ])

            # Plot word cloud
            if news_home['newsType'] == "Tweets":
                wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                                      contour_color='steelblue').generate(news_home['newsTitle'])
            else:
                wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                                      contour_color='steelblue').generate(news_home['newsContent'])

            # convert the image to bytes
            img_bytes = BytesIO()
            wordcloud.to_image().save(img_bytes, format='PNG')
            img_bytes.seek(0)
            wc_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            # Obtain keyword and frequency to plot bar graph
            labels = []
            freq = []
            keyword_dict = news_home['keyword_dict']
            for field in keyword_dict:
                labels.append(field['keyword'])
                freq.append(field['frequency'])

        if request.method == "POST":

            # If the bookmark button of the recommended news content is clicked, save the selected content into the library
            if 'bookmark-btn' in request.form:
                title_related_news = request.form.get('bookmark-main')
                check_bookmark = db_users_collection.find_one(
                    {"username": session['username'], "savedContent": title_related_news})
                if check_bookmark:
                    flash(
                        "Unable to save the selected article because the article has been saved in library", category='warning')
                else:
                    db_users_collection.update_one(
                        {'username': session['username']},
                        {'$push': {'savedContent': title_related_news}}
                    )
                    flash(
                        "Article has been successfully saved in your library", category='success')

            # If the bookmark button of the selected news content is clicked, save the selected content into the library
            if 'bookmark' in request.form:
                check_title_bookmark = db_users_collection.find_one(
                    {"username": session['username'], "savedContent": title})
                if check_title_bookmark:
                    flash(
                        "Unable to save the selected article because the article has been saved in library", category='warning')
                else:
                    db_users_collection.update_one(
                        {'username': session['username']},
                        {'$push': {'savedContent': title}}
                    )
                    flash(
                        "Article has been successfully saved in your library", category='success')

        return render_template('news-details.html', news_details=news_home, recommend_news=recommended_news, labels=labels, freq=freq, wc_base64=wc_base64)
    else:
        flash(Markup(
            "Please register as member first before accessing the feature"), category='warning')
        return redirect(url_for('home'))

# Keyword analysis page


@app.route('/keyword', methods=["GET", "POST"])
def keyword():
    if 'username' in session:
        output = []

        # Perform keywords aggregation obtained from the tweets news
        keyword_results = db_news['news-articles'].aggregate([
            {"$match": {"newsType": "Tweets"}},
            {'$unwind': "$keyword_dict"},
            {'$group': {'_id': '$keyword_dict.keyword', 'current_total_frequency': {'$sum': '$keyword_dict.frequency'}, "previous_frequency": {"$last": "$keyword_dict.frequency"}, "news": {"$addToSet": {
                "title": "$newsTitle", "url": "$newsURL", "type": "$newsType", "source": "$newsSource", "timeDate": "$newsTimeDatePublished", "imageURL": "$newsImageURL", "retweet": "$retweet_count", "favorite": "$favorite_count"}}}},
            {'$sort': {"current_total_frequency": -1}}
        ])

        analyzer = SentimentIntensityAnalyzer()

        # Perform VADER sentiment analysis, calculate accumulated keyword frequency, calculate trending score and obtain relevant tweets for every keywords
        for result in keyword_results:
            # Calculate polarity score using VADER
            score = analyzer.polarity_scores(result['_id'])['compound']
            # Identify sentiment based on sentiment polarity score
            sentiment = find_sentiment(score)
            if result['current_total_frequency'] == 0 or result['previous_frequency'] == 0:
                result['current_total_frequency'] = result['current_total_frequency']+1
                result['previous_frequency'] = result['previous_frequency']+1

            if result['current_total_frequency'] > 0 and result['previous_frequency'] > 0:
                trending_score = calculate_trending_score(
                    result["current_total_frequency"], result["previous_frequency"])
            else:
                trending_score = 0

            # Obtain a list of relevant tweets news for every keywords
            news_list = [{"title": news["title"], "url": news["url"], "type":news["type"], "source":news["source"], "timeDate":news["timeDate"],
                          "imageURL":news["imageURL"], "retweet":news["retweet"], "favorite":news["favorite"]} for news in result["news"]]
            # Sort tweets based on highest accumulation of favorite and retweet counts
            sorted_news_list = sorted(news_list, key=lambda news: (
                news['favorite'] + news['retweet']), reverse=True)
            output.append({"keyword": result["_id"], "current_total_frequency": result["current_total_frequency"],
                           "previous_frequency": result["previous_frequency"], "news": sorted_news_list,
                           "polarity_score": round(score, 2), "sentiment": sentiment, "keyword_trending_score": int(trending_score)})

        per_page = 10  # Allow for only 10 keywords in one page
        page = int(request.args.get('page', 1))

        # Retrieve total number of items in collection
        total = len(output)

        # Calculate the start and end index for the current page
        start_index = (page - 1) * per_page
        end_index = start_index + per_page

        # Retrieve items for the current page using list slicing
        items = output[start_index:end_index]

        # Create pagination object using Flask-Paginate
        pagination = Pagination(page=page, per_page=per_page,
                                total=total, css_framework='bootstrap4')

        # Function to provide the summary of the overall keyword analysis
        # Identify total number of keywords analyzed
        total_count_keyword = len(output)
        # Identify the keyword with highest frequency
        max_frequency_dict = max(
            output, key=lambda x: x['current_total_frequency'])
        # Identify the keyword with highest trending score
        max_trending_dict = max(
            output, key=lambda x: x['keyword_trending_score'])
        keyword_summary = []
        keyword_summary.append({"total_count": total_count_keyword, "max_keyword_freq": max_frequency_dict['keyword'],
                                "max_freq": max_frequency_dict['current_total_frequency'], "max_keyword_trending": max_trending_dict['keyword'],
                                "max_trending_score": max_trending_dict['keyword_trending_score']})

        # Obtain all lists of keywords sorted by highest frequency for today
        keyword_today = db_news['news-articles'].aggregate(
            [
                {"$match": {
                    "newsType": "Tweets",
                    "newsTimeDatePublished": {"$gte": today}
                }},
                {'$unwind': "$keyword_dict"},
                {'$group': {
                    '_id': '$keyword_dict.keyword',
                    'current_total_frequency': {'$sum': '$keyword_dict.frequency'},
                    "previous_frequency": {"$last": "$keyword_dict.frequency"},
                }},
                {'$sort': {"current_total_frequency": -1, "_id": 1}}
            ])

        # Calculate the trending score and identify the keywords with highest trending score and frequency
        keyword_today_list = []
        for result in keyword_today:
            if result['current_total_frequency'] > 0 and result['previous_frequency'] > 0:
                # Calculate the trending score based on frequency
                trending_score = calculate_trending_score(
                    result["current_total_frequency"], result["previous_frequency"])
            else:
                trending_score = 0
            keyword_today_list.append({"keyword": result["_id"], "current_total_frequency": result["current_total_frequency"],
                                      "previous_frequency": result["previous_frequency"], "keyword_trending_score": int(trending_score)})

        if keyword_today_list:
            # Select the keywords with highest frequency and trending score
            trending_keyword_today = max(keyword_today_list, key=lambda x: (
                x['current_total_frequency'], x['keyword_trending_score']))
        else:
            trending_keyword_today = "Null"

        # Function to allow user to search for specific keywords by entering query
        if request.method == "POST":
            if 'search_button' in request.form:
                search_string = request.form.get("search_string")
                regex_pattern = re.compile(
                    r".*?{}.*?".format(search_string), re.IGNORECASE)
                matching_results = []
                for result in output:
                    if regex_pattern.match(result["keyword"]):
                        matching_results.append(result)

                return render_template("keyword-analysis.html", matching_keywords=matching_results, search_string=search_string, total_count=total_count_keyword, keyword_summary=keyword_summary, trending_keyword_today=trending_keyword_today)
        return render_template("keyword-analysis.html", keyword_info=items, pagination=pagination, total_count=total_count_keyword, keyword_summary=keyword_summary, trending_keyword_today=trending_keyword_today)
    else:
        flash(Markup(
            "Please register as member first before accessing the feature"), category='warning')
        return redirect(url_for('home'))

# Topic Modelling Visualization using Bubble Plot page


@app.route('/topic_visualize', methods=["GET", "POST"])
def topic_visualize():

    if 'username' in session:
        # Get the current year and previous month
        current_month_year = datetime.now().year
        current_month = datetime.now().month

        # Perform the aggregation query to retrieve distinct months
        pipeline = [
            {
                "$project": {
                    "month": {"$month": "$newsTimeDatePublished"}
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

        # Display topic modelling bubble plot using data filtered by user based on month
        if request.args.get('filter_month'):
            if request.args.get('select_month'):
                month_selection = int(request.args.get('select_month'))

                # Check if pyLDAvis was already generated for the selected month
                html_file_name = f"pyLDAvis_{month_selection}_{current_month_year}.html"
                html_file_path = os.path.join(
                    app.root_path, 'templates', html_file_name)

                # If the html file is not found for the selected month, create the html of the selected month automatically
                if not os.path.exists(html_file_path):
                    preprocessed_text = []
                    for doc in db_news['news-articles'].find({
                        "newsType": "Tweets",
                        "newsTimeDatePublished": {
                            "$gte": datetime(current_month_year, month_selection, 1),
                            "$lt": datetime(current_month_year, month_selection + 1, 1)
                        }
                    }):
                        text = preprocess_text(doc["newsTitle"])
                        preprocessed_text.append(text)

                    bow_corpus = [lda_model_optimal.id2word.doc2bow(
                        doc) for doc in preprocessed_text]  # Corpus from the LDA model generated
                    # Plot bubble plot using pyLDAvis library
                    vis_data = gensimvis.prepare(
                        lda_model_optimal, bow_corpus, lda_model_optimal.id2word, sort_topics=False)
                    # Save the visualization in html format
                    pyLDAvis.save_html(vis_data, html_file_path)

                return render_template('topic_visualize.html', month_selection=month_selection, current_month_year=current_month_year, distinct_months=distinct_months)
            else:
                flash(Markup("Please select a month to filter"),
                      category='warning')
                return redirect(url_for('topic_visualize'))
        return render_template('topic_visualize.html', current_month=current_month, current_month_year=current_month_year, distinct_months=distinct_months)
    else:
        flash(Markup(
            "Please register as member first before accessing the feature"), category='warning')
        return redirect(url_for('home'))

# Trending tweets categorized by topics page


@app.route('/topic')
def topic():
    if 'username' in session:

        # Pipeline to query the information of top 3 trending tweets for each topic that are sorted based on highest retweet and favorite counts
        pipeline = [
            {
                "$match": {
                    "topic": {"$ne": None},
                    "newsType": "Tweets"
                }
            },
            {
                "$group": {
                    "_id": "$topic",
                    "newsTitles": {
                        "$push": {
                            "newsTitle": "$newsTitle",
                            "newsImageURL": "$newsImageURL",
                            "newsTimeDatePublished": "$newsTimeDatePublished",
                            "sentiment": "$sentiment",
                            "retweet_count": "$retweet_count",
                            "favorite_count": "$favorite_count",
                            "combined_count": {"$sum": ["$retweet_count", "$favorite_count"]}
                        }
                    }
                }
            },
            {
                "$unwind": "$newsTitles"
            },
            {
                "$sort": {
                    "newsTitles.retweet_count": -1,
                    "newsTitles.favorite_count": -1,
                    "newsTitles.combined_count": -1
                }
            },
            {
                "$group": {
                    "_id": "$_id",
                    "newsTitles": {
                        "$push": {
                            "newsTitle": "$newsTitles.newsTitle",
                            "newsImageURL": "$newsTitles.newsImageURL",
                            "newsTimeDatePublished": "$newsTitles.newsTimeDatePublished",
                            "sentiment": "$newsTitles.sentiment",
                        }
                    },
                    "retweet_counts": {
                        "$push": "$newsTitles.retweet_count"
                    },
                    "favorite_counts": {
                        "$push": "$newsTitles.favorite_count"
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id",
                    "newsTitles": {
                        "$slice": ["$newsTitles", 3]
                    },
                    "retweet_counts": {
                        "$slice": ["$retweet_counts", 3]
                    },
                    "favorite_counts": {
                        "$slice": ["$favorite_counts", 3]
                    }
                }
            },
            {
                "$sort": {
                    "topic": 1
                }
            }
        ]

        # Convert the result of the pipeline to list of results
        result_news_by_topic = list(
            db_news['news-articles'].aggregate(pipeline))

        # Calculate the sum of retweets and favorite counts for every tweets
        for item in result_news_by_topic:
            retweet_sum = sum(item['retweet_counts'])
            favorite_sum = sum(item['favorite_counts'])
            item['retweet_sum'] = retweet_sum
            item['favorite_sum'] = favorite_sum

        # Sort the tweets based on the highest accumulated retweets and favorite
        sorted_news_topic = sorted(
            result_news_by_topic, key=lambda x: x['retweet_sum'] + x['favorite_sum'], reverse=True)
        return render_template('topic-analysis.html', result=sorted_news_topic, previous_month=previous_month, current_year=current_year)
    else:
        flash(Markup(
            "Please register as member first before accessing the feature"), category='warning')
        return redirect(url_for('home'))

# Topic Sentiment page


@app.route('/sentiment', methods=["GET", "POST"])
def sentiment():
    if 'username' in session:

        # Obtain all unique topics from database
        topic_counts = db_news['news-articles'].aggregate([
            {'$match': {
                'topic': {'$ne': None}
            }
            },
            {"$group": {"_id": "$topic"}},
            {"$sort": {
                "_id": 1
            }}])
        all_topics_selection = [{"topic": t["_id"]} for t in topic_counts]

        # Pipeline to query the top 5 related news contents for each topic that are sorted based on the sentiment polarity score
        pipeline = [
            {'$match': {
                'topic': {'$ne': None}
            }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$topic",
                        "sentiment": "$sentiment",
                        "newsTitle": "$newsTitle",
                        "newsSource": "$newsSource",
                        "newsType": "$newsType",
                        "newsTimeDatePublished": "$newsTimeDatePublished",
                        "newsURL": "$newsURL",
                        "newsImageURL": "$newsImageURL",
                        "retweet_count": "$retweet_count",
                        "favorite_count": "$favorite_count"
                    },
                    "aggregate_compound_score": {"$avg": "$aggregate_compound_score"}
                }
            },
            {
                "$sort": {
                    "_id.topic": 1,
                    "_id.sentiment": 1,
                    "aggregate_compound_score": -1
                }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$_id.topic",
                        "sentiment": "$_id.sentiment"
                    },
                    "newsTitles": {
                        "$push": {
                            "newsTitle": "$_id.newsTitle",
                            "newsSource": "$_id.newsSource",
                            "newsType": "$_id.newsType",
                            "newsTimeDatePublished": "$_id.newsTimeDatePublished",
                            "aggregate_compound_score": "$aggregate_compound_score",
                            "newsURL": "$_id.newsURL",
                            "newsImageURL": "$_id.newsImageURL",
                            "retweet_count": "$_id.retweet_count",
                            "favorite_count": "$_id.favorite_count"
                        }
                    },
                    'sum_count': {"$sum": 1}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id.topic",
                    "sentiment": "$_id.sentiment",
                    "total_sentiment_count": "$sum_count",
                    "newsTitles": {
                        "$cond": [
                            {"$eq": ["$_id.sentiment", "Positive"]},
                            {"$slice": ["$newsTitles", 5]},
                            {"$slice": ["$newsTitles", -5]}
                        ]
                    }
                }
            },
            {
                "$sort": {
                    "topic": 1,
                    "sentiment": -1,
                }
            },
        ]
        topics = list(db_news['news-articles'].aggregate(pipeline))

        # Pipeline to query news sources and their respective frequency for each topic
        pipeline_pie = [
            {'$match': {
                'topic': {'$ne': None}
            }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$topic",
                        "sentiment": "$sentiment",
                        "newsSource": "$newsSource"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$_id.topic",
                        "sentiment": "$_id.sentiment"
                    },
                    "sources": {
                        "$push": {
                            "newsSource": "$_id.newsSource",
                            "count": "$count"
                        }
                    }
                }
            },

            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id.topic",
                    "sentiment": "$_id.sentiment",
                    "sources": "$sources"
                }
            },

            {
                "$sort": {
                    "topic": 1,
                    "sentiment": -1
                }
            }
        ]
        topic_pie = list(db_news['news-articles'].aggregate(pipeline_pie))

        # Convert both results from pipeline to dataframe
        df1 = pd.DataFrame(topic_pie)
        df2 = pd.DataFrame(topics)

        # concatenate the two dataframes
        df = df1.merge(df2, on=['topic', 'sentiment'], how='inner')

        # convert the dataframe to json
        topics = df.to_dict(orient="records")

        # Aggregate the results from both pipeline and convert to nested dictionaries for easier querying
        combined_data = []
        for d in topics:
            topic = d['topic']
            sentiment = d['sentiment']
            sources = d['sources']
            news_titles = d['newsTitles']
            total_count_sentiment = d['total_sentiment_count']

            topic_exists = False
            for item in combined_data:
                if item['topic'] == topic:
                    topic_exists = True
                    if sentiment not in item['sentiments']:
                        item['sentiments'][sentiment] = {
                            'total_count_sentiment': 0, 'sources': [], 'newsTitles': []}
                    item['sentiments'][sentiment]['total_count_sentiment'] += total_count_sentiment
                    item['sentiments'][sentiment]['sources'].extend(sources)
                    item['sentiments'][sentiment]['newsTitles'].extend(
                        news_titles)
                    break

            if not topic_exists:
                combined_data.append({
                    'topic': topic,
                    'sentiments': {
                        sentiment: {
                            'total_count_sentiment': total_count_sentiment,
                            'sources': sources,
                            'newsTitles': news_titles
                        }
                    }
                })

        per_page = 1  # Assign 1 topic in a single page
        page = int(request.args.get('page', 1))

        # Retrieve total number of items in collection
        total = len(combined_data)

        # Calculate the start and end index for the current page
        start_index = (page - 1) * per_page
        end_index = start_index + per_page

        # Retrieve items for the current page using list slicing
        items = combined_data[start_index:end_index]

        # Create pagination object using Flask-Paginate
        pagination = Pagination(page=page, per_page=per_page,
                                total=total, css_framework='bootstrap4')

        # If the user click the filter_topic button, display the selected topic and date ranges individually
        if request.args.get('filter_topic'):
            if request.args.get('select_topic') and request.args.get('start-date') and request.args.get('end-date'):
                selection_topic = request.args.get(
                    'select_topic')  # Obtain selected topic
                start_date = request.args.get(
                    'start-date')  # Obtain the start date
                end_date = request.args.get('end-date')  # Obtain the end date
                start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
                end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

                # Pipeline to query the top 5 related news contents for each topic based on the selected topic and date ranges
                pipeline = [
                    {'$match': {
                        "topic": selection_topic, "newsTimeDatePublished": {"$gte": start_datetime, "$lte": end_datetime}
                    }
                    },
                    {
                        "$group": {
                            "_id": {
                                "topic": "$topic",
                                "sentiment": "$sentiment",
                                "newsTitle": "$newsTitle",
                                "newsSource": "$newsSource",
                                "newsType": "$newsType",
                                "newsTimeDatePublished": "$newsTimeDatePublished",
                                "newsURL": "$newsURL",
                                "newsImageURL": "$newsImageURL",
                                "retweet_count": "$retweet_count",
                                "favorite_count": "$favorite_count"
                            },
                            "aggregate_compound_score": {"$avg": "$aggregate_compound_score"}
                        }
                    },
                    {
                        "$sort": {
                            "_id.topic": 1,
                            "_id.sentiment": 1,
                            "aggregate_compound_score": -1
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                "topic": "$_id.topic",
                                "sentiment": "$_id.sentiment"
                            },
                            "newsTitles": {
                                "$push": {
                                    "newsTitle": "$_id.newsTitle",
                                    "newsSource": "$_id.newsSource",
                                    "newsType": "$_id.newsType",
                                    "newsTimeDatePublished": "$_id.newsTimeDatePublished",
                                    "aggregate_compound_score": "$aggregate_compound_score",
                                    "newsURL": "$_id.newsURL",
                                    "newsImageURL": "$_id.newsImageURL",
                                    "retweet_count": "$_id.retweet_count",
                                    "favorite_count": "$_id.favorite_count"
                                }
                            },
                            'sum_count': {"$sum": 1}
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "topic": "$_id.topic",
                            "sentiment": "$_id.sentiment",
                            "total_sentiment_count": "$sum_count",
                            "newsTitles": {
                                "$cond": [
                                    {"$eq": ["$_id.sentiment", "Positive"]},
                                    {"$slice": ["$newsTitles", 5]},
                                    {"$slice": ["$newsTitles", -5]}
                                ]
                            }
                        }
                    },
                    {
                        "$sort": {
                            "topic": 1,
                            "sentiment": -1,
                        }
                    },
                ]
                topics_single = list(
                    db_news['news-articles'].aggregate(pipeline))

                # Pipeline to query news sources and their respective frequency for each topic based on the selected topic and date ranges
                pipeline_pie = [
                    {'$match': {
                        "topic": selection_topic, "newsTimeDatePublished": {"$gte": start_datetime, "$lte": end_datetime}
                    }
                    },
                    {
                        "$group": {
                            "_id": {
                                "topic": "$topic",
                                "sentiment": "$sentiment",
                                "newsSource": "$newsSource"
                            },
                            "count": {"$sum": 1}
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                "topic": "$_id.topic",
                                "sentiment": "$_id.sentiment"
                            },
                            "sources": {
                                "$push": {
                                    "newsSource": "$_id.newsSource",
                                    "count": "$count"
                                }
                            }
                        }
                    },

                    {
                        "$project": {
                            "_id": 0,
                            "topic": "$_id.topic",
                            "sentiment": "$_id.sentiment",
                            "sources": "$sources"
                        }
                    },

                    {
                        "$sort": {
                            "topic": 1,
                            "sentiment": -1
                        }
                    }
                ]
                topic_pie_single = list(
                    db_news['news-articles'].aggregate(pipeline_pie))

                # Topic filtering validation
                if not topics_single or not topic_pie_single:
                    flash(
                        Markup("No any data found for the selected filter."), category='warning')
                    return redirect(url_for('sentiment'))
                else:
                    # Convert both results from pipeline to dataframe
                    df1_single = pd.DataFrame(topic_pie_single)
                    df2_single = pd.DataFrame(topics_single)

                    # concatenate the two dataframes
                    df_single = df1_single.merge(
                        df2_single, on=['topic', 'sentiment'], how='inner')

                    # convert the dataframe to json
                    topics_single = df_single.to_dict(orient="records")

                    # Aggregate the results from both pipeline and convert to nested dictionaries for easier querying
                    combined_data_single = []
                    for d in topics_single:
                        topic = d['topic']
                        sentiment = d['sentiment']
                        sources = d['sources']
                        news_titles = d['newsTitles']
                        total_count_sentiment = d['total_sentiment_count']

                        topic_exists = False
                        for item in combined_data_single:
                            if item['topic'] == topic:
                                topic_exists = True
                                if sentiment not in item['sentiments']:
                                    item['sentiments'][sentiment] = {
                                        'total_count_sentiment': 0, 'sources': [], 'newsTitles': []}
                                item['sentiments'][sentiment]['total_count_sentiment'] += total_count_sentiment
                                item['sentiments'][sentiment]['sources'].extend(
                                    sources)
                                item['sentiments'][sentiment]['newsTitles'].extend(
                                    news_titles)
                                break

                        if not topic_exists:
                            combined_data_single.append({
                                'topic': topic,
                                'sentiments': {
                                    sentiment: {
                                        'total_count_sentiment': total_count_sentiment,
                                        'sources': sources,
                                        'newsTitles': news_titles
                                    }
                                }
                            })

                    return render_template('multiview-sentiment.html', recommend=combined_data_single, dropdown_result=all_topics_selection, selection_topic=selection_topic, start_date=start_date, end_date=end_date)

            # Topic selection validation
            elif not request.args.get('select_topic'):
                flash(Markup("No topic is selected."), category='warning')
                return redirect(url_for('sentiment'))

            # Start date selection validation
            elif not request.args.get('start-date'):
                flash(Markup("No start date is selected."), category='warning')
                return redirect(url_for('sentiment'))

            # End date selection validation
            elif not request.args.get('end-date'):
                flash(Markup("No end date is selected."), category='warning')
                return redirect(url_for('sentiment'))

        return render_template('multiview-sentiment.html', recommend=items, pagination=pagination, dropdown_result=all_topics_selection)

    else:
        flash(Markup(
            "Please register as member first before accessing the feature"), category='warning')
        return redirect(url_for('home'))

# Topic Sentiment Comparison page


@app.route('/sentiment_compare', methods=["GET", "POST"])
def sentiment_comparison():
    if 'username' in session:

        # Obtain all unique topics from database
        topic_counts = db_news['news-articles'].aggregate([
            {'$match': {
                'topic': {'$ne': None}
            }
            },
            {"$group": {"_id": "$topic"}},
            {"$sort": {
                "_id": 1
            }}])
        all_topics_selection = [{"topic": t["_id"]} for t in topic_counts]

        # Pipeline to extract unique dates without time
        pipeline_extract_time = [
            {
                '$project': {
                    'date': {
                        '$dateToString': {
                            'format': '%Y-%m-%d',
                            'date': '$newsTimeDatePublished'
                        }
                    }
                }
            },
            {
                '$group': {
                    '_id': '$date'
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'date': '$_id'
                }
            },
            {
                '$sort': {
                    'date': 1
                }
            }
        ]

        extract_unique_date = list(
            db_news['news-articles'].aggregate(pipeline_extract_time))

        # Pipeline to query all information for Finance and Health topics (Bar Chart)
        pipeline = [
            {'$match': {
                'topic': {'$ne': None, "$in": ['Finance', 'Health']}
            }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$topic",
                        "sentiment": "$sentiment",
                        "newsTitle": "$newsTitle",
                        "newsSource": "$newsSource",
                        "newsType": "$newsType",
                        "newsTimeDatePublished": "$newsTimeDatePublished",
                        "newsURL": "$newsURL",
                        "newsImageURL": "$newsImageURL",
                        "retweet_count": "$retweet_count",
                        "favorite_count": "$favorite_count"

                    },
                    "aggregate_compound_score": {"$avg": "$aggregate_compound_score"}
                }
            },
            {
                "$sort": {
                    "_id.topic": 1,
                    "_id.sentiment": 1,
                    "aggregate_compound_score": -1
                }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$_id.topic",
                        "sentiment": "$_id.sentiment"
                    },
                    "newsTitles": {
                        "$push": {
                            "newsTitle": "$_id.newsTitle",
                            "newsSource": "$_id.newsSource",
                            "newsType": "$_id.newsType",
                                        "newsTimeDatePublished": "$_id.newsTimeDatePublished",
                                        "aggregate_compound_score": "$aggregate_compound_score",
                                        "newsURL": "$_id.newsURL",
                                        "newsImageURL": "$_id.newsImageURL",
                                        "retweet_count": "$_id.retweet_count",
                                        "favorite_count": "$_id.favorite_count"
                        }
                    },
                    'sum_count': {"$sum": 1}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id.topic",
                    "sentiment": "$_id.sentiment",
                    "total_sentiment_count": "$sum_count",
                    "newsTitles": {
                        "$cond": [
                            {"$eq": ["$_id.sentiment", "Positive"]},
                            {"$slice": ["$newsTitles", 5]},
                            {"$slice": ["$newsTitles", -5]}
                        ]
                    }
                }
            },
            {
                "$sort": {
                    "topic": 1,
                    "sentiment": -1,
                }
            },
        ]

        # Pipeline to query the aggregate compound score for each day for Finance and Health topics (Line Chart)
        pipeline_time = [
            {
                '$match': {
                    'topic': {'$ne': None, '$in': ['Finance', 'Health']}
                }
            },
            {
                '$group': {
                    '_id': {
                        'topic': '$topic',
                        'date': {'$dateToString': {'format': "%Y-%m-%d", 'date': "$newsTimeDatePublished"}},
                        'sentiment': '$sentiment'
                    },
                    'avgScore': {'$avg': "$aggregate_compound_score"},
                    'count': {'$sum': 1}
                }
            },
            {
                '$sort': {
                    '_id.date': 1,
                    '_id.sentiment': 1
                }
            },

            {
                "$group": {
                    '_id': {
                        'topic': '$_id.topic',
                        'sentiment': '$_id.sentiment'
                    },
                    'data': {
                        '$push': {
                            'Date': '$_id.date',
                            'AvgCompoundScore': '$avgScore',
                            'Count': '$count'
                        }
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id.topic",
                    "sentiment": "$_id.sentiment",
                    "data": "$data"

                }
            },
            {
                '$sort': {
                    'topic': 1,
                    'sentiment': -1,
                }
            },
        ]

        # Convert the result from above pipeline to dataframe
        topic_time_compare = list(
            db_news['news-articles'].aggregate(pipeline_time))

        # Aggregate the results from the pipeline and convert to nested dictionaries for easier querying
        combined_data_datetime = []
        for d in topic_time_compare:
            topic = d['topic']
            sentiment = d['sentiment']
            data = d['data']

            topic_exists = False
            for item in combined_data_datetime:
                if item['topic'] == topic:
                    topic_exists = True
                    if sentiment not in item['sentiments']:
                        item['sentiments'][sentiment] = {'data': []}
                    item['sentiments'][sentiment]['data'].extend(data)
                    break

            if not topic_exists:
                combined_data_datetime.append({
                    'topic': topic,
                    'sentiments': {
                        sentiment: {
                            'data': data

                        }
                    }
                })

        # Pipeline to query frequency for each news source for Finance and Health topics (Bubble Chart)
        pipeline_pie = [
            {'$match': {
                'topic': {'$ne': None, "$in": ['Finance', 'Health']}
            }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$topic",
                        "sentiment": "$sentiment",
                        "newsSource": "$newsSource"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$_id.topic",
                        "sentiment": "$_id.sentiment"
                    },
                    "sources": {
                        "$push": {
                            "newsSource": "$_id.newsSource",
                            "count": "$count"
                        }
                    }
                }
            },

            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id.topic",
                    "sentiment": "$_id.sentiment",
                    "sources": "$sources"
                }
            },

            {
                "$sort": {
                    "topic": 1,
                    "sentiment": -1
                }
            }
        ]

        # Pipeline to query and calculate aggregate compound score for each news source for Finance and Health topics (Bubble Chart)
        pipeline_score = [
            {'$match': {
                'topic': {'$ne': None, "$in": ['Finance', 'Health']}
            }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$topic",

                        "newsSource": "$newsSource"
                    },
                    'avgScore': {'$avg': '$aggregate_compound_score'},
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": {
                        "topic": "$_id.topic",

                    },
                    "sources": {
                        "$push": {
                            "newsSource": "$_id.newsSource",
                            "count": "$count",
                            'avgScore': '$avgScore'
                        }
                    }
                }
            },

            {
                "$project": {
                    "_id": 0,
                    "topic": "$_id.topic",
                    "sources": "$sources"
                }
            },

            {
                "$sort": {
                    "topic": 1
                }
            }
        ]

        # Convert all results from pipelines to dataframe
        topics_compare = list(db_news['news-articles'].aggregate(pipeline))
        topic_pie_compare = list(
            db_news['news-articles'].aggregate(pipeline_pie))
        topic_score = list(db_news['news-articles'].aggregate(pipeline_score))

        # Convert the results from pipeline to dataframe
        df1 = pd.DataFrame(topics_compare)
        df2 = pd.DataFrame(topic_pie_compare)

        # concatenate the two dataframes
        df = df1.merge(df2, on=['topic', 'sentiment'], how='inner')
        # convert the dataframe to json
        results_topic_comparison = df.to_dict(orient="records")

        # Aggregate the results from both pipeline and convert to nested dictionaries for easier querying
        combined_data_compare = []
        for d in results_topic_comparison:
            topic = d['topic']
            sentiment = d['sentiment']
            sources = d['sources']
            news_titles = d['newsTitles']
            total_count_sentiment = d['total_sentiment_count']

            topic_exists = False
            for item in combined_data_compare:
                if item['topic'] == topic:
                    topic_exists = True
                    if sentiment not in item['sentiments']:
                        item['sentiments'][sentiment] = {
                            'total_count_sentiment': 0, 'sources': [], 'newsTitles': []}
                    item['sentiments'][sentiment]['total_count_sentiment'] += total_count_sentiment
                    item['sentiments'][sentiment]['sources'].extend(sources)
                    item['sentiments'][sentiment]['newsTitles'].extend(
                        news_titles)
                    break

            if not topic_exists:
                combined_data_compare.append({
                    'topic': topic,
                    'sentiments': {
                        sentiment: {
                            'total_count_sentiment': total_count_sentiment,
                            'sources': sources,
                            'newsTitles': news_titles
                        }
                    }
                })

        # If the user click the filter_topic button, display the selected topics and date ranges individually
        if request.args.get('compare_topic'):
            if request.args.get('topic_compare') and request.args.get('topic_compare2') and request.args.get('start_date_compare') and request.args.get('end_date_compare'):
                first_selection_topic = request.args.get(
                    'topic_compare')  # Obtain the first selected topic
                second_selection_topic = request.args.get(
                    'topic_compare2')  # Obtain the second selected topic
                start_date = request.args.get(
                    'start_date_compare')  # Obtain the start date
                end_date = request.args.get(
                    'end_date_compare')  # Obtain the end date
                start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
                end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

                if first_selection_topic != second_selection_topic:

                    # Pipeline to query all information for the selected topics and date ranges (Bar Chart)
                    pipeline = [
                        {'$match': {
                            'topic': {'$ne': None, "$in": [first_selection_topic, second_selection_topic]}, "newsTimeDatePublished": {"$gte": start_datetime, "$lte": end_datetime}
                        }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "topic": "$topic",
                                    "sentiment": "$sentiment",
                                    "newsTitle": "$newsTitle",
                                    "newsSource": "$newsSource",
                                    "newsType": "$newsType",
                                    "newsTimeDatePublished": "$newsTimeDatePublished",
                                    "newsURL": "$newsURL",
                                    "newsImageURL": "$newsImageURL",
                                    "retweet_count": "$retweet_count",
                                    "favorite_count": "$favorite_count"
                                },
                                "aggregate_compound_score": {"$avg": "$aggregate_compound_score"}
                            }
                        },
                        {
                            "$sort": {
                                "_id.topic": 1,
                                "_id.sentiment": 1,
                                "aggregate_compound_score": -1
                            }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "topic": "$_id.topic",
                                    "sentiment": "$_id.sentiment"
                                },
                                "newsTitles": {
                                    "$push": {
                                        "newsTitle": "$_id.newsTitle",
                                        "newsSource": "$_id.newsSource",
                                        "newsType": "$_id.newsType",
                                                    "newsTimeDatePublished": "$_id.newsTimeDatePublished",
                                                    "aggregate_compound_score": "$aggregate_compound_score",
                                                    "newsURL": "$_id.newsURL",
                                                    "newsImageURL": "$_id.newsImageURL",
                                                    "retweet_count": "$_id.retweet_count",
                                                    "favorite_count": "$_id.favorite_count"
                                    }
                                },
                                'sum_count': {"$sum": 1}
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "topic": "$_id.topic",
                                "sentiment": "$_id.sentiment",
                                "total_sentiment_count": "$sum_count",
                                "newsTitles": {
                                    "$cond": [
                                        {"$eq": [
                                            "$_id.sentiment", "Positive"]},
                                        {"$slice": [
                                            "$newsTitles", 5]},
                                        {"$slice": [
                                            "$newsTitles", -5]}
                                    ]
                                }
                            }
                        },
                        {
                            "$sort": {
                                "topic": 1,
                                "sentiment": -1,
                            }
                        },
                    ]

                    # Pipeline to query the aggregate compound score for each day for selected topics and date ranges (Line Chart)
                    pipeline_time = [
                        {
                            '$match': {
                                'topic': {'$ne': None, "$in": [first_selection_topic, second_selection_topic]}, "newsTimeDatePublished": {"$gte": start_datetime, "$lte": end_datetime}
                            }
                        },
                        {
                            '$group': {
                                '_id': {
                                    'topic': '$topic',
                                    'date': {'$dateToString': {'format': "%Y-%m-%d", 'date': "$newsTimeDatePublished"}},
                                    'sentiment': '$sentiment'
                                },
                                'avgScore': {'$avg': "$aggregate_compound_score"},
                                'count': {'$sum': 1}
                            }
                        },
                        {
                            '$sort': {
                                '_id.date': 1,
                                '_id.sentiment': 1
                            }
                        },

                        {
                            "$group": {
                                '_id': {
                                    'topic': '$_id.topic',
                                    'sentiment': '$_id.sentiment'
                                },
                                'data': {
                                    '$push': {
                                        'Date': '$_id.date',
                                        'AvgCompoundScore': '$avgScore',
                                        'Count': '$count'
                                    }
                                }
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "topic": "$_id.topic",
                                "sentiment": "$_id.sentiment",
                                "data": "$data"

                            }
                        },
                        {
                            '$sort': {
                                'topic': 1,
                                'sentiment': -1,
                            }
                        },
                    ]

                    # Convert the result from above pipeline to dataframe
                    topic_time_compare = list(
                        db_news['news-articles'].aggregate(pipeline_time))

                    # Aggregate the results from the pipeline and convert to nested dictionaries for easier querying
                    combined_data_datetime = []
                    for d in topic_time_compare:
                        topic = d['topic']
                        sentiment = d['sentiment']
                        data = d['data']

                        topic_exists = False
                        for item in combined_data_datetime:
                            if item['topic'] == topic:
                                topic_exists = True
                                if sentiment not in item['sentiments']:
                                    item['sentiments'][sentiment] = {
                                        'data': []}
                                item['sentiments'][sentiment]['data'].extend(
                                    data)
                                break

                        if not topic_exists:
                            combined_data_datetime.append({
                                'topic': topic,
                                'sentiments': {
                                    sentiment: {
                                        'data': data

                                    }
                                }
                            })

                    # Pipeline to query frequency for each news source for selected topics and date ranges (Bubble Chart)
                    pipeline_pie = [
                        {'$match': {
                            'topic': {'$ne': None, "$in": [first_selection_topic, second_selection_topic]}, "newsTimeDatePublished": {"$gte": start_datetime, "$lte": end_datetime}
                        }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "topic": "$topic",
                                    "sentiment": "$sentiment",
                                    "newsSource": "$newsSource"
                                },
                                "count": {"$sum": 1}
                            }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "topic": "$_id.topic",
                                    "sentiment": "$_id.sentiment"
                                },
                                "sources": {
                                    "$push": {
                                        "newsSource": "$_id.newsSource",
                                        "count": "$count"
                                    }
                                }
                            }
                        },

                        {
                            "$project": {
                                "_id": 0,
                                "topic": "$_id.topic",
                                "sentiment": "$_id.sentiment",
                                "sources": "$sources"
                            }
                        },

                        {
                            "$sort": {
                                "topic": 1,
                                "sentiment": -1
                            }
                        }
                    ]

                    # Pipeline to query and calculate aggregate compound score for selected topics and date ranges (Bubble Chart)
                    pipeline_score = [
                        {'$match': {
                            'topic': {'$ne': None, "$in": [first_selection_topic, second_selection_topic]}, "newsTimeDatePublished": {"$gte": start_datetime, "$lte": end_datetime}
                        }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "topic": "$topic",

                                    "newsSource": "$newsSource"
                                },
                                'avgScore': {'$avg': '$aggregate_compound_score'},
                                "count": {"$sum": 1}
                            }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "topic": "$_id.topic",

                                },
                                "sources": {
                                    "$push": {
                                        "newsSource": "$_id.newsSource",
                                        "count": "$count",
                                        'avgScore': '$avgScore'
                                    }
                                }
                            }
                        },

                        {
                            "$project": {
                                "_id": 0,
                                "topic": "$_id.topic",
                                "sources": "$sources"
                            }
                        },

                        {
                            "$sort": {
                                "topic": 1
                            }
                        }
                    ]

                    # Convert all results from pipelines to dataframe
                    topics_compare = list(
                        db_news['news-articles'].aggregate(pipeline))
                    topic_pie_compare = list(
                        db_news['news-articles'].aggregate(pipeline_pie))
                    topic_score = list(
                        db_news['news-articles'].aggregate(pipeline_score))

                    # Check whether the data exists for selected topic and date ranges
                    if not topics_compare or not topic_pie_compare:
                        flash(
                            Markup("No any data found for the selected filter."), category='warning')
                        return redirect(url_for('sentiment_comparison'))
                    else:

                        # Convert the results from pipeline to dataframe
                        df1 = pd.DataFrame(topics_compare)
                        df2 = pd.DataFrame(topic_pie_compare)

                        # concatenate the two dataframes
                        df = df1.merge(
                            df2, on=['topic', 'sentiment'], how='inner')

                        # convert the dataframe to json
                        results_topic_comparison = df.to_dict(orient="records")

                        # Aggregate the results from both pipeline and convert to nested dictionaries for easier querying
                        combined_data_compare = []
                        for d in results_topic_comparison:
                            topic = d['topic']
                            sentiment = d['sentiment']
                            sources = d['sources']
                            news_titles = d['newsTitles']
                            total_count_sentiment = d['total_sentiment_count']

                            topic_exists = False
                            for item in combined_data_compare:
                                if item['topic'] == topic:
                                    topic_exists = True
                                    if sentiment not in item['sentiments']:
                                        item['sentiments'][sentiment] = {
                                            'total_count_sentiment': 0, 'sources': [], 'newsTitles': []}
                                    item['sentiments'][sentiment]['total_count_sentiment'] += total_count_sentiment
                                    item['sentiments'][sentiment]['sources'].extend(
                                        sources)
                                    item['sentiments'][sentiment]['newsTitles'].extend(
                                        news_titles)
                                    break

                            if not topic_exists:
                                combined_data_compare.append({
                                    'topic': topic,
                                    'sentiments': {
                                        sentiment: {
                                            'total_count_sentiment': total_count_sentiment,
                                            'sources': sources,
                                            'newsTitles': news_titles
                                        }
                                    }
                                })

                        # Filter topic and date ranges validation where there must be 2 topics exist
                        if len(combined_data_compare) < 2:
                            flash(
                                Markup("No data found for the selected filter."), category='warning')
                            return redirect(url_for('sentiment_comparison'))
                        else:
                            return render_template('multiview-comparison.html', results_topic_compare=combined_data_compare, topic_time_compare=combined_data_datetime, topic_score=topic_score, dropdown_result=all_topics_selection, first_selection_topic=first_selection_topic, second_selection_topic=second_selection_topic, start_date=start_date, end_date=end_date, extract_unique_date=extract_unique_date)

                else:
                    # Topic validation to not allow for duplicate topics selection
                    flash(Markup("Similar topics are selected."),
                          category='warning')
                    return redirect(url_for('sentiment_comparison'))

            # Topic validation to prompt user to select first topic
            elif not request.args.get('topic_compare'):
                flash(Markup("First Topic is not selected."), category='warning')
                return redirect(url_for('sentiment_comparison'))

            # Topic validation to prompt user to select second topic
            elif not request.args.get('topic_compare2'):
                flash(Markup("Second Topic is not selected."), category='warning')
                return redirect(url_for('sentiment_comparison'))

            # Date validation to prompt user to select start date
            elif not request.args.get('start_date_compare'):
                flash(Markup("No start date is selected."), category='warning')
                return redirect(url_for('sentiment_comparison'))

            # Date validation to prompt user to select end date
            elif not request.args.get('end_date_compare'):
                flash(Markup("No end date is selected."), category='warning')
                return redirect(url_for('sentiment_comparison'))

        return render_template('multiview-comparison.html', results_topic_compare=combined_data_compare, topic_time_compare=combined_data_datetime, topic_score=topic_score, extract_unique_date=extract_unique_date, dropdown_result=all_topics_selection)

    else:
        flash(Markup(
            "Please register as member first before accessing the feature"), category='warning')
        return redirect(url_for('home'))

# User profile page


@app.route('/user_profile', methods=['POST', 'GET'])
def profile():
    if 'username' in session:

        # Allow user to update their username
        if request.method == 'POST':
            if 'save' in request.form:
                body = request.form
                username = body['username']
                query = {"email": session['email']}
                new_values = {"$set": {"username": username}}
                db_users_collection.update_one(query, new_values)
                session['username'] = username
                flash("Username has been successfully updated", category='success')
                return redirect(url_for('profile'))
        else:
            return render_template('user-profile.html')

# Logout page


@app.route('/logout')
def logout():

    # Remove every information about the user after logout
    session.pop('username', None)
    session.pop('confirmation', None)
    session.pop('email', None)
    session.pop('email_captured', None)
    return redirect(url_for('authenticate'))


# Intialize flask application and allow debugging
if __name__ == "__main__":
    app.run(debug=True)
