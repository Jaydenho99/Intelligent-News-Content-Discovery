
# To update the duration published (time ago) for every news contents

# Install libraries
get_ipython().system('pip install timeago')
get_ipython().system('pip install pymongo')

# Import libraries
from datetime import datetime,timedelta,date
from pymongo import MongoClient,collection,UpdateOne
import timeago

# Credentials access for mongodb database 
client = MongoClient("mongodb+srv://jayden:Leonho31@atlascluster.hrjoukd.mongodb.net/?retryWrites=true&w=majority")
db_news=client['news-content']

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





