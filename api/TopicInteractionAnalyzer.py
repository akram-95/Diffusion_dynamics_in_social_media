import json
import textwrap
from itertools import combinations

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim import corpora, models
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from pandas import Timestamp

from pyvis.network import Network
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from typing import List
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from api.utils import parse_json, identify_similarity_sentence_topics, get_result, calculate_cosine_sim, equal, \
    choose_colors
from datetime import datetime, timedelta

nltk.download('punkt')
G = nx.DiGraph()
colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray"]
nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", filter_menu=True)
nltk.download('stopwords')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
hdbscan_model = HDBSCAN(min_cluster_size=80, min_samples=40,
                        gen_min_span_tree=True,
                        prediction_data=True)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=list(stopwords.words('english')))

bertopicModel = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    language='english',
    calculate_probabilities=True,
    verbose=True,
)
stopwords1 = list(stopwords.words('english')) + ['http', 'https', 'amp', 'com']

# we add this to remove stopwords that can pollute topcs
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords1)

tweets_parsed = parse_json('twitter-politics-tweets.jsonl')[0:100000]

replies_parsed = parse_json('twitter-politics-tweets_reply.jsonl')
quotes_parsed = parse_json('twitter-politics-tweets_quote.jsonl')

# Convert lists of dictionaries into DataFrames
tweets_df = pd.DataFrame(tweets_parsed)
tweets_df['id'] = tweets_df['id'].astype('object')
replies_df = pd.DataFrame(replies_parsed)
prefix = 'reply_'
"""
replies_df = replies_df.rename(
    columns={old_key: prefix + old_key if old_key != 'in_reply_to_tweet_id' else old_key for old_key in
             replies_df.columns})
"""
replies_df['id'] = replies_df['id'].astype('object')
replies_df['in_reply_to_tweet_id'] = replies_df['in_reply_to_tweet_id'].astype('object')
quotes_df = pd.DataFrame(quotes_parsed)
prefix = 'quote_'
"""
quotes_df = quotes_df.rename(
    columns={old_key: prefix + old_key if old_key != 'quoted_tweet_id' else old_key for old_key in
             quotes_df.columns})
"""
quotes_df['id'] = quotes_df['id'].astype('object')
quotes_df['quoted_tweet_id'] = quotes_df['quoted_tweet_id'].astype('object')

# Group replies by tweet ID and aggregate into a list of dictionaries
grouped_replies = replies_df.groupby('in_reply_to_tweet_id').apply(lambda x: x.to_dict(orient='records')).reset_index()

# Rename the resulting column
grouped_replies = grouped_replies.rename(columns={0: 'replies'})

# Group replies by tweet ID and aggregate into a list of dictionaries
grouped_quotes = quotes_df.groupby('quoted_tweet_id').apply(lambda x: x.to_dict(orient='records')).reset_index()

# Rename the resulting column
grouped_quotes = grouped_quotes.rename(columns={0: 'quotes'})

# Merge replies and quotes with tweets based on tweet ID
merged_df = pd.merge(tweets_df, grouped_replies, left_on='id', right_on='in_reply_to_tweet_id', how='left')

result_df = pd.merge(merged_df, grouped_quotes, left_on='id', right_on='quoted_tweet_id',
                     how='left')


# Define a function to replace NaN values with an empty list
def fillna_with_empty_list(value):
    if isinstance(value, list):
        return value
    elif pd.isna(value):
        return []
    else:
        return [value]


result_df['quotes'] = result_df['quotes'].apply(fillna_with_empty_list)
result_df['replies'] = result_df['replies'].apply(fillna_with_empty_list)


# Display the resulting DataFrame

class TopicInteractionAnalyzer(object):
    def find_topics_by_tweets(self, inputDataFrame: pd.DataFrame):
        df = pd.DataFrame(columns=['topic', 'replies_quotes_topics'])
        topics, probs = bertopicModel.fit_transform(inputDataFrame['translation2'])
        for index, topic_id in enumerate(topics):
            topic_info = bertopicModel.get_topic_info(topic_id)
            replies = [sub['translation2'] for sub in inputDataFrame.iloc[index]['replies']]
            quotes = [sub['translation2'] for sub in inputDataFrame.iloc[index]['quotes']]
            replies_quotes = replies + quotes
            if len(replies_quotes) > 10:
                replies_quotes_topics = []
                try:
                    replies_quotes_topics = bertopicModel.transform(replies_quotes)[0]
                except Exception as e:
                    print(f"An error occurred during topic modeling: {e}")
                replies_quotes_topics_dict = []
                for replies_quotes_topic in set(replies_quotes_topics):
                    count = replies_quotes_topics.count(replies_quotes_topic)
                    replies_quotes_topics_dict.append({'count': count,
                                                       'replies_quotes_topic': replies_quotes_topic})
                df.loc[len(df)] = {'topic': topic_id, 'replies_quotes_topics': replies_quotes_topics_dict}
        self.normalize_topics_relation_weights(df)
        return df

    def normalize_topics_relation_weights(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            if len(row['replies_quotes_topics']) > 0:
                max_num = max([i['count'] for i in row['replies_quotes_topics']])
                sum_num = sum([i['count'] for i in row['replies_quotes_topics']])
                min_num = min([i['count'] for i in row['replies_quotes_topics']])
                if len(row['replies_quotes_topics']) == 1:
                    row['replies_quotes_topics'][0]['count'] = 1
                else:
                    for index_count, count in enumerate(row['replies_quotes_topics']):
                        normalized_number = (count['count'] / sum_num)
                        row['replies_quotes_topics'][index_count]['count'] = normalized_number
                        """
                        if min_num != max_num:
                            # normalized_number = (count['count'] - min_num) / (max_num - min_num)
                            normalized_number = (count['count'] / sum_num)
                            row['replies_quotes_topics'][index_count]['count'] = normalized_number
                        else:
                            row['replies_quotes_topics'][index_count]['count'] = 0
                        """
            df.iloc[index] = row

    def find_topics_interaction_evolution_overtime(self):
        result_df['date'] = pd.to_datetime(result_df['date'])
        # Segment data into time intervals (e.g., monthly)
        time_intervals = pd.date_range(start=result_df['date'].min(), end=result_df['date'].max(), freq='M')
        df_overtime = pd.DataFrame(columns=['date', 'result'])
        for time_interval in time_intervals:
            # Filter data for the current time interval
            data_interval = result_df[
                (result_df['date'] >= time_interval) & (result_df['date'] < time_interval + pd.DateOffset(months=1))]
            result = self.find_topics_by_tweets(data_interval)
            df_overtime.loc[len(df_overtime)] = {'date': time_interval, 'result': result.to_dict('records')}
        df_overtime['date'] = df_overtime['date'].astype(str)
        with open('result_relations.json', 'w') as json_file:
            json.dump(df_overtime.to_dict('records'), json_file, default=str)
        self.visualize_evolution(df_overtime)

    def visualize_evolution(self, edge_weights_over_time: pd.DataFrame):
        # Plotting

        for index, row in edge_weights_over_time.iterrows():
            scatter_plots = []
            dates = []
            main_topics_id = []
            weights = []
            derived_topics_id = []
            time_periods = row['date']
            dates.append(time_periods)
            result = row['result']
            seen = set()
            plt.figure(num=str(time_periods), figsize=(10, 6))
            for j, item in enumerate(result):
                main_topic_id = item['topic']
                derived_topics = item['replies_quotes_topics']
                for k, derived_topic in enumerate(derived_topics):
                    """and (main_topic_id, derived_topic['replies_quotes_topic'], time_periods) not in seen"""
                    if main_topic_id != derived_topic['replies_quotes_topic'] and (
                            main_topic_id, derived_topic['replies_quotes_topic']) not in seen:
                        """plt.plot(main_topic_id, derived_topic['replies_quotes_topic'],
                                  label=f"Main topic_id: {main_topic_id} to Derived topic_id: "
                                        f"{derived_topic['replies_quotes_topic']} with weight: {derived_topic['count']}",
                                  marker='o')
                         """
                        scatter = plt.scatter(main_topic_id, derived_topic['replies_quotes_topic'],
                                              s=derived_topic['count'] * 100, c='b',
                                              alpha=0.5)
                        main_topics_id.append(main_topic_id)
                        seen.add((
                            main_topic_id, derived_topic['replies_quotes_topic']))
                        derived_topics_id.append(derived_topic['replies_quotes_topic'])
                        scatter_plots.append(scatter)
                        weights.append(derived_topic['count'])
                        """"
                        plt.annotate(
                            f"main topic id: {main_topic_id} to derived topic id: {derived_topic['replies_quotes_topic']}",
                            (time_periods, derived_topic['count']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')"""
                        plt.text(main_topic_id, derived_topic['replies_quotes_topic'],
                                 f'Weight: {round(derived_topic["count"], 1)}', fontsize=8, ha='center',
                                 va='bottom')

            plt.gca().xaxis.set_major_locator(MultipleLocator())
            plt.gca().yaxis.set_major_locator(MultipleLocator())

            plt.title(f"Relation between Main topics and derived topics from Replies/Quotes for date {time_periods} ")
            plt.xlabel("Main Topic")
            plt.ylabel("Derived Topic from from Replies/Quotes")
            # plt.yticks(list(set(weights)))

            # Show plot
            plt.grid(True)
            plt.show()
