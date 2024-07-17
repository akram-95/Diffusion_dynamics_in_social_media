import textwrap
from itertools import combinations

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim import corpora, models
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from pyvis.network import Network
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from api.utils import parse_json, identify_similarity_sentence_topics, get_result, calculate_cosine_sim, equal, \
    choose_colors

nltk.download('punkt')
G = nx.DiGraph()
colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray"]
nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", filter_menu=True)
nltk.download('stopwords')
tweets_parsed = []
# tweets_parsed = parse_json('twitter-politics-tweets.jsonl')[0:10000]
print(len(tweets_parsed))
replies_parsed = []
quotes_parsed = []
# replies_parsed = parse_json('twitter-politics-tweets_reply.jsonl')[0:5000]
# quotes_parsed = parse_json('twitter-politics-tweets_quote.jsonl')
print(len(replies_parsed))

tweet_dict = {}
for tweet in tweets_parsed:
    tweet_id = tweet['id']
    tweet_dict[tweet_id] = {
        'tweet': tweet,
        'replies': [],
        'quotes': []
    }
    if 'translation2' in tweet:
        tweet_dict[tweet['translation2']] = tweet_id

for reply in replies_parsed:
    tweet_id = reply['in_reply_to_tweet_id']
    if tweet_id in tweet_dict:
        tweet_dict[tweet_id]['replies'].append(reply)

for quote in quotes_parsed:
    tweet_id = quote['quoted_tweet_id']
    if tweet_id in tweet_dict:
        tweet_dict[tweet_id]['quotes'].append(quote)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
hdbscan_model = HDBSCAN(min_cluster_size=80, min_samples=40,
                        gen_min_span_tree=True,
                        prediction_data=True)
stopwords1 = list(stopwords.words('english')) + ['http', 'https', 'amp', 'com']

# we add this to remove stopwords that can pollute topcs
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords1)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


class ApiService(object):

    def get_tweet_id_per_tweet(self, tweet):
        for name in tweets_parsed:
            if 'translation2' in name:
                translation2 = name['translation2']
                if equal(tweet.lower(), translation2.lower()):
                    return name['id']

    def get_replies_per_tweet_id(self, tweet_id):
        result = []
        for name in replies_parsed:
            if 'in_reply_to_tweet_id' in name and name['in_reply_to_tweet_id'] == tweet_id:
                result.append(name)
        return result

    def get_quotes_per_tweet_id(self, tweet_id):
        result = []
        for name in quotes_parsed:
            if 'in_reply_to_tweet_id' in name and name['in_reply_to_tweet_id'] == tweet_id:
                result.append(name)
        return result

    def extract_topics_with_represented_docs(self):
        replies = []
        quotes = []
        for key, value in tweet_dict.items():
            if isinstance(value, dict):
                items_replies = value['replies']
                for item in items_replies:
                    if 'translation2' in item:
                        replies.append(item['translation2'])

        for key, value in tweet_dict.items():
            if isinstance(value, dict):
                items_quotes = value['quotes']
                for item in items_quotes:
                    if 'translation2' in item:
                        quotes.append(item['translation2'])

        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            language='english',
            calculate_probabilities=True,
            verbose=True,
        )
        docs = [tweet_dict[d]['tweet']['translation2'] for d in tweet_dict if
                isinstance(tweet_dict[d], dict) and 'translation2' in tweet_dict[d]['tweet']]
        topics, probs = model.fit_transform(docs)
        self.find_topics_coherence(model, docs, topics)
        topics_replies = model.transform(replies)
        topics_quotes = model.transform(quotes)
        print("length of replies: ", len(topics_replies[0]))

        topic_info = model.get_topic_info()

        result = []
        result_for_correlation_analysis = {}

        for x in topic_info.values:
            if x[0] >= 0:
                represented_docs = x[4]
                topic_name = x[2]
                topics_occurences = {}
                replies_correlation = []
                quotes_correlation = []
                for doc in represented_docs:
                    tweet_id = tweet_dict[doc]

                    tweet_with_replies = tweet_dict[tweet_id]
                    if tweet_with_replies:
                        replies_ = tweet_with_replies['replies']
                        quotes_ = tweet_with_replies['quotes']
                        replies_correlation.append(replies_)
                        quotes_correlation.append(quotes_)

                        for item in replies_:
                            reply = item['translation2']
                            index_reply = None
                            try:
                                index_reply = replies.index(reply)
                            except ValueError:
                                index_reply = None

                            if index_reply:
                                topic_id = topics_replies[0][index_reply]
                                if topic_id != -1:
                                    topic_docs = model.get_topic_info(topic_id)
                                    for reply_topic_doc in topic_docs.values:
                                        print(f"{'docs :', reply_topic_doc}")
                                        reply_topic_occurence = reply_topic_doc[1]
                                        reply_topic_name = reply_topic_doc[2]
                                        print(f"{'reply topic name :', reply_topic_name}")
                                        if reply_topic_name in topics_occurences:
                                            topics_occurences[reply_topic_name] = topics_occurences[
                                                                                      reply_topic_name] + reply_topic_occurence
                                        else:
                                            topics_occurences[reply_topic_name] = reply_topic_occurence
                        for item in quotes_:
                            reply = item['translation2']
                            index_quote = None
                            try:
                                index_quote = quotes.index(reply)
                            except ValueError:
                                index_quote = None

                            if index_quote:
                                topic_id = topics_quotes[0][index_quote]
                                if topic_id != -1:
                                    topic_docs = model.get_topic_info(topic_id)
                                    for quote_topic_doc in topic_docs.values:
                                        print(f"{'docs :', quote_topic_doc}")
                                        quote_topic_occurence = quote_topic_doc[1]
                                        quote_topic_name = quote_topic_doc[2]
                                        print(f"{'quote topic name :', quote_topic_name}")
                                        if quote_topic_name in topics_occurences:
                                            topics_occurences[quote_topic_name] = topics_occurences[
                                                                                      quote_topic_name] + quote_topic_occurence
                                        else:
                                            topics_occurences[quote_topic_name] = quote_topic_occurence
                result_for_correlation_analysis[topic_name] = {
                    'replies': replies_correlation,
                    'quotes': quotes_correlation
                }

                print(f" original topic name: ", topic_name)
                print(f" topics relations: ", topics_occurences)
                self.normalize_topics_weights(topics_occurences)
                print(f" topics relations after normalization: ", topics_occurences)

                result.append({'topic': topic_name, 'relations': topics_occurences})
        print("start")
        print("Correlation analysis ", result_for_correlation_analysis)
        self.build_correlation_analysis(model, result_for_correlation_analysis)
        # self.build_network_analysis(result)

    def find_topics_evolution_and_popularity(self):
        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            language='english',
            calculate_probabilities=True,
            verbose=True,
        )
        docs = [tweet_dict[d]['tweet']['translation2'] for d in tweet_dict if
                isinstance(tweet_dict[d], dict) and 'translation2' in tweet_dict[d]['tweet']]

        dates = [tweet_dict[d]['tweet']['date'] for d in tweet_dict if
                 isinstance(tweet_dict[d], dict) and 'translation2' in tweet_dict[d]['tweet']]
        # Sample data with timestamps
        data = {
            "text": docs,
            "timestamp": dates
        }
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Segment data into time intervals (e.g., monthly)
        time_intervals = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='M')
        topic_evolution = {}
        topic_popularity = {}
        topic_keywords_over_time = {}

        for time_interval in time_intervals:
            # Filter data for the current time interval
            data_interval = df[
                (df['timestamp'] >= time_interval) & (df['timestamp'] < time_interval + pd.DateOffset(months=1))]

            docs = data_interval['text'].to_list()

            topics, _ = model.fit_transform(docs)
            topics = list(filter(lambda x: x >= 0, topics))

            # Track topic evolution
            topic_evolution[time_interval] = topics
            # Get keywords for each topic
            print('_'.join([x[0] for x in model.get_topic(-1)]))
            topics_keywords = {topic: '_'.join([x[0] for x in model.get_topic(topic)]) for topic in set(topics)}

            print(topics_keywords)

            # Store keywords for each topic
            topic_keywords_over_time[time_interval] = topics_keywords
            # Track topic popularity
            print("Length of topics : ", len(docs))
            try:
                topic_distribution = model.transform(docs)
                topic_popularity[time_interval] = {
                    topic: sum(1 for doc_topics in topic_distribution if topic in doc_topics)
                    for topic in set(topics)}
            except Exception as e:
                print(f"An error occurred during topic modeling: {e}")

        for time_interval, topics in topic_keywords_over_time.items():
            # Visualize topics with keywords
            print("topics keys: ", topics.keys())
            plt.figure(num=str(time_interval), figsize=(12, 8))
            plt.bar(topics.keys(),
                    [textwrap.fill(label, 10) for label in topics.values()], align='center', width=0.4)
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.xlabel('Topic Id')
            plt.ylabel('Keywords')
            plt.title(f'Topic Evolution for {time_interval}')
            plt.show()

    def find_topics_popularity(self):
        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            language='english',
            calculate_probabilities=True,
            verbose=True,
        )

        docs = [tweet_dict[d]['tweet']['translation2'] for d in tweet_dict if
                isinstance(tweet_dict[d], dict) and 'translation2' in tweet_dict[d]['tweet']]

        dates = [tweet_dict[d]['tweet']['date'] for d in tweet_dict if
                 isinstance(tweet_dict[d], dict) and 'translation2' in tweet_dict[d]['tweet']]
        # Sample data with timestamps
        data = {
            "text": docs,
            "timestamp": dates
        }
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Segment data into time intervals (e.g., monthly)
        time_intervals = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='M')
        topic_popularity = {}
        topic_results = {}
        topic_keywords_over_time = {}

        for time_interval in time_intervals:
            topic_popularity = {}
            # Filter data for the current time interval
            data_interval = df[
                (df['timestamp'] >= time_interval) & (df['timestamp'] < time_interval + pd.DateOffset(months=1))]

            docs = data_interval['text'].to_list()

            topics, _ = model.fit_transform(docs)
            topics = list(filter(lambda x: x >= 0, topics))
            topics_keywords = {topic: '_'.join([x[0] for x in model.get_topic(topic)]) for topic in set(topics)}
            try:
                topic_distribution = model.transform(docs)[0]
                topic_popularity[time_interval] = {
                    topic: topic_distribution.count(topic)
                    for topic in set(topics)}

                topic_results[time_interval] = {"popularity": topic_popularity, "keywords": topics_keywords}
            except Exception as e:
                print(f"An error occurred during topic modeling: {e}")

        # Visualize topic popularity and keywords over time
        for time_interval, results in topic_results.items():
            popularity = results["popularity"]
            top_topics_with_count = popularity[time_interval]
            top_topics = top_topics_with_count.keys()
            topic_counts = top_topics_with_count.values()
            print(top_topics_with_count)
            keywords = results["keywords"]
            print("topics : ", top_topics)
            print("counter : ", list(topic_counts))

            # Plot popularity and keywords
            plt.figure(num=str(time_interval), figsize=(12, 8))
            plt.bar([textwrap.fill(keywords[topic], 20) for topic in top_topics], list(topic_counts), color='skyblue',
                    align='center', width=0.4)
            plt.xlabel('Topics')
            plt.ylabel('Popularity (number of docs)')
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            # plt.gca().yaxis.set_major_locator(MultipleLocator(1))
            plt.title(f'Topic Popularity and Keywords for {time_interval}')
            plt.show()


def find_topics_coherence(self, model: BERTopic, docs, topics):
    """

    document_topic_indices, _ = model.transform(docs)
    print(len(document_topic_indices))

    # Initialize co-occurrence matrix
    num_topics = len(set(topics))
    co_occurrence_matrix = defaultdict(int)

    # Iterate through documents
    # Count co-occurrences of topic pairs across all documents
    for doc_topic_index in document_topic_indices:
        for other_topic_index in document_topic_indices:
            co_occurrence_matrix[doc_topic_index, other_topic_index] += 1

    # Print co-occurrence matrix
    for (topic_i, topic_j), co_occurrences in co_occurrence_matrix.items():
        print(f"Topics {topic_i} and {topic_j} co-occur {co_occurrences} times.")

    # Create co-occurrence matrix as a numpy array
    co_occurrence_array = np.zeros((num_topics, num_topics))
    for (topic_i, topic_j), co_occurrences in co_occurrence_matrix.items():
        co_occurrence_array[topic_i, topic_j] = co_occurrences

    # Initialize CountVectorizer to create topic-document matrix
    document_topic_indices, _ = model.transform(docs)
    document_topic_indices = [[topic] for topic in document_topic_indices]
    print(document_topic_indices)
    vectorizer = CountVectorizer(tokenizer=lambda x: x, vocabulary=set(topics))

    # Create topic-document matrix
    topic_document_matrix = vectorizer.fit_transform(document_topic_indices)

    # Calculate cosine similarity between topics
    topic_co_occurrences = cosine_similarity(topic_document_matrix.T)

    # Print topic co-occurrences
    print("Topic Co-occurrences (Cosine Similarity):")
    print(topic_co_occurrences)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence_array, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5)
    plt.title('Topic Co-occurrence Matrix')
    plt.xlabel('Topic Index')
    plt.ylabel('Topic Index')
    plt.show()
    """
    # Sample text data
    documents = ["text of document 1", "text of document 2", ...]
    docs1 = [
        "apple banana cherry",
        "banana cherry date",
        "date apple",
        "banana cherry apple",
        "date banana cherry",
        "Berlin is a big  city"
    ]

    # Tokenize documents
    tokenized_documents = [doc.split() for doc in docs]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    # Train LDA model
    lda_model = models.LdaModel(corpus, id2word=dictionary, passes=10)

    # Get document-topic distributions
    document_topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]

    print(document_topic_distributions)

    # Specify probability threshold
    probability_threshold = 0.4  # Set your desired threshold here

    # Filter topics with probability above the threshold and get keywords
    topics_with_keywords = []
    for document_topic_distribution in document_topic_distributions:
        topics = [(topic, prob) for topic, prob in document_topic_distribution if prob > probability_threshold]
        print("Topics :")
        print(topics)
        topics_with_keywords.append([(topic, lda_model.show_topic(topic)) for topic, _ in topics])

    print(topics_with_keywords)

    # Initialize co-occurrence matrix
    num_topics = lda_model.num_topics
    co_occurrence_matrix = np.zeros((num_topics, num_topics))

    # Update co-occurrence matrix
    for topics in topics_with_keywords:
        for i, (topic_i, keywords_i) in enumerate(topics):
            for j, (topic_j, keywords_j) in enumerate(topics):
                if i != j:
                    co_occurrence_matrix[topic_i][topic_j] += 1
                    co_occurrence_matrix[topic_j][topic_i] += 1

    print(co_occurrence_matrix)

    # Visualize co-occurrence matrix
    plt.figure(tifigsize=(8, 6))
    plt.imshow(co_occurrence_matrix, cmap='viridis', interpolation='nearest')
    plt.title('Topic Co-occurrences')
    plt.xlabel('Keywords for Topics')
    plt.ylabel('Keywords for Topics')
    plt.xticks(range(num_topics),
               [', '.join([word for word, _ in lda_model.show_topic(topic)]) for topic in range(num_topics)],
               rotation=90)
    plt.yticks(range(num_topics),
               [', '.join([word for word, _ in lda_model.show_topic(topic)]) for topic in range(num_topics)])

    # Annotate cells with co-occurrence values
    for i in range(num_topics):
        for j in range(num_topics):
            plt.text(j, i, str(int(co_occurrence_matrix[i, j])), ha='center', va='center', color='white')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def normalize_topics_weights(self, topic_dict: dict):
    if len(topic_dict) > 0:
        print(topic_dict)
        max_num = max(topic_dict.values())
        min_num = min(topic_dict.values())
        if len(topic_dict) == 1:
            topic_dict[list(topic_dict.keys())[0]] = 1
        else:
            for key, number in topic_dict.items():
                normalized_number = (number - min_num) / (max_num - min_num)
                topic_dict[key] = normalized_number


def build_correlation_analysis(self, model: BERTopic, result: dict):
    for topic1, data1 in result.items():
        for topic2, data2 in result.items():
            if topic1 != topic2:
                corr_coeff, p_value = pearsonr(model.transform(data1['replies'])[0],
                                               model.transform(data2['replies'])[0])
                print("Pearson correlation coefficient:", corr_coeff)
                print("p-value:", p_value)


def build_network_analysis(self, result):
    print(f" topics result for graph: ", result)
    for item in result:
        topic = item['topic']
        relations = item['relations']
        G.add_node(topic, type='topic')

        for topic_reply, count in relations.items():
            if topic != topic_reply:
                G.add_edge(topic, topic_reply, weight=count)

    print("Nodes:", G.nodes())
    print("Edges:", G.edges())
    print("Degree Centrality:", nx.degree_centrality(G))

    nt.from_nx(G)
    nt.show("network_analysis_graph.html", notebook=False)


def build_community_detection(self, g: nx.DiGraph):
    # Create a Pyvis network object
    net = Network(height='800px', width='100%')

    # Example: Identify clusters of related topics using community detection
    communities = list(nx.algorithms.community.greedy_modularity_communities(g))
    for idx, community in enumerate(communities):

        print(f"Community {idx + 1}:")
        for node in community:
            print(g.nodes[node]["label"])

    # Add nodes based on community
    for i, comm in enumerate(communities):
        net.add_nodes(comm, label=f'Community {i + 1}')
    # A
