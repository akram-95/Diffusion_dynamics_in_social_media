import networkx as nx
import nltk
from bertopic import BERTopic
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd

from api.utils import parse_json, identify_similarity_sentence_topics, get_result, calculate_cosine_sim, equal, \
    choose_colors

nltk.download('punkt')
G = nx.DiGraph()
colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray"]
nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", filter_menu=True)
nltk.download('stopwords')
sample_tweets = pd.read_csv("test_tweets.csv")
stripped_sample_tweets = sample_tweets[["tweet"]]
tweets_list_temp = []
replies_list = []
tweets_parsed = parse_json('twitter-politics-tweets.jsonl')[0:10000]
print(len(tweets_parsed))
replies_parsed = parse_json('twitter-politics-tweets_reply.jsonl')[0:500000]
quotes_parsed = parse_json('twitter-politics-tweets_quote.jsonl')
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
        topics, probs = model.fit_transform(
            [tweet_dict[d]['tweet']['translation2'] for d in tweet_dict if
             isinstance(tweet_dict[d], dict) and 'translation2' in tweet_dict[d]['tweet']])
        topics_replies = model.transform(replies)
        topics_quotes = model.transform(quotes)
        print("length of replies: ", len(topics_replies[0]))

        topic_info = model.get_topic_info()

        result = []

        for x in topic_info.values:
            if x[0] >= 0:
                print(x)
                represented_docs = x[4]
                topic_name = x[2]
                topics_occurences = {}
                for doc in represented_docs:
                    tweet_id = tweet_dict[doc]
                    tweet_with_replies = tweet_dict[tweet_id]
                    if tweet_with_replies:
                        replies_ = tweet_with_replies['replies']
                        quotes_ = tweet_with_replies['quotes']
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
                print(f" original topic name: ", topic_name)
                print(f" topics relations: ", topics_occurences)
                self.normalize_topics_weights(topics_occurences)
                print(f" topics relations after normalization: ", topics_occurences)

                result.append({'topic': topic_name, 'relations': topics_occurences})
        print("start")
        self.build_network_analysis(result)

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
        # Add edges

        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        # Visualize the network
        net.show('community_detection.html')
