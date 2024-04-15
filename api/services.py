import collections
import json
import string

import networkx as nx
import nltk
from bertopic import BERTopic
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy_langdetect import LanguageDetector
from IPython.display import display
from pyvis.network import Network

import gensim
import json
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob

from api.utils import parse_json, identify_similarity_sentence_topics, get_result, calculate_cosine_sim, equal, \
    choose_colors

nltk.download('punkt')
G = nx.DiGraph()
colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray"]
nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", filter_menu=True)


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


'''
Load the english and german models for spacy
'''
nlps = {}
try:
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector", last=True)
    nlp.add_pipe("spacytextblob")
    nlps['en'] = nlp

except:
    os.system("python3 -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    nlp.add_pipe("language_detector", last=True)
    nlps['en'] = nlp

try:
    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("spacytextblob")
    nlps['de'] = nlp
except:
    os.system("python3 -m spacy download de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("spacytextblob")
    nlps['de'] = nlp

nltk.download('stopwords')
sample_tweets = pd.read_csv("test_tweets.csv")
stripped_sample_tweets = sample_tweets[["tweet"]]
tweets_list_temp = []
replies_list = []
tweets_parsed = parse_json('twitter-politics-tweets.jsonl')[0:100000]
print(len(tweets_parsed))
replies_parsed = parse_json('twitter-politics-tweets_reply.jsonl')
quotes_parsed = parse_json('twitter-politics-tweets_quote.jsonl')
print(len(replies_parsed))

for name in tweets_parsed:
    if 'translation2' in name:
        translation2 = name['translation2']
        tweets_list_temp.append(name['translation2'])

for name in replies_parsed:
    if 'text' in name:
        replies_list.append(name['text'])

        # tweets_list_temp.append(name['translation2'])


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


stripped_sample_tweets['clean_text'] = stripped_sample_tweets['tweet'].apply(lambda x: remove_punctuation(x))
stripped_sample_tweets['text_lower'] = stripped_sample_tweets['clean_text'].apply(lambda x: x.lower())
stripped_sample_tweets.head()

tweets_list = []

for text in stripped_sample_tweets["text_lower"]:
    tweets_list.append(text)

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
        print(result)
        return result

    def get_quotes_per_tweet_id(self, tweet_id):
        result = []
        for name in quotes_parsed:
            if 'in_reply_to_tweet_id' in name and name['in_reply_to_tweet_id'] == tweet_id:
                result.append(name)

        return result

    def extract_topics_with_represented_docs(self):
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

        topics, probs = model.fit_transform(tweets_list_temp)
        assigned_docs = model.get_representative_docs()
        result = []
        print(assigned_docs)
        print("assigned docs")
        topic_info = model.get_topic_info()
        print(topic_info)
        for x in topic_info.values:
            if isinstance(x[3], list):
                topics_text = ' '.join(str(x) for x in x[3])
                if x[0] >= 0:
                    represented_docs = x[4]
                    represented_docs_result = []
                    for doc in represented_docs:
                        id = self.get_tweet_id_per_tweet(doc)
                        represented_docs_result.append(
                            {'id': id, 'doc': doc, 'replies': self.get_replies_per_tweet_id(id),
                             'quotes': self.get_quotes_per_tweet_id(id)})

                    result.append(
                        {'Topic': x[0], 'Count': x[1], 'Name': x[2], 'Representation': x[3], 'Topics': topics_text,
                         'Representation_Docs': represented_docs_result})

        with open("data.json", "w+") as file:
            file.write(json.dumps(result))

        self.build_network_analysis(result)

        return result

    def build_network_analysis(self, input):
        tweets = [
            {
                'user': 'user1',
                'text': 'Good morning! #happy',
                'mentions': ['user2']
            },
            {
                'user': 'user2',
                'text': 'Hello world! #fun',
                'mentions': ['user1', 'user3']
            },
            {
                'user': 'user3',
                'text': 'Feeling great today! #smile',
                'mentions': ['user2']
            }
        ]

        # Add nodes for each user
        """
        for index, tweet in enumerate(tweets):
            print()
            G.add_node(tweet['user'] color=colors[index])

        # Add edges for mentions
        for tweet in tweets:
            for mention in tweet['mentions']:
                G.add_edge(tweet['user'], mention)
        """

        colors = choose_colors(input)

        for i, item in enumerate(input):
            topic = item['Topic']
            represented_docs = item['Representation_Docs']
            print(colors[i])
            G.add_node(f"Topic {i}", type='topic', color=colors[i])
            for index, represented_doc in enumerate(represented_docs):
                replies = represented_doc['replies']
                quotes = represented_doc['quotes']
                tweet_id = represented_doc['id']
                # G.add_node(str(tweet_id), type='tweet')
                # G.add_edge(item['Topics'], str(tweet_id), label='tweet')
                # replies = replies_parsed[index * 5:(index * 5) + 5]
                # G.add_node(str(tweet_id), text=represented_doc['doc'], type='tweet', label=topic)
                for reply in replies:
                    if 'translation2' in reply:
                        print()
                        G.add_edge(f"Topic {i}", reply['id'], label=reply['id'])
                        # G.add_node(reply['id'], type='reply')
                        # G.add_edge(item['Topics'], reply['id'], label='reply')
                        # G.add_edge(tweet_id, reply['id'])
                        # G.add_node(str(reply['id']), text=reply['translation2'])
                        # G.add_edge(str(tweet_id), reply['id'])
                for quote in quotes:
                    if 'translation2' in quote:
                        G.add_edge(f"Topic {i}", quote['id'], label=quote['id'])
                        print()
                        # G.add_edge(tweet_id, quote['id'])

        print("Nodes:", G.nodes())
        print("Edges:", G.edges())
        print("Degree Centrality:", nx.degree_centrality(G))
        pos = nx.spring_layout(G)  # Define the layout for the nodes
        # nx.draw(G, with_labels=True, node_color='skyblue', edge_color='red')
        # plt.show()
        """
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10,
                font_color='black')  # Draw the graph
        plt.title("Network Analysis Visualization")  # Set the title for the plot
        plt.show()
        """
        nt.from_nx(G)
        # Visualize the graph
        nt.show("network_analysis_graph.html", notebook=False)


def check_parse_function(self):
    tweets_list_remove_punctuation = [remove_punctuation(x) for x in tweets_list_temp]
    tweets_list_remove_stop_words = [x for x in tweets_list_remove_punctuation if
                                     x not in stop_words]

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        language='english',
        calculate_probabilities=True,
        verbose=True,
        nr_topics="auto"
    )

    topics, probs = model.fit_transform(tweets_list_remove_stop_words)
    topic_labels = model.generate_topic_labels(nr_words=3,
                                               topic_prefix=False,
                                               word_length=10,
                                               separator=", ")
    mappings = model.topic_mapper_.get_mappings()
    topic_info = model.get_topic_info()
    # Get the clusters and their corresponding documents
    clusters = model.get_clusters()

    # Print the top words for each cluster
    for cluster_id, docs in clusters.items():
        print(f"Cluster {cluster_id}:")
        print(model.get_topics(cluster_id))
        print("Documents:")

        print("\n")

    result = []
    for x in topic_info.values:
        if isinstance(x[3], list):
            topics_text = ' '.join(str(x) for x in x[3])

            if x[0] >= 0:
                result.append(
                    {'Topic': x[0], 'Count': x[1], 'Name': x[2], 'Representation': x[3], 'Topics': topics_text,
                     'Representation_Docs': x[4]})

    topics_with_users = []

    for topic in result:
        topics = topic['Topics']
        users_ids = []
        for item in replies_parsed:
            if 'translation2' in item:
                text = item['translation2']
                cosine_sim = calculate_cosine_sim(topics, text)
                if cosine_sim > 0:
                    user_id = item['user_id']
                    if user_id not in users_ids:
                        users_ids.append(user_id)

        for item in quotes_parsed:
            if 'translation2' in item:
                text = item['translation2']
                cosine_sim = calculate_cosine_sim(topics, text)
                if cosine_sim > 0:
                    user_id = item['user_id']
                    if user_id not in users_ids:
                        users_ids.append(user_id)
        topics_with_users.append({'Topics': topic['Representation'], 'users': users_ids})

    users_topics_vector = []
    print(len(topics_with_users))

    for index, topics_with_user in enumerate(topics_with_users):
        topics_text = '_'.join(str(x) for x in topics_with_user['Topics'])
        users_ids = topics_with_user['users']
        print(len(users_ids))
        for user_id in users_ids:
            users_topics_vector.append([topics_text, user_id])
    plt.rcParams.update({'font.size': 4, })
    for i in range(len(users_topics_vector)):
        point = users_topics_vector[i]  # the element ith in data
        x = point[0]  # the first coordenate of the point, x
        y = point[1]  # the second coordenate of the point, y
        # plt.scatter(x, y)
    ##plt.show()
    hair_color = np.array(['blonde', 'brunette', 'red', 'black', 'brunette', 'black', 'red', 'black'])
    eye_color = np.array(['amber', 'gray', 'green', 'hazel', 'amber', 'gray', 'green', 'hazel'])
    skin_color = np.array(['fair', 'brown', 'brown', 'brown', 'fair', 'brown', 'fair', 'fair'])
    person = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    data = pd.DataFrame(
        {'person': person, 'hair_color': hair_color, 'eye_color': eye_color, 'skin_color': skin_color})
    data = data.set_index('person')
    display(data)

    topics_intersections_with_user = []
    for topics_with_user in topics_with_users:
        topics_text_1 = ' '.join(str(x) for x in topics_with_user['Topics'])
        user_ids_1 = topics_with_user['users']
        for topics_with_user_second in topics_with_users:
            topics_text_2 = ' '.join(str(x) for x in topics_with_user_second['Topics'])
            user_ids_2 = topics_with_user_second['users']
            result = collections.Counter(user_ids_1) & collections.Counter(user_ids_2)
            intersected_list = list(result.elements())
            if len(intersected_list) > 0:
                topics_intersections_with_user.append(
                    {'Topic1': topics_text_1, "Topic2": topics_text_2, "users1": user_ids_1, "users2": user_ids_2})

    for item in topics_intersections_with_user:
        entities1 = [(i, i.label_, i.label) for i in nlp(item['Topic1']).ents]
        entities2 = [(i, i.label_, i.label) for i in nlp(item['Topic2']).ents]


def getTopicDetectionResult(self, input):
    data_words = list(self.sent_to_words(input))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_words_nostops = self.remove_stopwords(data_words)
    data_words_bigrams = self.make_bigrams(bigram_mod, data_words_nostops)
    data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    topics = []
    for i, topic in lda_model.show_topics(num_topics=10, formatted=False):
        print('Topic {}: {}'.format(i, ', '.join([word[0] for word in topic])))
        topics.append('Topic {}: {}'.format(i, ', '.join([word[0] for word in topic])))
    return topics


def detect_language(self, text):
    nlp_ = nlps.get('en')
    doc = nlp_(text)
    return doc._.language['language'], doc._.language['score']


def detect_sentiment(self, text):
    detected_language = self.detect_language(text)
    detected_language_found = 'en'
    if detected_language[0] == 'en' or detected_language[0] == 'de':
        detected_language_found = detected_language[0]

    nlp_ = nlps.get(detected_language_found)

    doc = nlp_(text)
    return doc._.blob.polarity


def get_topic_detection_by_bertopic(self, input):
    print(len(tweets_list_temp))
    print(self.detect_sentiment(input))

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=5,
        language='english',
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = model.fit_transform(tweets_list_temp)
    topic_info = model.get_topic_info()
    result = []
    for x in topic_info.values:
        result.append(
            {'Topic': x[0], 'Count': x[1], 'Name': x[2], 'Representation': x[3], 'Representation_Docs': x[4]})

    return result


def remove_stopwords(self, texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(self, bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(self, bigram_mod, trigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def sent_to_words(self, sentence):
    return gensim.utils.simple_preprocess(str(sentence), deacc=True)


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return span.text


'''   Topic
Count
Name
Representation
Representative_Docs
'''
