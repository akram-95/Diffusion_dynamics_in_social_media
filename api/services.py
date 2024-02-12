import string

from bertopic import BERTopic
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import gensim
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

sample_tweets = pd.read_csv("test_tweets.csv")
stripped_sample_tweets = sample_tweets[["tweet"]]


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
nlp = spacy.load('en', disable=['parser', 'ner'])


class ApiService(object):

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

    def get_topic_detection_by_bertopic(self, input):
        print(input)
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

        topics, probs = model.fit_transform(tweets_list)
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


'''   Topic
Count
Name
Representation
Representative_Docs
'''
