from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import gensim
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups

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
