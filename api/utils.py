import json
import math
import re
from collections import Counter
from itertools import chain

from nltk import word_tokenize
from nltk.corpus import stopwords
from numpy import dot
from numpy.linalg import norm
import numpy as np, colorsys

cos_sim = lambda x, y: dot(x, y) / (norm(x) * norm(y))


def parse_json(filepath):
    data = []
    for line in open(filepath, "r"):
        if 'translation2' in line:
            data.append(json.loads(line))
    return data


def identify_similarity_sentence_topics(topics, sentence):
    vocab = [token.lower() for token in word_tokenize(sentence)]
    sentence_tokens = [token.lower() for token in word_tokenize(sentence)]
    sentence_vec = [1 if token in sentence_tokens else 0 for token in vocab]
    # print(topics)
    for token_vocab in topics:
        # print(vocab)
        topic_vect = [1 if token.lower() == token_vocab.lower() else 0 for token in vocab]

        print(topic_vect)
        print(cos_sim(sentence_vec, topic_vect))


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)


def get_result(content_a, content_b):
    text1 = content_a
    text2 = content_b

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result


def calculate_cosine_sim(X, Y):
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 = []
    l2 = []

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)

    c = 0
    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
    return cosine


def equal(a, b):
    # Ignore non-space and non-word characters
    regex = re.compile(r'[^\s\w]')
    return regex.sub('', a) == regex.sub('', b)


def choose_colors(node_list):
    num_colors = len(node_list)

    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (30 + np.random.rand() * 70) / 100.0
        saturation = (30 + np.random.rand() * 70) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
