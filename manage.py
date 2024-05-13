#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import json
import gzip

from tqdm import tqdm
from elasticsearch.helpers import scan
from elasticsearch import Elasticsearch

from api import utils
from api.TopicInteractionAnalyzer import TopicInteractionAnalyzer
from api.services import ApiService
from dotenv import load_dotenv

# from api.services import ApiService
# from api.utils import parse_json
load_dotenv()
elascticsearch_url = os.getenv("elascticsearch_url")
username = os.getenv("username")
password = os.getenv("password")

es = Elasticsearch(
    elascticsearch_url,
    basic_auth=(username, password),
    request_timeout=30,
    retry_on_timeout=True,
)

INDEX_TWEETS = "twitter-politics-tweets"
INDEX_REPLIES = "twitter-politics-tweets_reply"
INDEX_QUOTE = "twitter-politics-tweets_quote"
fields_tweet = [
    "id",
    "user_id",
    "date",
    "text",
    "lang",
    "translation2",
    "sentiment"
]
fields_reply = [
    "id",
    "user_id",
    "in_reply_to_tweet_id",
    "date",
    "text",
    "lang",
    "translation2",
    "sentiment"
]
fields_quote = [
    "id",
    "user_id",
    "quoted_tweet_id",
    "date",
    "text",
    "lang",
    "translation2",
    "sentiment"
]

q = {
    "query": {
        "bool": {
            "filter": [
                {"term": {
                    "lang": {
                        "value": "de",
                    }
                }},
                {"range": {
                    "date": {
                        "gte": "2021-01-01",
                        "lte": "2024-01-01"
                    }
                }}
            ]
        }
    }
}


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Diffusion_dynamics_in_social_media.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    """
    with gzip.open(f"{INDEX_REPLIES}.jsonl.gz", "wt") as f:
        for hit in tqdm(scan(
                es, index=INDEX_REPLIES, size=10000, query={**q, "_source": fields_reply, "sort": ["_doc"]}
        )):
            f.write(json.dumps(hit["_source"]) + "\n")
            
    """

    # print(utils.calculate_cosine_sim("I love horror movies", "Lights out is a horror movie"))
    TopicInteractionAnalyzer().find_topics_interaction_evolution_overtime()

    main()
