from django.http import JsonResponse
import requests
import json
import os
from dotenv import load_dotenv
from decouple import config
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions


load_dotenv()

backend_url = os.getenv(
    'backend_url', default="http://localhost:3030")
sentiment_analyzer_url = os.getenv(
    'sentiment_analyzer_url',
    default="http://localhost:5050/")


def get_request(endpoint, **kwargs):

    params = ""
    if (kwargs):
        for key, value in kwargs.items():
            params=params+key+"="+value+"&"
    request_url = backend_url+endpoint+"?"+params
    print("GET from {} ".format(request_url))
    try:
        # Call get method of requests library with URL and parameters
        response = requests.get(request_url)
        return response.json()
    except Exception as e:
        # If any error occurs
        print(f"Error: {e}")


# Update the `get_dealerships` render list of dealerships all by default, particular state if state is passed


def get_dealerships(request, state="All"):

    if (state == "All"):
        endpoint = "/fetchDealers"
    else:
        endpoint = "/fetchDealers/"+state
    dealerships = get_request(endpoint)
    return JsonResponse({"status": 200, "dealers": dealerships})


def post_review(data_dict):

    request_url = backend_url+"/insert_review"
    try:
        response = requests.post(request_url, json=data_dict)
        print(response.json())
        return response.json()
    except Exception as e:
        print(f"Error: {e}")


# Calls the Watson NLU API and analyses the sentiment of a review
def analyze_review_sentiments(review_text):
    # Watson NLU configuration
    try:
        if os.environ['env_type'] == 'PRODUCTION':
            url = os.environ['WATSON_NLU_URL']
            api_key = os.environ["WATSON_NLU_API_KEY"]
    except KeyError:
        url = config('WATSON_NLU_URL')
        api_key = config('WATSON_NLU_API_KEY')

    version = '2024-6-11'
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(
        version=version, authenticator=authenticator)
    nlu.set_service_url(url)

    # get sentiment of the review
    try:
        response = nlu.analyze(text=review_text, features=Features(
            sentiment=SentimentOptions())).get_result()
        print(json.dumps(response))
        # sentiment_score = str(response["sentiment"]["document"]["score"])
        sentiment_label = response["sentiment"]["document"]["label"]
    except Exception as e:
        print("Review is too short for sentiment analysis. "
              + "Assigning default sentiment value 'neutral' instead")
        sentiment_label = "neutral"

    # print(sentiment_score)
    print(sentiment_label)

    return sentiment_label
