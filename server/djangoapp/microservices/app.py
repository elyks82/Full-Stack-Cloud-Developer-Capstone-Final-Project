from flask import Flask
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask("Sentiment Analyzer")

sia = SentimentIntensityAnalyzer()

@app.get('/')
def home():
    return "Welcome to the Sentiment Analyzer. Use /analyze/text to get the sentiment"

@app.get('/analyze/<input_txt>')
def analyze_sentiment(input_txt):
    scores = sia.polarity_scores(input_txt)
    print(scores)
    pos = int(scores['pos'])
    neg = int(scores['neg'])
    neu = int(scores['neu'])
    res = "positive"
    if(neg>pos and neg>neu):
        res = "negative"
    elif(neu>neg and neu>pos):
        res = "neutral"
    return res
    
if __name__=="__main__":
    app.run(debug=True) 
