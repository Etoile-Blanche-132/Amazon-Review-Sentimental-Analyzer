import json
import pandas as pd
from NLP import Pretreatment

POSITIVE, NEGATIVE = 1, 0

class DATA():
    def __init__(self):
        review, overall = [], []

        with open('Books.json') as file:
            rawData = json.load(file)

        for i in range(len(rawData)):
            if rawData[i]['overall'] = 5:
                review.append(rawData[i]['reviewText'])
                overall.append(POSITIVE)

            elif rawData[i]['overall'] = 1:
                review.append(rawData[i]['reviewText'])
                overall.append(NEGATIVE)

        review = Pretreatment(review)
    
    def Thresholding(data):
        threshold = 10