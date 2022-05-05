from typing import Optional,List
import uvicorn
from fastapi import FastAPI, Response, Request , Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import nltk
import string
class point(BaseModel):
    lat: float
    long: float
class point_arr(BaseModel):
    list : List[point] = []

class data_res(BaseModel):
    UID:str
    message:str
class spam_data(BaseModel):
    st : str
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello"}

@app.post("/data")
def return_UID(data:point_arr): 
    ls = []
    for i in data.list:
        ls.append([i.lat,i.long])
    ls = np.array(ls)
    return k_means (ls) 

    
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
def k_means(X):
    Kmean = KMeans(n_clusters=3)
    Kmean.fit(X)
    centers = Kmean.cluster_centers_
    result_arr = Kmean.labels_
    most_occuring = np.bincount(result_arr).argmax()
    max_radius = 0
    for i in range(len(result_arr)):
        if result_arr[i] == most_occuring:
            dist = np.linalg.norm(centers[most_occuring] - X[i])
            if dist > max_radius:
                max_radius = dist
    center_lat = centers[most_occuring][0]
    center_long = centers[most_occuring][1]
    return {"radius":max_radius,"lat":center_lat,"long":center_long}
def pre_process(sms):
    remove_punct = "".join([word.lower() for word in sms if word not in punctuation])
    tokenize = nltk.tokenize.word_tokenize(remove_punct)
    remove_stopwords = [word for word in tokenize if word not in stopwords]
    return remove_stopwords

def categorize_words():
    spam_words = []
    ham_words = []
    #handling messages associated with spam
    for sms in data['processed'][data['label'] == 'spam']:
        for word in sms:
            spam_words.append(word)
    #handling messages associated with ham
    for sms in data['processed'][data['label'] == 'ham']:
        for word in sms:
            ham_words.append(word)
    return spam_words, ham_words
def predict(sms):
    spam_counter = 0
    ham_counter = 0
    #count the occurances of each word in the sms string
    for word in sms:
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)
    print('***RESULTS***')
    #if the message is ham
    if ham_counter > spam_counter:
        accuracy = round((ham_counter / (ham_counter + spam_counter) * 100))
        print('messege is not spam, with {}% certainty'.format(accuracy))
        return False
    #if the message is equally spam and ham
    elif ham_counter == spam_counter:
        print('message could be spam')
        return True
    #if the message is spam
    else:
        accuracy = round((spam_counter / (ham_counter + spam_counter)* 100))
        print('message is spam, with {}% certainty'.format(accuracy))
        return True
@app.post("/spam")
def spam_filter(stri: spam_data):
    pst = pre_process(stri.st)
    return predict(pst)
if __name__ == "__main__":
    data = pd.read_csv('SMSSpamCollection.txt', sep = '\t', header=None, names=["label", "sms"])
    nltk.download('stopwords')
    nltk.download('punkt')

    stopwords = nltk.corpus.stopwords.words('english')
    punctuation = string.punctuation
    data['processed'] = data['sms'].apply(lambda x: pre_process(x))
    spam_words, ham_words = categorize_words()

    uvicorn.run(app,host =  "0.0.0.0", port = 8000)
    