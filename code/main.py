from typing import Optional,List
import uvicorn
from fastapi import FastAPI, Response, Request , Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class point(BaseModel):
    lat: float
    long: float
class point_arr(BaseModel):
    list : List[point] = []

class data_res(BaseModel):
    UID:str
    message:str
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
if __name__ == "__main__":
    uvicorn.run(app,host =  "0.0.0.0", port = 8000)