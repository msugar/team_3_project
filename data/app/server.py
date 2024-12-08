from fastapi import FastAPI
import pickle
import numpy as np

# Sorce for code (just 10 minutes video): https://www.youtube.com/watch?v=vA0C0k72-b4

# Load model
model = pickle.load('app/TODO')

# I'm not sure if in our case we need class_names
class_names = np.array([TODO])

#Create app using FastAPI
app = FastAPI()

@app.get('/')

def reed_root():
    return {'message': 'Disease prediction API'}

@app.post('/predict')
def predict(data: dict):
    """
    Predict disease using symptom features 
    im not shure what put here, need your help
    """

features = np.array(data['features'].reshape(1, -1))