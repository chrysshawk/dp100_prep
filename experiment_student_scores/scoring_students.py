import json
import joblib
import numpy as np
from azureml.core.model import Model

def init(): # Called when the service is loaded
    global model
    model_path = Model.get_model_path('student_score_model')
    model = joblib.load(model_path)

def run(raw_data): # Run when a request is received
    data = np.array(json.loads(raw_data)['data'])
    predictions = model.predict(data)
    return json.dumps(predictions)
