from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Create FastAPI app
app = FastAPI()

# Enable CORS so your HTML front-end can access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input model
class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load your trained model and scaler
diabetes_model = pickle.load(open("model.sav", "rb"))
diabetes_model_scaler = pickle.load(open("scaler.sav", "rb"))

# Define prediction endpoint
@app.post("/diabetes_prediction")
def diabetes_pred(input_parameters: ModelInput):
    # Convert JSON → dict
    input_dict = input_parameters.model_dump()

    # Convert dict → list (keep correct order)
    input_list = [
        input_dict["Pregnancies"],
        input_dict["Glucose"],
        input_dict["BloodPressure"],
        input_dict["SkinThickness"],
        input_dict["Insulin"],
        input_dict["BMI"],
        input_dict["DiabetesPedigreeFunction"],
        input_dict["Age"],
    ]

    # Scale the input
    scaled_input = diabetes_model_scaler.transform([input_list])

    # Make prediction
    prediction = diabetes_model.predict(scaled_input)

    if prediction[0] == 0:
        return {"result": "The person is not diabetic"}
    else:
        return {"result": "The person is diabetic"}











# 0	6	148	72	35	0	33.6	0.627	50	1
# 1	1	85	66	29	0	26.6	0.351	31	0
# 2	8	183	64	0	0	23.3	0.672	32	1
# 3	1	89	66	23	94	28.1	0.167	21	0
# 4	0	137	40	35	168	43.1	2.288	33	1
