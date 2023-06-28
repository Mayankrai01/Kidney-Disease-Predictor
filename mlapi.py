#bring in lightweight dependencies 
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app=FastAPI()

class input(BaseModel):
    age: float
    blood_pressure: float
    specific_gravity:float
    albumin:float
    sugar:float
    red_blood_cells: int
    pus_cell: int
    pus_cell_clumps: int
    bacteria: int
    blood_glucose_random: float
    blood_urea: float
    serum_creatinine: float
    sodium: float
    potassium: float
    haemoglobin: float
    packed_cell_volume: float
    white_blood_cell_count: float
    red_blood_cell_count: float
    hypertension: int
    diabetes_mellitus: int
    coronary_artery_disease: int
    appetite: int
    peda_edema: int
    aanemia: int

with open('kidney.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:input):
    df=pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    response = model.predict(df)
    return {"prediction":int(response)}