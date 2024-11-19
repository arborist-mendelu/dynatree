from fastapi import FastAPI
import dynatree.dynatree as dynatree
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Povolení CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Povolit všechny zdroje (můžete nahradit specifickými doménami)
    allow_credentials=True,
    allow_methods=["*"],  # Povolit všechny metody (GET, POST, atd.)
    allow_headers=["*"],  # Povolit všechny hlavičky
)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/image/")
def read_item(tree="BK01", method="normal", day="2021-03-22", measurement="M03", probe="Elasto(90)"):
    m = dynatree.DynatreeMeasurement(tree=tree, measurement_type=method, day=day, measurement=measurement)
    data = m.data_pulling.loc[:,'Elasto(90)']
    data = data.reset_index()
    return {
        "tree":m.tree, 
        "measurement":m.measurement,
        "method":m.measurement_type, 
        "day":m.day,
        "measurement_data": data.to_json(), 
        "probe":probe}

#http://127.0.0.1:8001/image/?tree=BK01&method=normal&measurement=M03&day=2021-03-22&probe=