from fastapi import FastAPI
import dynatree.dynatree as dynatree
import matplotlib.pyplot as plt

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/image/")
def read_item(tree, method, day, measurement, probe):
    m = dynatree.DynatreeMeasurement(tree=tree, measurement_type=method, day=day, measurement=measurement)
    data = m.data_pulling.loc[:,'Elasto(90)']
    data = data.iloc[:5].reset_index()
    return {
        "tree":m.tree, 
        "measurement":m.measurement,
        "method":m.measurement_type, 
        "day":m.day,
        "measurement_data": data.to_json(), 
        "probe":probe}

#http://127.0.0.1:8001/image/?tree=BK01&method=normal&measurement=M03&day=2021-03-22&probe=