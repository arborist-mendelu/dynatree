from fastapi import FastAPI
from solara import component

import dynatree.dynatree as dynatree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import JSONResponse
import io
import base64
from bokeh.plotting import figure
import bokeh.embed
from fastapi.responses import HTMLResponse
from bokeh.resources import CDN
from typing import Optional
from bokeh.models import HoverTool
from dynatree.find_measurements import get_all_measurements_acc

df = get_all_measurements_acc()
df["mtype"] = df["date"]+"_"+ df["type"]
df[["mtype"]].drop_duplicates()
all_methods = list(df[["mtype"]].drop_duplicates().values.reshape(-1))
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

@app.get("/methods")
async def methods():
    print("methods fetched")
    return all_methods

@app.get("/method/{method}")
async def method(method:str):
    ans = list(df[df["mtype"] ==  method].loc[:,"tree"].drop_duplicates().values)
    return ans

@app.get("/tree/{tree}/{method}")
async def tree(tree:str, method:str):
    ans = list(df[(df["mtype"] ==  method) & (df["tree"]==tree)].loc[:,"measurement"].drop_duplicates().values)
    return ans

@app.get("/range")
async def get_time_range(
    method: str, tree: str, measurement: str, sensor: str
):
    return {"min": 0.0, "max": 1000.0}

# @app.post("/draw_graph/")
# def read_item(*args, **kwargs):
#     print(f"args {args}")
#     print(f"kwargs {kwargs}")
#     return [args,kwargs]

@app.get("/draw_graph/")
@app.post("/draw_graph/")
def spec_read_itemm(tree="BK01", method="normal", day="2021-03-22", measurement="M02",
              probe="Elasto(90)", start: Optional[float] = None, end: Optional[float] = None,
              format='json'):
    if "_" in method:
        day,method = method.split("_")
    print("   Received request:", tree, method, day, measurement, format, probe)
    m = dynatree.DynatreeMeasurement(tree=tree, measurement_type=method, day=day, measurement=measurement)
    if ("Elasto" in probe) or ("Inclino" in probe):
        data = m.data_pulling
    elif "a0" in probe:
        data = m.data_acc5000
    else:
        data = m.data_optics
    if data is None:
        return None
    data = data.loc[:,probe]

    if (start is not None) and (end is not None):
        data = data.loc[start:end]

    if format in ['html','bokeh']:
        p = figure(title=f"Dynatree plot  {m} {probe}", x_axis_label='Time', y_axis_label='Value',
                   # width=800, height=400,
                   sizing_mode="stretch_both",
                   )
        p.line(data.index, data.values, line_width=2)
        # Přidání HoverTool pro zobrazení souřadnic
        hover = HoverTool()
        hover.tooltips = [("Time", "@x"), ("Value", "@y")]  # Zobrazení hodnot x a y
        p.add_tools(hover)
        if format == 'bokeh':
            script, div = bokeh.embed.components(p)
            response_data  = {"status": "success",
              "graph_data": {
                "target_id": "bokeh-plot",
                "script": script,
                "div": div
              }}
            return JSONResponse(content=response_data)

        html = bokeh.embed.file_html(p, CDN, "DYNATREE_plot")
        return HTMLResponse(content=html)

    fig, ax = plt.subplots(figsize=(12,5))
    data.plot(ax=ax)
    ax.grid()
    ax.set(title=f"{m} {probe}")
    plt.tight_layout()
    # Uložení grafu do paměti jako PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    if format == 'png':
        return Response(content=buffer.getvalue(), media_type="image/png")

    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    data = data.reset_index()
    return {
        "tree":m.tree, 
        "measurement":m.measurement,
        "method":m.measurement_type, 
        "day":m.day,
        "measurement_data": data.to_json(),
        "image": img_base64,
        "probe":probe}

#http://127.0.0.1:8001/image/?tree=BK01&method=normal&measurement=M03&day=2021-03-22&probe=