from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

import solara.server.starlette

def myroot(request: Request):
    return JSONResponse({"framework": "solara"})

routes = [
    Route("/", endpoint=myroot),
    Mount("/dynatree/", routes=solara.server.starlette.routes),
]

app = Starlette(routes=routes)
