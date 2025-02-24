from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import RedirectResponse, HTMLResponse
from starlette.routing import Mount, Route
from starlette.middleware.sessions import SessionMiddleware
import solara.server.starlette
from passlib.hash import pbkdf2_sha256


valid_hashes = [
 "$pbkdf2-sha256$29000$rbU2prR2jhGidK4VgnAu5Q$e.CvUxgiY3uImVIuUTrKYFWRh/eak5oNVS.WMbBt3mI",
 "$pbkdf2-sha256$29000$/F9LSYkx5rx3TmlNiZGSUg$YO/PMhUB9imJjjqoZC48OGLn3UOYq8GmnxhMDdEi9eo",
 "$pbkdf2-sha256$29000$vTem1BojhJCS0vo/B.D8fw$F/XHKhni22p9.kfiRB/c9WqgLMg.NkhgLBr/eTnPmsU",
 "$pbkdf2-sha256$29000$LaXUOmfMGQNAaK31PmesdQ$/CKKV.V6J3SkaUaKI.UEIOub2rYdyI/tcznMzMWF27s",
   ]

# Funkce pro kontrolu přihlášení
def is_authenticated(request: Request) -> bool:
    return request.session.get("user") is not None

async def myroot(request: Request):
    print("myroot")
    if not is_authenticated(request):
        return RedirectResponse(url="/login")
    print("redirecting, user authenticated")
    return RedirectResponse(url="/dynatree/")

async def login(request: Request):
    if request.method == "POST":
        form = await request.form()
        heslo = form.get("password")
        if True in [pbkdf2_sha256.verify(heslo, i) for i in valid_hashes]:
            request.session["user"] = "authenticated"
            return RedirectResponse(url="/", status_code=302)
        return HTMLResponse(render_login_page(error="Špatné heslo, zkuste to znovu."), status_code=401)
    return HTMLResponse(render_login_page())

def render_login_page(error: str = "") -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="cs">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Přihlášení</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8f9fa; }}
            .container {{ margin-top: 50px; }}
            .form-section {{
                padding: 20px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .description-section img {{ max-width: 100%; height: auto; }}
            .hidden {{ opacity: 0.5; }}
            .fade-out {{ opacity: 1; transition: opacity 0.5s ease-out; }}
            .fade-out.hidden {{ opacity: 0.5; }}
        </style>
    </head>
    <body>
    <div class="container">
        <div class="row">
            <div class="col-md-6 description-section">
                <img src="../static/sciencists_and_tree_.webp" alt="" class="d-none d-lg-block">
            </div>
            <div class="col-md-6 form-section fade-out" id="content">
                <h2 class="mb-4">Přihlášení do projektu Dynatree</h2>
                <div class="alert alert-primary" role="alert">
                    Web na který se snažíte přistoupit je chráněný heslem. Heslo je obvyklé, používané v projektu.
                </div>
                {f'<div class="alert alert-danger" role="alert">{error}</div>' if error else ''}
                <form method="POST">
                    <div class="form-group">
                        <label for="password">Zadejte heslo:</label>
                        <input type="password" class="form-control" id="password" name="password" placeholder="Heslo">
                        <label>
                            <input type="checkbox" id="remember" name="remember"> Zapamatovat si mě na 31 dní
                        </label>
                    </div>
                    <button type="submit" class="btn btn-primary btn-block">Přihlásit se</button>
                </form>
            </div>
        </div>
    </div>
    <script>
        document.querySelector('form').addEventListener('submit', function() {{
            document.getElementById('content').classList.add('hidden');
        }});
    </script>
    </body>
    </html>
    """

async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

routes = [
    Route("/", endpoint=myroot),
    Route("/login", endpoint=login, methods=["GET", "POST"]),
    Route("/logout", endpoint=logout),
    Mount("/dynatree/", routes=solara.server.starlette.routes),
]

app = Starlette(routes=routes)
app.add_middleware(SessionMiddleware, secret_key="supersecretkeystarlette")
