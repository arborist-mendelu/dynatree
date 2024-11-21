from flask import Flask, render_template, redirect, url_for, request, session
from passlib.hash import pbkdf2_sha256
import solara.server.flask
import random
#import logging
#logger = logging.getLogger("flask")

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Pro spravne fungovani session

solara_app = solara.server.flask.app
solara_app.root_path = "./dynatree"

valid_hashes = [
 "$pbkdf2-sha256$29000$rbU2prR2jhGidK4VgnAu5Q$e.CvUxgiY3uImVIuUTrKYFWRh/eak5oNVS.WMbBt3mI", 
 "$pbkdf2-sha256$29000$/F9LSYkx5rx3TmlNiZGSUg$YO/PMhUB9imJjjqoZC48OGLn3UOYq8GmnxhMDdEi9eo",
 "$pbkdf2-sha256$29000$vTem1BojhJCS0vo/B.D8fw$F/XHKhni22p9.kfiRB/c9WqgLMg.NkhgLBr/eTnPmsU",
 "$pbkdf2-sha256$29000$LaXUOmfMGQNAaK31PmesdQ$/CKKV.V6J3SkaUaKI.UEIOub2rYdyI/tcznMzMWF27s",
   ]

app.register_blueprint(solara.server.flask.blueprint, url_prefix="/dynatree")
#logger.setLevel(logging.WARNING)


@app.route('/')
def home():
#    logger.warning(f"Home entered {session}")
    if not 'logged_in' in session or not session['logged_in']:
        return redirect('/login')
        # return render_template('index.html')
    return redirect('/dynatree/')

@app.route('/login', methods=['GET', 'POST'])
def login():
#    logger.warning(f"Login entered {session}")
#     next_page = request.args.get('next')  # Získání cílové stránky
#     print (f"Next page is {next_page}")
    if request.method == 'POST':
        heslo = request.form['password']
        remember = request.form.get('remember')
        if  True in [pbkdf2_sha256.verify(heslo, i) for i in valid_hashes]:
            session['logged_in'] = True
            if remember:
                session.permanent = True
            else:
                session.permanent = False
            next_page = request.args.get('next')  # Načti `next` parametr z URL
            return redirect(next_page or '/dynatree/')  # Přesměrování na původní stránku nebo na domovskou
            # return render_template('index.html')
            # return redirect('/dynatree/')
        else:
            return render_template('login.html', error="Špatné heslo. Zkuste to znovu.")
    image_number = random.randint(0, 5)
    return render_template('login.html', image=image_number)

@app.route('/logout')
def logout():
#    logger.warning(session)
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.before_request
def check_if_logged_in():
    target_url = request.path
    print(f"target url is {request.path}")
    if "/api" in target_url:
        return
    if not 'logged_in' in session or not session['logged_in']:
        if request.endpoint not in ['login', 'register', 'static', 'blueprint-solara.public']:
            # Přesměruj na přihlašovací stránku, pokud uživatel není přihlášen
            return redirect(url_for('login', next=target_url))

@app.route('/api')
def api():
    return "Tady bude api, není nutné se přihlašovat. Možná."

if __name__ == '__main__':
    app.run(debug=True)
