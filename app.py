from flask import Flask, render_template, redirect, url_for, request, session
from passlib.hash import pbkdf2_sha256
import solara.server.flask

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Pro spravne fungovani session

valid_hashes = [
 "$pbkdf2-sha256$29000$rbU2prR2jhGidK4VgnAu5Q$e.CvUxgiY3uImVIuUTrKYFWRh/eak5oNVS.WMbBt3mI", 
 "$pbkdf2-sha256$29000$/F9LSYkx5rx3TmlNiZGSUg$YO/PMhUB9imJjjqoZC48OGLn3UOYq8GmnxhMDdEi9eo",
 "$pbkdf2-sha256$29000$vTem1BojhJCS0vo/B.D8fw$F/XHKhni22p9.kfiRB/c9WqgLMg.NkhgLBr/eTnPmsU",
 "$pbkdf2-sha256$29000$LaXUOmfMGQNAaK31PmesdQ$/CKKV.V6J3SkaUaKI.UEIOub2rYdyI/tcznMzMWF27s",
   ]

app.register_blueprint(solara.server.flask.blueprint, url_prefix="/dynatree")

@app.route('/')
def home():
    if 'logged_in' in session and session['logged_in']:
        # return render_template('index.html')
        return redirect('/dynatree')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        heslo = request.form['password']
        if  True in [pbkdf2_sha256.verify(heslo, i) for i in valid_hashes]:
            session['logged_in'] = True
            # return render_template('index.html')
            return redirect('/dynatree')
        else:
            return render_template('login.html', error="Špatné heslo. Zkuste to znovu.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

import solara.server.flask

# @app.before_request
# def check_if_logged_in():
#     if not current_user.is_authenticated and request.endpoint not in ['login','register']:
#         # Přesměruj na přihlašovací stránku, pokud uživatel není přihlášen
#         return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
