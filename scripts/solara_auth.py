
import dataclasses
from typing import Optional, cast, Dict
from passlib.hash import pbkdf2_sha256

import solara
import solara.lab

valid_hashes = [
 "$pbkdf2-sha256$29000$rbU2prR2jhGidK4VgnAu5Q$e.CvUxgiY3uImVIuUTrKYFWRh/eak5oNVS.WMbBt3mI",
 "$pbkdf2-sha256$29000$/F9LSYkx5rx3TmlNiZGSUg$YO/PMhUB9imJjjqoZC48OGLn3UOYq8GmnxhMDdEi9eo",
 "$pbkdf2-sha256$29000$vTem1BojhJCS0vo/B.D8fw$F/XHKhni22p9.kfiRB/c9WqgLMg.NkhgLBr/eTnPmsU",
 "$pbkdf2-sha256$29000$LaXUOmfMGQNAaK31PmesdQ$/CKKV.V6J3SkaUaKI.UEIOub2rYdyI/tcznMzMWF27s",
   ]

@dataclasses.dataclass
class User:
    username: str
    admin: bool = False

# used only to force updating of the page
force_update_counter = solara.reactive(0)

# session storage has no lifecycle management, and will only be cleared when the server is restarted.
session_storage: Dict[str, str] = {}

user = solara.reactive(cast(Optional[User], session_storage.get(solara.get_session_id(), None)))
login_failed = solara.reactive(False)

def check_auth(children):
    if user.value is None:
        children_auth = [LoginForm()]
    else:
        children_auth = children
    return children_auth


def store_in_session_storage(value):
    session_storage[solara.get_session_id()] = value
    force_update_counter.value += 1

username = solara.reactive("")
password = solara.reactive("")

def LoginForm():
    with solara.Card("Přihlášení do autentizované části"):
        solara.Markdown(
        f"""
          * Na username nezáleží
          * Heslo je obvyklé
        """
        )
        solara.InputText(label="Username", value=username)
        solara.InputText(label="Password", password=True, value=password, on_value=lambda x: login(username.value, x))

        solara.Button(label="Login", on_click=lambda: login(username.value, password.value))
        if login_failed.value:
            solara.Error("Wrong username or password")

def login(username: str, password: str):
    # this function can be replace by a custom username/password check
    # if username == "unod" and  (True in [pbkdf2_sha256.verify(password, i) for i in valid_hashes]):
    print(solara.get_session_id() in session_storage)
    print(solara.get_session_id() , session_storage)

    if solara.get_session_id() in session_storage and session_storage[solara.get_session_id()] == True:
        print("The login from session variable")
        login_failed.value = False
        store_in_session_storage(True)
        user.value = True
    elif True in [pbkdf2_sha256.verify(password, i) for i in valid_hashes]:
        print("login, password successfully verified")
        login_failed.value = False
        store_in_session_storage(True)
        user.value = True
    else:
        user.value = False
        print("login failed")
        login_failed.value = True
    force_update_counter.value += 1

def logout():
    session_storage[solara.get_session_id()] = False
    force_update_counter.value += 1

def needs_login(login):
    if (solara.get_session_id() in session_storage) and (session_storage[solara.get_session_id()] == True):
        return False
    return True