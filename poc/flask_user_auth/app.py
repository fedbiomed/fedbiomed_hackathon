from http import HTTPStatus
import os
from site import USER_SITE
from typing import Optional
from functools import wraps

#from argon2 import PasswordHasher
from flask import Flask, jsonify, request, redirect, Response, abort, url_for
from database import Database
import flask_security
from flask_security import  auth_required, UserMixin
from flask_login import LoginManager, login_required, login_user, logout_user#, UserMixin
from flask_login import login_user, current_user

from flask_perm import Perm

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
node_database = Database('node_db.json')
gui_database = Database('gui_db.json')
#print("DEUG", app.config['NODE_DB_PATH'], app.config['GUI_DB_PATH'])

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY",
                                          'pf9Wkove4IKEAXvy-cQkeDPhv9Cb3Ag-wyJILbq_dFw')  # to be generated when running environment scripts

# setting for authentification cookies 
app.config['REMEMBER_COOKIE_HTTPONLY'] = False
app.config['REMEMBER_COOKIE_SECURE'] = True

app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = False

ENABLE_REMEMBER_ME = True

login_manager.session_protection = "strong"  # add extra security for REMEMBER_ME cookies (prevent cookies thieves)




class User(UserMixin):
    fs_uniquifier = 12345
    roles = ["BasicUser"]
    # def is_authenticated(self):
    #     return False
    
    def __init__(self):
        self.username = None
        self.password = None

    def is_active(self):
        return True
    def get(self,  id=None):
        return self

def admin_login(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        print("USER", USER.roles)
        if not check_if_user_admin(USER):
            abort(401)
        else:
            value = func(*args, **kwargs)
        return value
    return wrapper
    
    # def is_anonymous(self):
    #     return False


class UserRoles():
    pass

USER = User()

@app.before_first_request
def create_user():
    table = gui_database.db().table('_default')
    table.clear_cache()
    
    query = gui_database.query()
    
    table.update({"login": "user",
                  "password": 12345, 
                  "role": 'admin'})

@app.before_request
def before_request():
    # redirect connections from HTTP to HTTPS
    if not request.is_secure:
        print("REDIRECTION")
        url = request.url.replace('http://', 'https://', 1)
        code = 301 #http permanent redirect
        return redirect(url, code=code)

# def https_redirect() -> Optional[Response]:
#     if request.scheme == 'http':
#         return redirect(url_for(request.endpoint,
#                                 _scheme='https',
#                                 _external=True),
#                         HTTPStatus.PERMANENT_REDIRECT)
        
@app.route("/")
def unprotected():
    return "<p>this content is not protected</p>"

@app.route("/protected")
#@auth_required(grace=100)
@login_required
def protected():
    return "<p>this content is protected</p>"

@app.route('/login', methods=['POST', 'GET'])
def login():
    req = request.json
    
    if req is None:
        return "Please login"
    login = req.get('login', None)
    passwd = req.get('password', None)
    
    if not all((login, passwd)):
        print("ERROR: login or passwd missing")
    
    table = gui_database.db().table('_default')
    table.clear_cache()
    
    query = gui_database.query()
    
    res = table.get((query.login == login) & (query.password == passwd))
    print(res)
    is_admin = True if res.get("role") == 'admin' else False
    print("is_admin", is_admin)
    
    if res is not None:
        print("user authenticated!")
        USER.username = login
        USER.password = passwd
        login_user(USER, remember=ENABLE_REMEMBER_ME)
        
        if is_admin:
            USER.roles = ['admin']  # TODO: use enum classes instead
        return jsonify({'success': True}), 200
    else: 
        print("user auth failed!")
        # TODO: do the redirection
        return jsonify({'success': False}), 401
    

@app.route("/logout", methods=['GET'])
def logout2():
    logout_user()
    USER = User()
    print("disconnection successful")
    # TODO : redirect towards login webpage
    return redirect("/")

# protect a view with a principal for that need
@app.route('/admin')
@login_required
@admin_login
def do_admin_index():
    return Response('Only if you are an admin')

@login_manager.user_loader
def load_user(user_id):
    user = User()
    return USER#.get(user_id)

# app.before_request(https_redirect)

def check_if_user_admin(user: User) -> bool:
    login = user.username
    passwd = user.password
    table = gui_database.db().table('_default')
    table.clear_cache()
    
    query = gui_database.query()
    res = table.get((query.login == login) & (query.password == passwd))
    print("RES", res)
    if res is not None and res.get("role") == "admin":
        user.roles = ["admin"]
        return True
    else:
        user.roles = ["BasicUser"]
        return False 

if __name__ == "__main__":
    
    app.run(ssl_context='adhoc')
