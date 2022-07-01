from http import HTTPStatus
import os
from site import USER_SITE
from typing import Optional
from flask import Flask, jsonify, request, redirect, Response, url_for
from database import Database
import flask_security
from flask_security import  auth_required, UserMixin
from flask_login import LoginManager, login_required, login_user, logout_user#, UserMixin
from flask_login import login_user

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
node_database = Database('node_db.json')
gui_database = Database('gui_db.json')
#print("DEUG", app.config['NODE_DB_PATH'], app.config['GUI_DB_PATH'])

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY",
                                          'pf9Wkove4IKEAXvy-cQkeDPhv9Cb3Ag-wyJILbq_dFw')  # to be generated when running environment scripts

app.config['REMEMBER_COOKIE_HTTPONLY'] = False
app.config['REMEMBER_COOKIE_SECURE'] = True

app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = False

ENABLE_REMEMBER_ME = True

login_manager.session_protection = "strong"  # add extra security for REMEMBER_ME cookies (prevent cookies thieves)

class User(UserMixin):
    fs_uniquifier = 12345
    # def is_authenticated(self):
    #     return False
    def is_active(self):
        return True
    def get(self,  id=None):
        return self
    # def is_anonymous(self):
    #     return False

USER = User()

@app.before_first_request
def create_user():
    table = gui_database.db().table('_default')
    table.clear_cache()
    
    query = gui_database.query()
    
    table.insert({"login": "user",
                  "password": 1234})

@app.before_request
def before_request():
    # redirect connections from HTTP to HTTPS
    if not request.is_secure:
        url = request.url.replace('http://', 'https://', 1)
        code = 301
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
    
    if res is not None:
        print("user authenticated!")
        login_user(USER, remember=ENABLE_REMEMBER_ME)
    else: 
        print("user auth failed!")
        # TODO: do the redirection
    return jsonify({'success': True}), 200

@app.route("/logout", methods=['GET'])
def logout2():
    logout_user()
    print("disconnection successful")
    return redirect("/")



@login_manager.user_loader
def load_user(user_id):
    return USER.get(user_id)

# app.before_request(https_redirect)


if __name__ == "__main__":
    
    app.run(ssl_context='adhoc')
