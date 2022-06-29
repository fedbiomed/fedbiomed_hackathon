import os
from site import USER_SITE
from flask import Flask, jsonify, request
from database import Database
import flask_security
from flask_security import  auth_required, UserMixin
from flask_login import LoginManager, login_required, login_user#, UserMixin
from flask_login import login_user

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
node_database = Database('node_db.json')
gui_database = Database('gui_db.json')
#print("DEUG", app.config['NODE_DB_PATH'], app.config['GUI_DB_PATH'])

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", 'pf9Wkove4IKEAXvy-cQkeDPhv9Cb3Ag-wyJILbq_dFw')

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
                  "password":1234})

@app.route("/")
def unprotected():
    return "<p>this content is not protected</p>"

@app.route("/protected")
#@auth_required(grace=100)
@login_required
def protected():
    return "<p>this content is protected</p>"

@app.route('/login', methods=['POST'])
def login():
    req = request.json
    login = req.get('login', None)
    passwd = req.get('password', None)
    
    if any((login, passwd)) is None:
        print("ERROR: login or passwd missing")
    
    table = gui_database.db().table('_default')
    table.clear_cache()
    
    query = gui_database.query()
    
    res = table.get((query.login == login) & (query.password == passwd))
    
    if res is not None:
        print("user authenticated!")
        login_user(USER)
    else: 
        print("user auth failed!")
    return jsonify({'success': True}), 200

@login_manager.user_loader
def load_user(user_id):
    return USER.get(user_id)