""" Database Connection details"""

import mysql.connector
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

mydb = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="admin",
    database='NEWS_ARTICLES'
)
my_cursor = mydb.cursor(dictionary=True)


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://admin:admin@localhost/NEWS_ARTICLES'
app.config['SECRET_KEY'] = 'mysecret'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app=app)

