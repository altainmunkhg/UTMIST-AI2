import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from flask_migrate import Migrate

load_dotenv(dotenv_path=".env.dev", override=True)

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "postgresql://postgres:postgres@localhost:5432/rl_db"
    )
    app.config["MIGRATION_DIR"] = "migrations"
    db.init_app(app)
    migrate = Migrate(app, db)
    with app.app_context():
        from models import Team, Submission, Match  # Import your models here

        db.create_all()  # Create tables for your models
    return app
