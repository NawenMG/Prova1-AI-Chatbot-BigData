from flask import Flask
from config import Config
from flask_cors import CORS

def create_app():
    """
    Crea e configura l'app Flask.
    Returns:
        Flask: Applicazione Flask configurata.
    """
    app = Flask(__name__)

    # Carica la configurazione
    app.config.from_object(Config)

    # Abilita CORS
    CORS(app)

    # Registra blueprint per moduli
    from app.streaming.routes import app as streaming_app
    app.register_blueprint(streaming_app, url_prefix="/streaming")

    # Aggiungi altre registrazioni di blueprint qui se necessario

    # Route di stato
    @app.route("/")
    def home():
        return {"message": "Applicazione Flask inizializzata correttamente."}, 200

    return app
