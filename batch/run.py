from flask import Flask
from config import Config
from app.batch.routes import app as streaming_app

# Inizializza Flask
app = Flask(__name__)

# Carica la configurazione
app.config.from_object(Config)

# Registra i blueprint dei moduli
app.register_blueprint(streaming_app, url_prefix="/streaming")

# Aggiungi una route per verificare lo stato
@app.route("/")
def home():
    return {"message": "L'applicazione Flask per il processamento dei dati è attiva!"}, 200


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], host="0.0.0.0", port=5000)
