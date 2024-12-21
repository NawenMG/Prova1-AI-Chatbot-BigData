import os

class Config:
    """
    Configurazione di base per Flask e Airflow.
    """
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
    DEBUG = os.getenv("DEBUG", True)

    # Airflow
    AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")

    # Ozone
    OZONE_URL = os.getenv("OZONE_URL", "http://localhost:9870")
    OZONE_USER = os.getenv("OZONE_USER", "ozone_user")

    # Spark
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

    # Hive
    HIVE_HOST = os.getenv("HIVE_HOST", "localhost")

    # TensorFlow
    TENSORFLOW_MODEL_PATH = os.getenv("TENSORFLOW_MODEL_PATH", "models/tensorflow_model.h5")

    # Milvus
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

    # Visualizzazioni
    VISUALIZATION_OUTPUT_DIR = os.getenv("VISUALIZATION_OUTPUT_DIR", "output/visualizations")
