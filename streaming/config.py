import os

class Config:
    """
    Configurazione di base per l'applicazione Flask e i servizi integrati.
    """
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
    DEBUG = os.getenv("DEBUG", True)

    # Airflow
    AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPICS = os.getenv("KAFKA_TOPICS", "topic1,topic2").split(",")
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "streaming_pipeline_group")

    # Flink
    FLINK_HOST = os.getenv("FLINK_HOST", "localhost")
    FLINK_PORT = int(os.getenv("FLINK_PORT", 8081))
    FLINK_JOB_PATH = os.getenv("FLINK_JOB_PATH", "flink_jobs/data_cleaning_job.py")

    # Impala
    IMPALA_HOST = os.getenv("IMPALA_HOST", "localhost")
    IMPALA_PORT = int(os.getenv("IMPALA_PORT", 21050))
    IMPALA_CLEANED_TABLE = os.getenv("IMPALA_CLEANED_TABLE", "cleaned_streaming_data")

    # TensorFlow
    TENSORFLOW_MODEL_PATH = os.getenv("TENSORFLOW_MODEL_PATH", "models/streaming_model.h5")
    TENSORFLOW_EMBEDDING_DIM = int(os.getenv("TENSORFLOW_EMBEDDING_DIM", 128))

    # Milvus
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "streaming_embeddings")

    # Visualization
    VISUALIZATION_OUTPUT_DIR = os.getenv("VISUALIZATION_OUTPUT_DIR", "output/visualizations")

    # Altre Configurazioni
    DATA_DIR = os.getenv("DATA_DIR", "data")
