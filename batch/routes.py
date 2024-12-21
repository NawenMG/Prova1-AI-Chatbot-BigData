from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from flask import Flask, jsonify
from airflow.api.common.experimental.trigger_dag import trigger_dag
from app.batch.services.ozone_service import OzoneService
from app.batch.services.spark_service import SparkService
from app.batch.services.hive_service import HiveService
from app.batch.services.tensorflow_service import TensorFlowService
from app.batch.services.visualization_service import VisualizationService
from app.batch.services.milvus_service import MilvusService
import pandas as pd

# Inizializza Flask
app = Flask(__name__)

# Inizializza i servizi
ozone_service = OzoneService(ozone_url="http://localhost:9870", user="ozone_user")
spark_service = SparkService()
hive_service = HiveService()
tensorflow_service = TensorFlowService()
visualization_service = VisualizationService()
milvus_service = MilvusService()

# Configurazioni della pipeline
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Crea il DAG
with DAG(
    dag_id='batch_data_pipeline',
    default_args=default_args,
    description='Pipeline per il processamento dei dati batch',
    schedule_interval='@daily',
    catchup=False
) as dag:

    def fetch_raw_data():
        """Step 1: Scarica i dati grezzi da Ozone."""
        bucket_path = "/my_bucket"
        file_name = "raw_data.csv"
        local_path = "data/raw/raw_data.csv"
        ozone_service.fetch_file(bucket_path, file_name, local_path)
        print(f"Dati grezzi scaricati in: {local_path}")

    def clean_data_with_spark():
        """Step 2: Pulisci i dati con Spark."""
        bucket_path = "/my_bucket"
        file_name = "raw_data.csv"
        hive_table = "cleaned_data"
        output_path = "data/cleaned/cleaned_data.csv"
        spark_service.clean_and_store_data(bucket_path, file_name, hive_table, output_path, ozone_service, hive_service)
        print(f"Dati puliti salvati in Hive nella tabella: {hive_table}")

    def process_with_tensorflow():
        """Step 3: Esegui elaborazione con TensorFlow."""
        query = "SELECT * FROM cleaned_data"
        results = tensorflow_service.demand_forecasting(hive_service, query)
        print(f"Risultati TensorFlow: {results[:5]}")  # Stampa le prime 5 predizioni

    def save_embeddings_to_milvus():
        """Step 4: Genera e salva gli embedding in Milvus."""
        query = "SELECT * FROM cleaned_data"
        tensorflow_service.train_and_store_embeddings(hive_service, milvus_service, query)
        print("Embedding salvati in Milvus.")

    def visualize_results():
        """Step 5: Visualizza i risultati."""
        results = pd.DataFrame({
            "Actual": [1, 2, 3, 4, 5],  # Placeholder
            "Predicted": [1.1, 1.9, 3.2, 4.1, 4.8]  # Placeholder
        })
        visualization_service.plot_results_with_matplotlib(results)
        print("Visualizzazione completata.")

    # Definisci i task
    fetch_data_task = PythonOperator(
        task_id='fetch_raw_data',
        python_callable=fetch_raw_data
    )

    clean_data_task = PythonOperator(
        task_id='clean_data_with_spark',
        python_callable=clean_data_with_spark
    )

    process_data_task = PythonOperator(
        task_id='process_with_tensorflow',
        python_callable=process_with_tensorflow
    )

    save_embeddings_task = PythonOperator(
        task_id='save_embeddings_to_milvus',
        python_callable=save_embeddings_to_milvus
    )

    visualize_results_task = PythonOperator(
        task_id='visualize_results',
        python_callable=visualize_results
    )

    # Definisci la sequenza dei task
    fetch_data_task >> clean_data_task >> process_data_task >> save_embeddings_task >> visualize_results_task


# Controller Flask per avviare la pipeline
@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    """
    Avvia la pipeline definita come DAG in Airflow.
    """
    dag_id = 'batch_data_pipeline'  # Nome del DAG definito sopra

    try:
        # Trigger del DAG tramite Airflow API
        trigger_dag(dag_id=dag_id)
        return jsonify({"message": f"Pipeline {dag_id} avviata con successo"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
