from flask_socketio import SocketIO, emit
from flask import Flask, jsonify
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.api.common.experimental.trigger_dag import trigger_dag
from app.streaming.services.kafka_service import KafkaService
from app.streaming.services.flink_service import FlinkService
from app.streaming.services.impala_service import ImpalaService
from app.streaming.services.tensorflow_service import TensorFlowService
from app.streaming.services.visualization_service import VisualizationService
from app.streaming.services.milvus_service import MilvusService
import pandas as pd
import json

# Inizializza Flask
app = Flask(__name__)
socketio = SocketIO(app)

# Inizializza i servizi
kafka_service = KafkaService(bootstrap_servers="localhost:9092")
flink_service = FlinkService()
impala_service = ImpalaService()
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
    dag_id='streaming_data_pipeline',
    default_args=default_args,
    description='Pipeline per il processamento dei dati in streaming',
    schedule_interval=None,  # Nessun trigger automatico, viene avviata manualmente
    catchup=False
) as dag:

    def consume_data_from_kafka():
        """Step 1: Consuma i dati da Kafka."""
        topics = ["topic1", "topic2"]
        group_id = "streaming_pipeline_group"
        kafka_data = kafka_service.consume_messages(topics=topics, group_id=group_id, max_messages=100)
        socketio.emit('update', {'status': 'Kafka data consumed', 'data': kafka_data})
        print(f"Dati consumati da Kafka: {kafka_data}")

    def clean_data_with_flink():
        """Step 2: Pulisci i dati con Flink."""
        flink_job_path = "flink_jobs/data_cleaning_job.py"
        output_table = "cleaned_streaming_data"
        flink_service.process_stream(flink_job_path, kafka_service, topics=["topic1", "topic2"], impala_service=impala_service, impala_table=output_table)
        socketio.emit('update', {'status': 'Data cleaned with Flink', 'table': output_table})
        print(f"Dati puliti salvati in Impala nella tabella: {output_table}")

    def process_with_tensorflow():
        """Step 3: Esegui elaborazione con TensorFlow."""
        query = "SELECT * FROM cleaned_streaming_data"
        results = tensorflow_service.real_time_recommendation(impala_service, milvus_service, query)
        socketio.emit('update', {'status': 'TensorFlow processing completed', 'results': results})
        print(f"Risultati TensorFlow: {results}")

    def save_embeddings_to_milvus():
        """Step 4: Genera e salva gli embedding in Milvus."""
        query = "SELECT * FROM cleaned_streaming_data"
        tensorflow_service.anomaly_detection(impala_service, milvus_service, query)
        socketio.emit('update', {'status': 'Milvus embeddings saved'})
        print("Embedding e anomalie gestiti con TensorFlow e salvati in Milvus.")

    def visualize_results():
        """Step 5: Visualizza i risultati."""
        results = pd.DataFrame({
            "Actual": [1, 2, 3, 4, 5],
            "Predicted": [1.1, 1.9, 3.2, 4.1, 4.8]
        })
        visualization_service.plot_results_with_matplotlib(results)
        socketio.emit('update', {'status': 'Visualization complete', 'results': results.to_dict()})
        print("Visualizzazione completata.")

    # Definisci i task
    consume_data_task = PythonOperator(
        task_id='consume_data_from_kafka',
        python_callable=consume_data_from_kafka
    )

    clean_data_task = PythonOperator(
        task_id='clean_data_with_flink',
        python_callable=clean_data_with_flink
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
    consume_data_task >> clean_data_task >> process_data_task >> save_embeddings_task >> visualize_results_task


# Controller Flask per avviare la pipeline
@app.route('/start_streaming_pipeline', methods=['POST'])
def start_streaming_pipeline():
    """
    Avvia la pipeline definita come DAG in Airflow.
    """
    dag_id = 'streaming_data_pipeline'  # Nome del DAG definito sopra

    try:
        # Trigger del DAG tramite Airflow API
        trigger_dag(dag_id=dag_id)
        return jsonify({"message": f"Pipeline {dag_id} avviata con successo"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, debug=True)
