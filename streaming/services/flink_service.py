import subprocess
from app.streaming.services.kafka_service import KafkaService
from app.streaming.services.impala_service import ImpalaService

class FlinkService:
    def __init__(self, flink_host="localhost", flink_port=8081):
        """
        Inizializza il servizio Flink.
        Args:
            flink_host (str): Host del servizio Flink.
            flink_port (int): Porta del servizio Flink.
        """
        self.flink_host = flink_host
        self.flink_port = flink_port

    def process_stream(self, flink_job_path, kafka_service: KafkaService, topics, impala_service: ImpalaService, impala_table, group_id="flink_group", max_messages=10):
        """
        Processa i dati in streaming con un job Flink e salva i dati puliti in Impala.
        Args:
            flink_job_path (str): Percorso del job Flink.
            kafka_service (KafkaService): Istanza del servizio Kafka.
            topics (list): Lista dei topic Kafka da consumare.
            impala_service (ImpalaService): Istanza del servizio Impala.
            impala_table (str): Nome della tabella Impala in cui salvare i dati puliti.
            group_id (str): Gruppo consumer Kafka.
            max_messages (int): Numero massimo di messaggi da consumare per topic.
        """
        # Step 1: Consuma i dati dai topic Kafka
        print(f"Consumo dei dati dai topic: {topics}")
        kafka_data = kafka_service.consume_messages(topics=topics, group_id=group_id, max_messages=max_messages)

        # Step 2: Scrivi i dati consumati in un file temporaneo
        temp_input_file = "data/temp_kafka_input.txt"
        with open(temp_input_file, 'w') as f:
            for topic, messages in kafka_data.items():
                for message in messages:
                    f.write(message + '\n')
        print(f"Dati temporanei salvati in: {temp_input_file}")

        # Step 3: Esegui il job Flink con i dati consumati
        temp_output_file = "data/temp_cleaned_output.txt"
        try:
            subprocess.run(
                [
                    "flink", "run",
                    "--jobmanager", f"{self.flink_host}:{self.flink_port}",
                    flink_job_path,
                    temp_input_file,
                    temp_output_file
                ],
                check=True
            )
            print("Job Flink completato.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Errore durante il job Flink: {e}")

        # Step 4: Leggi i dati puliti dal file di output
        cleaned_data = []
        with open(temp_output_file, 'r') as f:
            for line in f:
                cleaned_data.append(line.strip().split(','))  # Supponiamo che i dati siano separati da virgole
        print(f"Dati puliti letti dal file: {temp_output_file}")

        # Step 5: Salva i dati puliti in Impala
        try:
            impala_service.save_data(impala_table, cleaned_data)
            print(f"Dati puliti salvati nella tabella Impala: {impala_table}")
        except Exception as e:
            raise RuntimeError(f"Errore durante il salvataggio dei dati in Impala: {e}")
