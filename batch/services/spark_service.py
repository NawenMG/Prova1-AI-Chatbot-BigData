import subprocess
from app.batch.services.ozone_service import OzoneService
from app.batch.services.hive_service import HiveService

class SparkService:
    def __init__(self, spark_master="local[*]"):
        """
        Inizializza il servizio Spark.
        Args:
            spark_master (str): Configurazione del master di Spark.
        """
        self.spark_master = spark_master

    def clean_and_store_data(self, bucket_path, file_name, hive_table, output_path, ozone_service: OzoneService, hive_service: HiveService):
        """
        Pulisce i dati utilizzando Spark e li conserva in Hive.
        Args:
            bucket_path (str): Percorso del bucket Ozone.
            file_name (str): Nome del file da scaricare.
            hive_table (str): Nome della tabella Hive.
            output_path (str): Percorso del file di output pulito.
            ozone_service (OzoneService): Istanza di OzoneService per recuperare i dati.
            hive_service (HiveService): Istanza di HiveService per conservare i dati puliti.
        Returns:
            None
        """
        # Step 1: Scarica il file da Ozone
        input_path = ozone_service.fetch_file(bucket_path, file_name, "data/raw/input_data.csv")

        # Step 2: Avvia un job Spark per la pulizia dei dati
        try:
            subprocess.run(
                [
                    "spark-submit",
                    "--master", self.spark_master,
                    "spark_jobs/data_cleaning.py",
                    input_path,
                    output_path
                ],
                check=True
            )
            print(f"Dati puliti salvati temporaneamente in: {output_path}")

            # Step 3: Carica i dati puliti nella tabella Hive
            hive_service.store_data(output_path, hive_table)
            print(f"Dati puliti caricati con successo nella tabella Hive: {hive_table}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Errore durante il job Spark: {e}")
