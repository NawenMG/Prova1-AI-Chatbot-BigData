from hdfs import InsecureClient

class OzoneService:
    def __init__(self, ozone_url, user):
        """
        Inizializza il servizio per interagire con Apache Ozone.
        Args:
            ozone_url (str): URL del servizio Ozone (es: http://localhost:9870).
            user (str): Nome utente per l'accesso a Ozone.
        """
        self.client = InsecureClient(ozone_url, user=user)

    def list_files(self, bucket_path):
        """
        Elenca i file presenti in un bucket specifico.
        Args:
            bucket_path (str): Percorso del bucket su Ozone (es: /bucket_name).
        Returns:
            list: Lista di file nel bucket.
        """
        try:
            return self.client.list(bucket_path)
        except Exception as e:
            raise RuntimeError(f"Errore durante il listing dei file: {e}")

    def fetch_file(self, bucket_path, file_name, local_path):
        """
        Scarica un file specifico da un bucket su Ozone.
        Args:
            bucket_path (str): Percorso del bucket su Ozone (es: /bucket_name).
            file_name (str): Nome del file da scaricare.
            local_path (str): Percorso locale dove salvare il file.
        Returns:
            str: Percorso locale del file scaricato.
        """
        try:
            ozone_file_path = f"{bucket_path}/{file_name}"
            self.client.download(ozone_file_path, local_path)
            return local_path
        except Exception as e:
            raise RuntimeError(f"Errore durante il download del file: {e}")

    def upload_file(self, local_path, bucket_path, file_name):
        """
        Carica un file locale su un bucket Ozone.
        Args:
            local_path (str): Percorso del file locale.
            bucket_path (str): Percorso del bucket su Ozone (es: /bucket_name).
            file_name (str): Nome con cui salvare il file su Ozone.
        Returns:
            str: Percorso completo del file su Ozone.
        """
        try:
            ozone_file_path = f"{bucket_path}/{file_name}"
            with open(local_path, 'rb') as file:
                self.client.write(ozone_file_path, file, overwrite=True)
            return ozone_file_path
        except Exception as e:
            raise RuntimeError(f"Errore durante l'upload del file: {e}")
