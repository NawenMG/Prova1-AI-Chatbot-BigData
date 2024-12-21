import subprocess

class HiveService:
    def __init__(self, hive_host="localhost"):
        """
        Inizializza il servizio Hive.
        Args:
            hive_host (str): Host del server Hive.
        """
        self.hive_host = hive_host

    def store_data(self, data_path, table_name):
        """
        Carica i dati in una tabella Hive.
        Args:
            data_path (str): Percorso del file CSV da caricare.
            table_name (str): Nome della tabella Hive dove caricare i dati.
        """
        try:
            print(f"Carico i dati da {data_path} nella tabella {table_name}...")
            subprocess.run(
                [
                    "hive",
                    "-e",
                    f"LOAD DATA LOCAL INPATH '{data_path}' INTO TABLE {table_name}"
                ],
                check=True
            )
            print(f"Dati caricati con successo nella tabella {table_name}.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Errore durante il caricamento dei dati in Hive: {e}")

    def query_data(self, query):
        """
        Esegue una query Hive e restituisce i risultati.
        Args:
            query (str): Query Hive da eseguire.
        Returns:
            str: Risultato della query.
        """
        try:
            print(f"Eseguo la query: {query}")
            result = subprocess.run(
                ["hive", "-e", query],
                capture_output=True, text=True, check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Errore durante l'esecuzione della query Hive: {e}")

    def check_table_exists(self, table_name):
        """
        Verifica se una tabella Hive esiste.
        Args:
            table_name (str): Nome della tabella da verificare.
        Returns:
            bool: True se la tabella esiste, False altrimenti.
        """
        try:
            query = f"SHOW TABLES LIKE '{table_name}'"
            result = self.query_data(query)
            return table_name in result
        except Exception as e:
            raise RuntimeError(f"Errore durante la verifica della tabella {table_name}: {e}")
