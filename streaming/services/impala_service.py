from impala.dbapi import connect as impala

class ImpalaService:
    def __init__(self, host="localhost", port=21050):
        """
        Inizializza il servizio Impala.
        Args:
            host (str): Host del server Impala.
            port (int): Porta del server Impala.
        """
        self.conn = impala.connect(host=host, port=port)
        self.cursor = self.conn.cursor()

    def save_data(self, table_name, data):
        """
        Salva i dati in una tabella Impala.
        Args:
            table_name (str): Nome della tabella in cui salvare i dati.
            data (list): Dati da salvare.
        """
        for record in data:
            query = f"INSERT INTO {table_name} VALUES ({', '.join(record)})"
            self.cursor.execute(query)
        print(f"Dati salvati nella tabella {table_name}.")

    def query_data(self, query):
        """
        Esegue una query su Impala.
        Args:
            query (str): Query SQL.
        Returns:
            list: Risultati della query.
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
