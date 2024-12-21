from pymilvus import connections, Collection, utility

class MilvusService:
    def __init__(self, host="localhost", port="19530", collection_name="streaming_embeddings"):
        """
        Inizializza il servizio Milvus.
        Args:
            host (str): Host del server Milvus.
            port (str): Porta del server Milvus.
            collection_name (str): Nome della collezione in Milvus.
        """
        self.collection_name = collection_name
        connections.connect(host=host, port=port)
        print(f"Connesso a Milvus su {host}:{port}")

    def create_collection(self, dim=128):
        """
        Crea una collezione per memorizzare embedding vettoriali.
        Args:
            dim (int): Dimensione degli embedding.
        """
        if utility.has_collection(self.collection_name):
            print(f"La collezione '{self.collection_name}' esiste già.")
            return

        from pymilvus import CollectionSchema, FieldSchema, DataType

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, description="Collezione per embedding in tempo reale")
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collezione '{self.collection_name}' creata con dimensione {dim}.")

    def insert_embeddings(self, embeddings):
        """
        Inserisce embedding vettoriali nella collezione.
        Args:
            embeddings (list): Lista di embedding vettoriali.
        Returns:
            list: ID dei record inseriti.
        """
        collection = Collection(self.collection_name)
        result = collection.insert([embeddings])
        print(f"Inseriti {len(embeddings)} embedding nella collezione '{self.collection_name}'.")
        return result.primary_keys

    def search_embeddings(self, query_vectors, top_k=5):
        """
        Cerca embedding simili a quelli forniti.
        Args:
            query_vectors (list): Vettori di query da cercare.
            top_k (int): Numero di risultati da restituire.
        Returns:
            list: Risultati della ricerca.
        """
        collection = Collection(self.collection_name)
        collection.load()
        results = collection.search(query_vectors, "embedding", {"metric_type": "L2"}, top_k=top_k)
        for result in results:
            print(f"Risultati della ricerca: {result}")
        return results

    def drop_collection(self):
        """
        Elimina la collezione Milvus.
        """
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Collezione '{self.collection_name}' eliminata.")
        else:
            print(f"La collezione '{self.collection_name}' non esiste.")
