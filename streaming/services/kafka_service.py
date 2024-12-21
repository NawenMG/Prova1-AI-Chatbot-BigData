from kafka import KafkaConsumer

class KafkaService:
    def __init__(self, bootstrap_servers):
        """
        Inizializza il servizio Kafka.
        Args:
            bootstrap_servers (str): Indirizzo del broker Kafka (es: "localhost:9092").
        """
        self.bootstrap_servers = bootstrap_servers

    def consume_messages(self, topics, group_id="default_group", max_messages=10):
        """
        Consuma i messaggi da uno o più topic Kafka.
        Args:
            topics (list): Lista dei topic da consumare.
            group_id (str): ID del gruppo consumer.
            max_messages (int): Numero massimo di messaggi da consumare.
        Returns:
            dict: Messaggi consumati, raggruppati per topic.
        """
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda x: x.decode('utf-8')
        )

        messages = {topic: [] for topic in topics}
        for message in consumer:
            topic = message.topic
            if topic in messages:
                messages[topic].append(message.value)
                print(f"Messaggio ricevuto da topic '{topic}': {message.value}")
                # Interrompiamo se raggiungiamo il limite per il topic
                if len(messages[topic]) >= max_messages:
                    topics.remove(topic)
            # Se tutti i topic hanno raggiunto il limite, interrompiamo
            if not topics:
                break

        return messages
