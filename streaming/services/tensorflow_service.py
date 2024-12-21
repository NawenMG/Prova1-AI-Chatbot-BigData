import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from app.streaming.services.milvus_service import MilvusService
from app.streaming.services.impala_service import ImpalaService
import numpy as np

class TensorFlowService:
    def __init__(self, model_path="models/tensorflow_model.h5", milvus_dim=128):
        """
        Inizializza il servizio TensorFlow.
        Args:
            model_path (str): Percorso per salvare il modello addestrato.
            milvus_dim (int): Dimensione degli embedding vettoriali per Milvus.
        """
        self.model_path = model_path
        self.milvus_dim = milvus_dim

    def _prepare_data(self, impala_service: ImpalaService, query: str):
        """
        Recupera e prepara i dati da Impala.
        Args:
            impala_service (ImpalaService): Istanza del servizio Impala.
            query (str): Query per recuperare i dati.
        Returns:
            tuple: features, labels
        """
        print("Recupero dei dati da Impala...")
        impala_data = impala_service.query_data(query)
        if not impala_data:
            raise RuntimeError("Nessun dato recuperato da Impala.")
        
        data = pd.DataFrame(impala_data)
        print(f"Dati recuperati: {data.shape}")
        features = data.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
        labels = data.iloc[:, -1].values    # L'ultima colonna come etichette
        return features, labels

    def real_time_recommendation(self, impala_service: ImpalaService, milvus_service: MilvusService, query: str):
        """
        Sistema di raccomandazione in tempo reale.
        Args:
            impala_service (ImpalaService): Istanza del servizio Impala.
            milvus_service (MilvusService): Istanza del servizio Milvus.
            query (str): Query per recuperare i dati.
        Returns:
            list: Risultati della raccomandazione.
        """
        # Step 1: Prepara i dati
        features, labels = self._prepare_data(impala_service, query)

        # Step 2: Modello di raccomandazione
        user_input = tf.keras.Input(shape=(1,), name="user_input")
        item_input = tf.keras.Input(shape=(1,), name="item_input")

        user_embedding = tf.keras.layers.Embedding(input_dim=features.shape[0], output_dim=64)(user_input)
        item_embedding = tf.keras.layers.Embedding(input_dim=features.shape[1], output_dim=64)(item_input)

        dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, item_embedding])
        output = tf.keras.layers.Flatten()(dot_product)

        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        model.fit([features[:, 0], features[:, 1]], labels, epochs=1, batch_size=32, verbose=1)

        # Step 3: Genera embedding e salva in Milvus
        embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        embeddings = embedding_model.predict(features)
        milvus_service.create_collection(dim=self.milvus_dim)
        milvus_service.insert_embeddings(embeddings)
        print("Embedding salvati in Milvus.")

        return embeddings

    def anomaly_detection(self, impala_service: ImpalaService, milvus_service: MilvusService, query: str):
        """
        Rileva anomalie utilizzando un autoencoder e salva embedding in Milvus.
        Args:
            impala_service (ImpalaService): Istanza del servizio Impala.
            milvus_service (MilvusService): Istanza del servizio Milvus.
            query (str): Query per recuperare i dati.
        Returns:
            list: Anomalie rilevate.
        """
        # Step 1: Prepara i dati
        features, _ = self._prepare_data(impala_service, query)

        # Step 2: Modello Autoencoder
        input_dim = features.shape[1]
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(features, features, epochs=1, batch_size=32, verbose=1)

        # Step 3: Calcola l'errore di ricostruzione
        reconstructed = autoencoder.predict(features)
        reconstruction_error = np.mean(np.abs(features - reconstructed), axis=1)

        # Step 4: Identifica le anomalie
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = features[reconstruction_error > threshold]
        print(f"Anomalie rilevate: {anomalies}")

        # Step 5: Salva embedding in Milvus
        encoder = tf.keras.Model(inputs=input_layer, outputs=encoded)
        embeddings = encoder.predict(features)
        milvus_service.create_collection(dim=self.milvus_dim)
        milvus_service.insert_embeddings(embeddings)
        print("Embedding salvati in Milvus.")

        return anomalies

    def contextual_bandits(self, context, actions):
        """
        Implementa Contextual Bandits per decisioni dinamiche.
        Args:
            context (np.ndarray): Contesto dell'utente.
            actions (np.ndarray): Azioni disponibili.
        Returns:
            int: Azione scelta.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(context.shape[1],)),
            tf.keras.layers.Dense(actions.shape[0], activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        probabilities = model.predict(context)
        chosen_action = np.argmax(probabilities, axis=1)[0]
        print(f"Azione scelta: {chosen_action}")
        return chosen_action
