import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from app.batch.services.hive_service import HiveService  # Importa il Hive Service
from app.batch.services.milvus_service import MilvusService  # Importa il Milvus Service
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

    def _prepare_data(self, hive_service: HiveService, query: str):
        """
        Recupera e prepara i dati da Hive.
        Args:
            hive_service (HiveService): Istanza del Hive Service.
            query (str): Query per recuperare i dati.
        Returns:
            tuple: features, labels
        """
        print("Recupero dei dati da Hive...")
        hive_data = hive_service.query_data(query)
        if not hive_data:
            raise RuntimeError("Nessun dato recuperato da Hive.")
        
        data = pd.read_csv(pd.compat.StringIO(hive_data), header=None)
        print(f"Dati recuperati: {data.shape}")
        features = data.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
        labels = data.iloc[:, -1].values    # L'ultima colonna come etichette
        return features, labels

    def collaborative_filtering(self, hive_service: HiveService, query: str):
        """
        Implementa un sistema di raccomandazione basato su Collaborative Filtering.
        Args:
            hive_service (HiveService): Istanza del Hive Service.
            query (str): Query per recuperare i dati.
        Returns:
            tf.keras.Model: Modello addestrato.
        """
        features, labels = self._prepare_data(hive_service, query)
        print("Addestro un sistema di raccomandazione basato su Collaborative Filtering...")
        # Creazione del modello TensorFlow per la fattorizzazione della matrice
        num_users, num_items = features.shape
        user_input = tf.keras.Input(shape=(1,), name="user_input")
        item_input = tf.keras.Input(shape=(1,), name="item_input")

        user_embedding = tf.keras.layers.Embedding(num_users, 64)(user_input)
        item_embedding = tf.keras.layers.Embedding(num_items, 64)(item_input)

        dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, item_embedding])
        output = tf.keras.layers.Flatten()(dot_product)

        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit([features[:, 0], features[:, 1]], labels, epochs=10, batch_size=32)
        model.save(self.model_path)
        print(f"Modello Collaborative Filtering salvato in: {self.model_path}")
        return model

    def demand_forecasting(self, hive_service: HiveService, query: str):
        """
        Previsione della domanda utilizzando LSTM.
        Args:
            hive_service (HiveService): Istanza del Hive Service.
            query (str): Query per recuperare i dati.
        Returns:
            np.ndarray: Predizioni.
        """
        features, labels = self._prepare_data(hive_service, query)
        print("Previsione della domanda utilizzando LSTM...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        # Modello LSTM
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        predictions = model.predict(X_test)
        print("Predizioni LSTM completate.")
        return predictions

    def sentiment_analysis(self, hive_service: HiveService, query: str):
        """
        Classificazione del sentiment utilizzando Reti LSTM.
        Args:
            hive_service (HiveService): Istanza del Hive Service.
            query (str): Query per recuperare i dati.
        Returns:
            tf.keras.Model: Modello addestrato.
        """
        features, labels = self._prepare_data(hive_service, query)
        print("Analisi del sentiment...")
        # Tokenizzazione del testo
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(features.flatten())
        sequences = tokenizer.texts_to_sequences(features.flatten())
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

        # Modello LSTM per Sentiment Analysis
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 128, input_length=100),
            tf.keras.layers.LSTM(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(padded_sequences, labels, epochs=10, batch_size=32)
        print("Modello Sentiment Analysis addestrato.")
        return model

    def customer_segmentation(self, hive_service: HiveService, query: str):
        """
        Segmentazione dei clienti utilizzando Autoencoder e K-means.
        Args:
            hive_service (HiveService): Istanza del Hive Service.
            query (str): Query per recuperare i dati.
        Returns:
            tuple: Autoencoder addestrato, cluster assegnati.
        """
        features, _ = self._prepare_data(hive_service, query)
        print("Segmentazione dei clienti...")

        # Autoencoder
        input_dim = features.shape[1]
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(features, features, epochs=10, batch_size=32)

        encoder = tf.keras.Model(inputs=input_layer, outputs=encoded)
        reduced_features = encoder.predict(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=0)
        clusters = kmeans.fit_predict(reduced_features)
        print(f"Segmentazione completata. Silhouette Score: {silhouette_score(features, clusters)}")
        return autoencoder, clusters

    def churn_prediction(self, hive_service: HiveService, query: str):
        """
        Previsione del tasso di abbandono utilizzando un MLP.
        Args:
            hive_service (HiveService): Istanza del Hive Service.
            query (str): Query per recuperare i dati.
        Returns:
            tf.keras.Model: Modello addestrato.
        """
        features, labels = self._prepare_data(hive_service, query)
        print("Previsione del tasso di abbandono...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        # Modello MLP
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        print("Modello Churn Prediction addestrato.")
        return model
