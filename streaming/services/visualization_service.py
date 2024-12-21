import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np

class VisualizationService:
    @staticmethod
    def plot_results_with_matplotlib(data, output_image="results_matplotlib.png"):
        """
        Rappresenta i risultati con Matplotlib.
        Args:
            data (pd.DataFrame): DataFrame con colonne 'Actual' e 'Predicted'.
            output_image (str): Percorso per salvare il grafico.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Actual'], data['Predicted'], alpha=0.6, label="Predicted vs Actual")
        plt.plot(data['Actual'], data['Actual'], color='red', label="Ideal Line")
        plt.title("Predicted vs Actual (Matplotlib)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.savefig(output_image)
        plt.close()
        print(f"Grafico salvato in: {output_image}")

    @staticmethod
    def plot_results_with_seaborn(data, output_image="results_seaborn.png"):
        """
        Rappresenta i risultati con Seaborn.
        Args:
            data (pd.DataFrame): DataFrame con colonne 'Actual' e 'Predicted'.
            output_image (str): Percorso per salvare il grafico.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Actual', y='Predicted', data=data, alpha=0.6)
        sns.lineplot(x='Actual', y='Actual', color='red', label="Ideal Line")
        plt.title("Predicted vs Actual (Seaborn)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.savefig(output_image)
        plt.close()
        print(f"Grafico salvato in: {output_image}")

    @staticmethod
    def plot_results_with_pandas(data, output_image="results_pandas.png"):
        """
        Rappresenta i risultati con Pandas Plotting.
        Args:
            data (pd.DataFrame): DataFrame con colonne 'Actual' e 'Predicted'.
            output_image (str): Percorso per salvare il grafico.
        """
        ax = data.plot(kind='scatter', x='Actual', y='Predicted', alpha=0.6, figsize=(10, 6), title="Predicted vs Actual (Pandas)")
        ax.plot(data['Actual'], data['Actual'], color='red', label="Ideal Line")
        plt.savefig(output_image)
        plt.close()
        print(f"Grafico salvato in: {output_image}")

    @staticmethod
    def plot_results_with_plotly(data, output_html="results_plotly.html"):
        """
        Rappresenta i risultati con Plotly.
        Args:
            data (pd.DataFrame): DataFrame con colonne 'Actual' e 'Predicted'.
            output_html (str): Percorso per salvare il grafico interattivo.
        """
        fig = px.scatter(data, x='Actual', y='Predicted', title="Predicted vs Actual (Plotly)", opacity=0.6)
        fig.add_shape(type="line", x0=data['Actual'].min(), y0=data['Actual'].min(),
                      x1=data['Actual'].max(), y1=data['Actual'].max(), line=dict(color="Red",))
        fig.write_html(output_html)
        print(f"Grafico interattivo salvato in: {output_html}")

    @staticmethod
    def plot_results_with_pyplot(data, output_image="results_pyplot.png"):
        """
        Rappresenta i risultati con Pyplot.
        Args:
            data (pd.DataFrame): DataFrame con colonne 'Actual' e 'Predicted'.
            output_image (str): Percorso per salvare il grafico.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data['Actual'], data['Predicted'], 'o', alpha=0.6, label="Predicted vs Actual")
        plt.plot(data['Actual'], data['Actual'], '-', color='red', label="Ideal Line")
        plt.title("Predicted vs Actual (Pyplot)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.legend()
        plt.savefig(output_image)
        plt.close()
        print(f"Grafico salvato in: {output_image}")

    @staticmethod
    def compare_results(data, output_image="results_comparison.png"):
        """
        Confronta gli errori con grafici multipli.
        Args:
            data (pd.DataFrame): DataFrame con colonne 'Actual' e 'Predicted'.
            output_image (str): Percorso per salvare il grafico.
        """
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Scatter Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(data['Actual'], data['Predicted'], alpha=0.6)
        plt.plot(data['Actual'], data['Actual'], color='red', label="Ideal Line")
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()

        # Subplot 2: Residuals Plot
        plt.subplot(2, 2, 2)
        residuals = data['Actual'] - data['Predicted']
        plt.hist(residuals, bins=20, alpha=0.7)
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

        # Subplot 3: Error Line Plot
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(len(data)), residuals, label="Residuals", alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', label="Zero Error Line")
        plt.title("Residuals Over Data Points")
        plt.xlabel("Data Points")
        plt.ylabel("Residuals")
        plt.legend()

        # Save the combined plot
        plt.tight_layout()
        plt.savefig(output_image)
        plt.close()
        print(f"Grafico comparativo salvato in: {output_image}")
