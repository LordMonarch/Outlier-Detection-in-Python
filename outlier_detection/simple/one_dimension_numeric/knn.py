"""
KNN – k-Nearest-Neighbors Outlier Detection

KNN ist ein Verfahren zur Identifikation von Ausreißern basierend auf
den Abständen zu den k-nächsten Nachbarn eines jeden Datenpunkts.
Die Grundidee: Punkte, die weit von ihren Nachbarn entfernt sind, gelten
als potenzielle Ausreißer. Wenn die Entfernung der Punkte erst abnimmt bis 0 und dann wieder
zunimmt, bildet sich ein 'Schwanz'. Der Kennzeichnet den Grenzwert der **visuel** aus
dem Histogram ermittelt werden muss.

Berechnung:
1. Für jeden Datenpunkt die Abstände zu seinen k nächsten Nachbarn berechnen.
2. Den maximalen Abstand zu den k Nachbarn bestimmen (max_dist).
3. Einen Schwellenwert festlegen; Punkte mit max_dist > threshold werden
   als Ausreißer markiert.

Formel:
    max_dist_i = max(Distanz(x_i, x_j) für j in k-Nachbarn von i)

Bedeutungen:
- Kleiner max_dist: Punkt liegt nah an seinen Nachbarn → typischer Datenpunkt.
- Großer max_dist: Punkt liegt weit entfernt → potenzieller Ausreißer.
- Der Ansatz ist besonders geeignet für Daten ohne feste Verteilungsannahmen.

Typische Nutzung:
- Outlier Detection bei Messreihen oder Feature-Daten
- Analyse von Clustern und räumlicher Streuung
- Identifikation von ungewöhnlichen Datenpunkten in univariaten oder
  mehrdimensionalen Datensätzen
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import BallTree

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER

logger = loguru.logger

NAME_KNN = "knn"


class KNN(Detection):
    """
    KNN: Outlier-Detection mittels k-Nearest-Neighbors

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Berechnung von Ausreißern basierend auf den Abständen zu den k-nächsten
    Nachbarn (k-Nearest-Neighbors, KNN) sowie zur Visualisierung der Ergebnisse.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_KNN : str
        Name der Spalte, in der die maximalen Distanzen zu den k nächsten Nachbarn gespeichert werden.
    - k : int
        Anzahl der Nachbarn, die für die Berechnung der KNN-Distanzen verwendet werden.
    - threshold : float
        Schwellenwert für die Ausreißererkennung basierend auf den maximalen Nachbarabständen.
    - max_dist_arr : array-like
        Speichert die maximalen Distanzen zu den k nächsten Nachbarn.

    Methoden
    -------
    - knn()
        Berechnet für alle Werte in `data` die maximalen Abstände zu den k
        nächsten Nachbarn und markiert Werte als Ausreißer, die den Schwellenwert überschreiten.
    - diagram()
        Visualisiert die Verteilung der KNN-Abstände als Histogramm und zeigt
        den Schwellenwert für Ausreißer als vertikale Linie an.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - k : int, optional
        Anzahl der nächsten Nachbarn, die für die Berechnung verwendet werden (Standard: 25).
    - threshold : float, optional
        Schwellenwert für die Ausreißererkennung (Standard: 0.2).

    Beispiel
    -------
    ```
    k = KNN(data=[1, 2, 3, 100], k=3, threshold=0.5)
    k.knn()
    k.diagram()
    ```
    """

    def __init__(
            self, data: list[Union[int, float]], k: int = 25, threshold: int = 0.2
    ):
        super().__init__(data)
        self.k = k
        self.threshold = threshold
        self.max_dist_arr = None

    def knn(self) -> None:
        pdf = self.data

        X = pdf[NAME_DATA].values.reshape(-1, 1)
        # Create a BallTree and calculates the distances
        # between each pair of records
        tree = BallTree(X, leaf_size=2)
        # Retrieves the distances to the 25 nearest neighbors
        # for each record (k+1, erste ist Distanz zu sich selbst)
        dist, ind = tree.query(X, k=self.k + 1)
        pdf[NAME_KNN] = pd.Series([max(x) for x in dist])
        pdf[NAME_IS_OUTLIER] = pdf[NAME_KNN] > self.threshold

        logger.success(
            f"KNN Outliers gefunden k={self.k} threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        max_dist_arr = pdf[NAME_KNN]
        max_dist_arr = max_dist_arr[max_dist_arr > 0.05]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(max_dist_arr, bins="auto", alpha=0.7, color="steelblue")
        xmin, xmax = axes[0].get_xlim()
        xmax = max(self.threshold, xmax)

        axes[0].set_xlabel(NAME_KNN)
        axes[0].set_ylabel("anzahl")
        axes[0].set_title("Threshold bei 'Schwanzanfang' visuel finden")
        axes[0].axvline(self.threshold, color="red", linestyle="-.", linewidth=2)
        axes[0].axvspan(self.threshold, xmax, color="red", alpha=0.3)

        axes[1].hist(pdf[NAME_DATA], bins="auto", alpha=0.7, color="steelblue")
        axes[1].set_xlabel(NAME_DATA)
        axes[1].set_ylabel("anzahl")
        axes[1].set_title(f"Original Daten mit Ausreißern ({len(self.outliers)})")
        for outlier in self.outliers:
            plt.axvline(outlier, color="red", linestyle="-.", linewidth=2)

        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    d = fetch_openml("segment", version=1, parser="auto")
    d = pd.DataFrame(data=d.data)
    d = d["hue-mean"].to_list()

    k1 = KNN(d)
    k1.knn()
    print(k1.data)
    k1.diagram()
