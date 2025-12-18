"""
# IQR – Interquartile Range

Der Interquartilsabstand (IQR) ist ein robustes Maß für die Streuung von Daten.
Er beschreibt den Bereich, in dem die mittleren 50 % der Werte liegen, und ist daher
weniger anfällig für Ausreißer als die Standardabweichung.

Berechnung:
1. Unteres Quartil (Q1, 25%-Perzentil) bestimmen.
2. Oberes Quartil (Q3, 75%-Perzentil) bestimmen.
3. Differenz bilden:
       IQR = Q3 - Q1

Interpretation:
- Kleiner IQR: Die mittleren 50 % der Daten liegen dicht beieinander.
- Großer IQR: Hohe Streuung im zentralen Datenbereich.
- Der IQR eignet sich besonders gut zur Erkennung von Ausreißern,
  da er auf Quartilen basiert und somit robust gegenüber Extremwerten ist.

Ausreißererkennung (klassische Tukey-Regel):
- Untere Grenze: Q1 - 2.2 (oder 1.5) * IQR
- Obere Grenze: Q3 + 2.2 (oder 1.5) * IQR
  Werte außerhalb dieses Bereichs gelten typischerweise als Ausreißer.
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER

logger = loguru.logger


class IQR(Detection):
    """
    IQR: Outlier-Detection mittels Interquartilsabstand

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Identifikation von Ausreißern auf Basis des Interquartilsabstands (IQR)
    sowie zur Visualisierung der Ergebnisse.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_DATA: str
        Name der Spalten, die die Kategorien enthält. Sie wird auf Ausreißer untersucht.
    - NAME_IS_OUTLIER : str
        Name der Spalte, ist es ein Ausreißer?
    - iqr_lower_limit : float
        Untere Grenze für Ausreißer basierend auf dem IQR.
    - iqr_upper_limit : float
        Obere Grenze für Ausreißer basierend auf dem IQR.

    Methoden
    -------
    - iqr()
        Berechnet den IQR für die Daten und markiert alle Werte als Ausreißer,
        die außerhalb des Bereichs [Q1 - threshold*IQR, Q3 + threshold*IQR] liegen.
    - diagram()
        Visualisiert die Datenverteilung als Histogramm und zeigt die IQR-basierten
        Ausreißergrenzen als vertikale Linien an.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - threshold : float, optional
        Multiplikator für den IQR zur Definition der Ausreißergrenzen
        (Standard: 2.2) Auch 1.5 möglich.

    Beispiel
    -------
    ```
    i = IQR(data=[1, 2, 3, 100], threshold=2.2)
    >> i.iqr()
    >> i.diagram()
    ```
    """

    def __init__(self, data: list[Union[int, float]], threshold: float = 2.2):
        super().__init__(data)
        self.threshold = threshold
        self.iqr_lower_limit = None
        self.iqr_upper_limit = None

    def iqr(self) -> None:
        pdf = self.data

        q1 = pdf[NAME_DATA].quantile(0.25)
        q3 = pdf[NAME_DATA].quantile(0.75)
        igr = q3 - q1

        self.iqr_lower_limit = q1 - (self.threshold * igr)
        self.iqr_upper_limit = q3 + (self.threshold * igr)

        pdf[NAME_IS_OUTLIER] = (pdf[NAME_DATA] >= self.iqr_upper_limit) | (
            pdf[NAME_DATA] <= self.iqr_lower_limit
        )

        logger.success(
            f"IQR Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(pdf[NAME_DATA], bins="auto", alpha=0.7, color="steelblue")
        xmin, xmax = axes[0].get_xlim()
        xmin = min(self.iqr_lower_limit * -1, xmin)
        xmax = max(self.iqr_upper_limit, xmax)

        axes[0].set_xlabel(NAME_DATA)
        axes[0].set_ylabel("anzahl")
        axes[0].set_title("Thresholdbereich")
        axes[0].axvline(self.iqr_upper_limit, color="red", linestyle="-.", linewidth=2)
        axes[0].axvspan(self.iqr_upper_limit, xmax, color="red", alpha=0.3)
        axes[0].axvline(self.iqr_lower_limit, color="red", linestyle="-.", linewidth=2)
        axes[0].axvspan(self.iqr_lower_limit, xmin, color="red", alpha=0.3)

        axes[1].hist(pdf[NAME_DATA], bins="auto", alpha=0.7, color="steelblue")
        axes[1].set_xlabel(NAME_DATA)
        axes[1].set_ylabel("anzahl")
        axes[1].set_title(f"Original Daten mit Ausreißern ({len(self.outliers)})")
        for outlier in self.outliers:
            plt.axvline(outlier, color="red", linestyle="-.", linewidth=2)

        plt.show()


if __name__ == "__main__":
    import numpy as np

    d = np.random.normal(size=10_000)
    z0 = IQR(list(d))
    z0.iqr()
    print(z0.data)
    z0.diagram()
