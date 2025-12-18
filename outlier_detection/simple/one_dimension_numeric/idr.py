"""
# IDR – Interdecile Range

Der Interdezilenabstand (IDR) ist ein robustes Streuungsmaß und beschreibt
den Wertebereich, in dem die mittleren 80 % der Daten liegen. Er wird als
Differenz zwischen dem 90. Perzentil (D9) und dem 10. Perzentil (D1) berechnet.

Damit ist der IDR deutlich weniger anfällig für Ausreißer als z. B.
die Standardabweichung, aber empfindlicher als der IQR.

Berechnung:
1. 10%-Perzentil (D1) bestimmen.
2. 90%-Perzentil (D9) bestimmen.
3. Differenz bilden:
       IDR = D9 - D1

Interpretation:
- Kleiner IDR: Die zentralen 80 % der Daten liegen eng zusammen.
- Großer IDR: Höhere Streuung im Hauptdatenbereich.
- Da 10 % der kleinsten und 10 % der größten Werte ignoriert werden,
  bietet der IDR einen guten Kompromiss zwischen Robustheit und Sensitivität.

Ausreißererkennung (klassische Tukey-Regel):
- Untere Grenze: Q1 - 1.0 * IDR
- Obere Grenze: Q3 + 1.0 * IDR
  Werte außerhalb dieses Bereichs gelten typischerweise als Ausreißer.
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER

logger = loguru.logger


class IDR(Detection):
    """
    IDR: Outlier-Detection mittels Interdecile Range

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Identifikation von Ausreißern auf Basis des Interdecile Range (IDR)
    sowie zur Visualisierung der Ergebnisse.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_DATA: str
        Name der Spalten, die die Kategorien enthält. Sie wird auf Ausreißer untersucht.
    - NAME_IS_OUTLIER : str
        Name der Spalte, ist es ein Ausreißer?
    - iqr_lower_limit : float
        Untere Grenze für Ausreißer basierend auf dem IDR.
    - iqr_upper_limit : float
        Obere Grenze für Ausreißer basierend auf dem IDR.

    Methoden
    -------
    - idr()
        Berechnet den Interdecile Range (IDR) für die Daten und markiert alle
        Werte als Ausreißer, die außerhalb des Bereichs [Q1 - threshold*IDR, Q9 + threshold*IDR] liegen.
    - diagram()
        Visualisiert die Datenverteilung als Histogramm und zeigt die IDR-basierten
        Ausreißergrenzen als vertikale Linien an.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - threshold : float, optional
        Multiplikator für den IDR zur Definition der Ausreißergrenzen
        (Standard: 1.0).

    Beispiel
    -------
    ```
    i = IDR(data=[1, 2, 3, 100], threshold=1.0)
    i.idr()
    i.diagram()
    ```
    """

    def __init__(self, data: list[Union[int, float]], threshold: float = 1.0):
        super().__init__(data)
        self.threshold = threshold
        self.iqr_lower_limit = None
        self.iqr_upper_limit = None

    def idr(self) -> None:
        pdf = self.data

        q1 = pdf[NAME_DATA].quantile(0.1)
        q3 = pdf[NAME_DATA].quantile(0.9)

        idr = q3 - q1

        self.iqr_lower_limit = q1 - (self.threshold * idr)
        self.iqr_upper_limit = q3 + (self.threshold * idr)

        pdf[NAME_IS_OUTLIER] = (pdf[NAME_DATA] >= self.iqr_upper_limit) | (
            pdf[NAME_DATA] <= self.iqr_lower_limit
        )

        logger.success(
            f"IDR Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
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
    z0 = IDR(list(d))
    z0.idr()
    print(z0.data)
    z0.diagram()
