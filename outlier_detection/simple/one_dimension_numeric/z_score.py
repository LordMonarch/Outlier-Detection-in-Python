"""
# Z-Score

Der Z-Score ist eine standardisierte Kennzahl, die beschreibt,
wie viele Standardabweichungen ein Wert x vom Mittelwert µ
einer Verteilung entfernt ist.

Formel:
    z = (x - µ) / σ

Bedeutungen:
- z = 0: Wert liegt exakt auf dem Mittelwert.
- z > 0: Wert liegt über dem Mittelwert.
- z < 0: Wert liegt unter dem Mittelwert.
- |z| >= 2 oder 3: Häufiges Kriterium zur Erkennung von Ausreißern.

Der Z-Score wird häufig verwendet zur:
- Normalisierung/Standardisierung von Daten
- Erkennung ungewöhnlicher Werte (Outlier Detection)
- Vergleichbarkeit von Werten unterschiedlicher Skalen
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt

from outlier_detection.base import Detection
from outlier_detection.columns import (
    NAME_IS_OUTLIER,
    NAME_MEAN,
    NAME_STD_DEV,
    NAME_DATA,
)
from outlier_detection.utils import mean, std_dev

logger = loguru.logger

NAME_Z_SCORE = "z-score"


class Z_SCORE(Detection):
    """
    Z_SCORE: Outlier-Detection mittels Z-Score

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Berechnung des Z-Scores und zur Visualisierung der Ergebnisse.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_Z_SCORE : str
        Name der Spalte, in der die berechneten Z-Scores gespeichert werden.
    - NAME_MEAN : str
        Name der Spalte, die den Mittelwert berechnet.
    - NAME_STD_DEV : str
        Name der Spalte, die die Standardabweichung berechnet.

    Methoden
    -------
    - z_score()
        Berechnet für alle Werte in `data` den Z-Score und markiert Werte
        als Ausreißer, die außerhalb des Bereichs [-threshold, threshold] liegen.
    - diagram()
        Visualisiert die Z-Score-Verteilung als Histogramm und zeigt die
        Schwellenwerte für Ausreißer als vertikale Linien an.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - threshold : float, optional
        Schwellenwert für die Ausreißererkennung (Standard: 3.0).

    Beispiel
    -------
    ```
    z = Z_SCORE(data=[1, 2, 3, 100], threshold=2.5)
    z.z_score()
    z.diagram()
    ```
    """

    def __init__(self, data: list[Union[int, float]], threshold: float = 3.0):
        super().__init__(data)
        self.threshold = threshold

    def z_score(self) -> None:
        pdf = self.data
        pdf = mean(pdf, in_col=NAME_DATA, out_col=NAME_MEAN)
        pdf = std_dev(pdf, in_col=NAME_DATA, out_col=NAME_STD_DEV)
        pdf[NAME_Z_SCORE] = (pdf[NAME_DATA] - pdf[NAME_MEAN]) / pdf[NAME_STD_DEV]

        pdf[NAME_IS_OUTLIER] = (pdf[NAME_Z_SCORE] >= self.threshold) | (
                pdf[NAME_Z_SCORE] <= self.threshold * -1
        )

        logger.success(
            f"z-score Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(pdf[NAME_Z_SCORE], bins="auto", alpha=0.7, color="steelblue")
        xmin, xmax = axes[0].get_xlim()
        xmin = min(self.threshold * -1, xmin)
        xmax = max(self.threshold, xmax)

        axes[0].set_xlabel(NAME_Z_SCORE)
        axes[0].set_ylabel("anzahl")
        axes[0].set_title("Thresholdbereich")
        axes[0].axvline(self.threshold, color="red", linestyle="-.", linewidth=2)
        axes[0].axvspan(self.threshold, xmax, color="red", alpha=0.3)
        axes[0].axvline(self.threshold * -1, color="red", linestyle="-.", linewidth=2)
        axes[0].axvspan(self.threshold * -1, xmin, color="red", alpha=0.3)

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
    z0 = Z_SCORE(list(d))
    z0.z_score()
    print(z0.data)
    z0.diagram()
