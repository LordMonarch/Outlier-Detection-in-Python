"""
# MAD – Median Absolute Deviation

Der MAD ist ein robustes Streuungsmaß, das beschreibt,
wie stark die Werte einer Verteilung typischerweise vom Median abweichen.
Im Gegensatz zur Standardabweichung ist der MAD unempfindlicher gegenüber Ausreißern.

Berechnung:
1. Median der Daten bestimmen.
2. Für jeden Wert die absolute Abweichung vom Median berechnen:
       |x_i - median|
3. Median dieser absoluten Abweichungen berechnen:
        median(|x_i - median|).
4. Den MAD-Score berechnen: |x_i - median| / median(|x_i - median|)

Formel:
    MAD = |x_i - median| / median(|x_i - median|)


Bedeutungen:
- Kleiner MAD: Die Daten liegen eng um den Median.
- Großer MAD: Starke Streuung bzw. größere Variabilität der Werte.
- Da der MAD robust ist, eignet er sich gut zur Ausreißererkennung.
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt

from outlier_detection.base import Detection
from outlier_detection.columns import (
    NAME_DATA,
    NAME_IS_OUTLIER,
    NAME_MEDIAN,
    NAME_ABS_DIFF,
)

logger = loguru.logger

NAME_MAD = "mad"
NAME_MEDIAN_ABS_DIFF = NAME_MEDIAN + "_" + NAME_ABS_DIFF


class MAD(Detection):
    """
    MAD: Outlier-Detection mittels Median Absolute Deviation

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Berechnung der Median Absolute Deviation (MAD) und zur Visualisierung
    der Ergebnisse.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_MAD : str
        Name der Spalte, in der die normalisierten absoluten Abweichungen gespeichert werden.
    - NAME_MEDIAN: str
        Name der Spalte, in der der Median der Werte gespeichert wird.
    - NAME_ABS_DIFF: str
        Name der Spalte, in der die positive Differenz der Werte und des Median (absolute Abweichung)
        gespeichert sind.
    - NAME_MEDIAN_ABS_DIFF : str
        Name der Spalte, die die Median der absoluten Abweichungen enthält.

    Methoden
    -------
    - mad()
        Berechnet für alle Werte in `data` die MAD und markiert Werte als
        Ausreißer, die den Schwellenwert überschreiten.
    - diagram()
        Visualisiert die MAD-Verteilung als Histogramm und zeigt den
        Schwellenwert für Ausreißer als vertikale Linie an.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - threshold : float, optional
        Schwellenwert für die Ausreißererkennung (Standard: 4.0).

    Beispiel
    -------
    ```
    m = MAD(data=[1, 2, 3, 100], threshold=3.5)
    m.mad()
    m.diagram()
    ```
    """

    def __init__(self, data: list[Union[int, float]], threshold: float = 4.0):
        super().__init__(data)
        self.threshold = threshold

    def mad(self):
        pdf = self.data

        pdf[NAME_MEDIAN] = pdf[NAME_DATA].median()
        pdf[NAME_ABS_DIFF] = (pdf[NAME_DATA] - pdf[NAME_MEDIAN]).abs()
        pdf[NAME_MEDIAN_ABS_DIFF] = pdf[NAME_ABS_DIFF].median()
        pdf[NAME_MAD] = pdf[NAME_ABS_DIFF] / pdf[NAME_MEDIAN_ABS_DIFF]

        pdf[NAME_IS_OUTLIER] = pdf[NAME_MAD] >= self.threshold

        logger.success(
            f"MAD Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(pdf[NAME_MAD], bins="auto", alpha=0.7, color="steelblue")
        xmin, xmax = axes[0].get_xlim()
        xmax = max(self.threshold, xmax)

        axes[0].set_xlabel(NAME_MAD)
        axes[0].set_ylabel("anzahl")
        axes[0].set_title("Thresholdbereich")
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
    import numpy as np

    d = np.random.normal(size=10_000)
    z0 = MAD(list(d))
    z0.mad()
    print(z0.data)
    z0.diagram()
