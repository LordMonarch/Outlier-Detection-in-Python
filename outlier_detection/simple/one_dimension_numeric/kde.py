"""
KDE – Kernel Density Estimation (Kerndichteschätzung)

Die Kerndichteschätzung (KDE) ist ein Verfahren, um aus diskreten Daten
eine glatte, kontinuierliche Schätzung der Wahrscheinlichkeitsdichtefunktion (PDF)
zu erzeugen. Im Gegensatz zu Histogrammen basiert die KDE nicht auf festen Bins,
sondern auf der Überlagerung glatter Kernel-Funktionen (z. B. Gauß-Kerne).

KDE liefert eine deutlich weichere und oft präzisere Darstellung der Verteilung
als Histogramme, kann jedoch empfindlich auf Ausreißer reagieren – je nach Kerntyp
und Bandbreite.

Funktionsweise:
1. Für jeden Datenpunkt wird eine Kernel-Funktion („Glocke“) erzeugt.
2. Diese Funktionen werden summiert und normalisiert.
3. Die Glättung wird durch die **Bandbreite** (Bandwidth, h) gesteuert:
       Kleine Bandbreite  → viele Details, potenziell verrauscht
       Große Bandbreite   → starke Glättung, Informationsverlust

Einfluss von Ausreißern:
- Ausreißer erzeugen einen Kernel weit weg vom zentralen Datenbereich.
- Bei kleiner Bandbreite kann dies zu „Nebenwellen“ oder unerwünschten Peaks führen.
- Bei großer Bandbreite wird der Einfluss eines Ausreißers stark verwischt.
- Robustere Alternativen:

Typische Nutzung:
- Glatte Visualisierung der Datenverteilung
- Erkennen von Modalitäten (ein- oder mehrgipflige Verteilungen)
- Explorative Analyse, wenn Histogramme zu grob oder instabil sind
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER
from outlier_detection.simple.one_dimension_numeric import iqr

logger = loguru.logger

NAME_KDE = "kde"


class KDE(Detection):
    """
    KDE: Outlier-Detection mittels Kernel Density Estimation

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Identifikation von Ausreißern basierend auf einer Kernel Density Estimation (KDE)
    sowie zur Visualisierung der Ergebnisse.
    Verwendet zur Ermittlung der Grenzwerte IQR.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_KDE : str
        Name der Spalte, in der die berechneten KDE-Werte gespeichert werden.
    - _bandwidth : float
        Bandbreite des Gaussian-Kernels, der die Glätte der Dichtefunktion steuert.

    Methoden
    -------
    - kde()
        Berechnet die Kernel Density für die Daten, identifiziert seltene Werte
        als Ausreißer basierend auf IQR der KDE-Scores und markiert diese.
    - diagram()
        Visualisiert die Datenverteilung als Histogramm und hebt die identifizierten
        Ausreißer als vertikale Linien hervor.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - bandwidth : float, optional
        Bandbreite für die KDE-Berechnung (Standard: 0.2).

    Beispiel
    -------
    ```
    k = KDE(data=[1, 2, 2, 3, 100, 101], bandwidth=0.3)
    k.kde()
    k.diagram()
    ```
    """

    def __init__(self, data: list[Union[int, float]], bandwidth: float = 0.2):
        super().__init__(data)
        self.bandwidth = bandwidth
        self._q = None

    def kde(self) -> None:
        pdf = self.data
        X = pdf[NAME_DATA].values.reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(X)
        kde_scores = pd.Series(kde.score_samples(X))
        pdf[NAME_KDE] = kde_scores

        self._q = iqr.IQR(kde_scores.to_list())
        self._q.iqr()
        rare_values = self._q.outliers

        pdf[NAME_IS_OUTLIER] = False
        # Index ist identisch, daher Outlier-Index auf Daten Mappen
        for index in rare_values.index.to_list():
            pdf.loc[index, NAME_IS_OUTLIER] = True

        logger.success(
            f"KDE Outliers gefunden bandwidth={self.bandwidth}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(pdf[NAME_KDE], bins="auto", alpha=0.7, color="steelblue")
        xmin, xmax = axes[0].get_xlim()
        xmin = min(self._q.iqr_lower_limit * -1, xmin)
        xmax = max(self._q.iqr_upper_limit, xmax)

        axes[0].set_xlabel(NAME_KDE)
        axes[0].set_ylabel("anzahl")
        axes[0].set_title("Thresholdbereich")
        axes[0].axvline(
            self._q.iqr_upper_limit, color="red", linestyle="-.", linewidth=2
        )
        axes[0].axvspan(self._q.iqr_upper_limit, xmax, color="red", alpha=0.3)
        axes[0].axvline(
            self._q.iqr_lower_limit, color="red", linestyle="-.", linewidth=2
        )
        axes[0].axvspan(self._q.iqr_lower_limit, xmin, color="red", alpha=0.3)

        axes[1].hist(pdf[NAME_DATA], bins="auto", alpha=0.7, color="steelblue")
        axes[1].set_xlabel(NAME_DATA)
        axes[1].set_ylabel("anzahl")
        axes[1].set_title(f"Original Daten mit Ausreißern ({len(self.outliers)})")
        for outlier in self.outliers:
            plt.axvline(outlier, color="red", linestyle="-.", linewidth=2)

        plt.show()


if __name__ == "__main__":
    import pandas as pd
    import outlier_detection.files as f
    from pathlib import Path

    path = Path(f.__file__).parent.joinpath("segment.csv").absolute()
    data = pd.read_csv(path, header=0)
    data = data["hue-mean"].to_list()

    k = KDE(data)
    k.kde()
    print(k.data)
    k.diagram()
