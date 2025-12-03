"""
Histogram Binning – Erklärung

Histogram Binning bezeichnet den Prozess, numerische Daten in diskrete Intervalle
(Bins) einzuteilen, um ihre Verteilung sichtbar oder analysierbar zu machen.
Jeder Bin repräsentiert dabei einen Wertebereich und zählt, wie viele Datenpunkte
innerhalb dieses Bereichs liegen.

Das Verfahren ist grundlegend für Histogramme, Diskretisierung von Daten und viele
vorbereitende Schritte in der Datenanalyse oder beim maschinellen Lernen.

Grundprinzip:
1. Gesamtbereich der Daten bestimmen (min bis max).
2. Diesen Bereich in eine feste Anzahl gleich großer Bins unterteilen
   oder die Binbreite basierend auf einer Regel berechnen.
3. Für jeden Datenpunkt bestimmen, in welchen Bin er fällt.
4. Die Anzahl der Punkte pro Bin zählen.
5. Sind in einem Bin weniger Punkte als es Bins gibt (Standard 10 Bins), handelt es sich um Ausreißer
"""

from typing import Union

import loguru
import matplotlib.pyplot as plt

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER

logger = loguru.logger


class Histogram(Detection):
    """
    Histogram: Outlier-Detection mittels Histogramm-Binning

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Identifikation von Ausreißern basierend auf selten besetzten Histogramm-Bins
    sowie zur Visualisierung der Ergebnisse.

    Eigenschaften / Konstanten
    --------------------------
    - _bins : int
        Anzahl der Bins, die für die Histogrammberechnung verwendet werden.

    Methoden
    -------
    - histogram()
        Berechnet die Histogramm-Bins für die Daten und markiert alle Werte
        in selten besetzten Bins (weniger als `_bins` Elemente) als Ausreißer.
    - diagram()
        Visualisiert die Datenverteilung als Histogramm und hebt die identifizierten
        Ausreißer als vertikale Linien hervor.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - bins : int, optional
        Anzahl der Histogramm-Bins (Standard: 10).

    Beispiel
    -------
    ```
    h = Histogram(data=[1, 2, 2, 3, 100, 101], bins=5)
    h.histogram()
    h.diagram()
    ```
    """

    def __init__(self, data: list[Union[int, float]], bins: int = 10):
        super().__init__(data)
        self.bins = bins

    def histogram(self) -> None:
        pdf = self.data
        pdf_histo = pd.cut(pdf[NAME_DATA], bins=self.bins, retbins=True)[0]
        counts = pdf_histo.value_counts().sort_index()

        rare_ranges = []
        for v in counts.index:
            count = counts[v]
            if count < self.bins:
                rare_ranges.append(str(v))

        for i in range(len(pdf)):
            pdf[NAME_IS_OUTLIER] = pdf_histo.astype(str).isin(rare_ranges)

        logger.success(
            f"IQR Outliers gefunden bins={self.bins}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(pdf[NAME_DATA], bins=self.bins, alpha=0.7, color="steelblue")
        for outlier in self.outliers:
            axes[0].axvline(outlier, color="red", linestyle="-.", linewidth=2)

        axes[0].set_xlabel(NAME_DATA)
        axes[0].set_ylabel("anzahl")
        axes[0].set_title(f"Ausreißer in Bins={self.bins}")

        axes[1].hist(pdf[NAME_DATA], bins="auto", alpha=0.7, color="steelblue")
        axes[1].set_xlabel(NAME_DATA)
        axes[1].set_ylabel("anzahl")
        axes[1].set_title(f"Original Daten mit Ausreißern ({len(self.outliers)})")
        for outlier in self.outliers:
            plt.axvline(outlier, color="red", linestyle="-.", linewidth=2)

        plt.show()


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import fetch_openml

    data = fetch_openml("segment", version=1, parser="auto")
    data = pd.DataFrame(data.data)
    data = data["hue-mean"].to_list()

    h = Histogram(data)
    h.histogram()
    print(h.data)
    h.diagram()
