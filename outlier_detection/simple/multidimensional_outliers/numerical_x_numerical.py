"""
Numerical X Numerical: Outlier-Detection für ein DataFrame mittels MAD-Summe.

Findet Ausreißer im kompletten DataFrame aus numerischen Daten.
Alle Daten müssen vom Typ *float* oder *int* sein.
Berechnet sie MAD-Summe je Zeile. Entsprechend werden komplette Zeilen als Ausreißer entfernt.

Funktionsweise:
1. Der MAD-Score von jeder Spalte wird berechnet.
2. Alle MAD-Scores werden je Spalte mit dem MinMaxScaler auf Werte zwischen 0.0 und 1.0 normalisiert.
3. Die Zeilen-Summe aller normalisierten MAD-Werte wird gebildet.
3. Prüfe den Grenzwert auf die Zeilen-Summe. Ausreißer sind alle Werte die größer als der Grenzwert sind.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from outlier_detection.base import DetectionFullDataFrame
from outlier_detection.columns import NAME_IS_OUTLIER
from outlier_detection.simple.one_dimension_numeric.mad import NAME_MAD, MAD

NAME_MAD_SUM = NAME_MAD + "_sum"


class NumericalFullDataFrame(DetectionFullDataFrame):
    """
    NumericalFullDataFrame: Outlier-Detection für ein DataFrame mittels MAD-Summe.

    Diese Klasse erweitert die DetectionFullDataFrame-Klasse und bietet Methoden zur
    Identifikation von Ausreißern basierend auf der Berechnung der MAD-Summen
    sowie zur Visualisierung der Ergebnisse.
    Es gibt keine Visualisierung der Originaldaten.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_MAD_<i> : str
        Name der Spalte, in der die normalisierten absoluten Abweichungen gespeichert werden.
    - NAME_MAD_SUM : str
        Name der Spalte, in der die Zeilensumme der MAD-Werte steht.
    - NAME_IS_OUTLIER : str
        Name der Spalte, ist es ein Ausreißer?
    - threshold : float
        Grenzwert für die MAD-Summen. Bei Überschreitung ist es ein Ausreißer.
        Der Wert muss von Hand gesetzt werden, vorher Daten analysieren!

    Methoden
    -------
    - _calc_mad(values: list[float])
        Berechnet den MAD-Score für eine übergebene Spalte. Gibt die Ergebnisspalte zurück.
        Das Format ist jeweils *list*.
    - mad_sum()
        Berechnet die MAD-Summen für die Daten, identifiziert
        als Ausreißer basierend auf den gegebenen Grenzwert: *threshold*
        und markiert diese.
    - _build_data_count(vc: DataFrame)
        Erstellt die Hilfstabelle *data_count*. Benötigt als Parameter die Matrix mit den
        gezählten Kategorie-Kombinationen. (vc = df.value_counts([A, B])
    - _build_data_prob(vc: DataFrame)
        Erstellt die Hilfstabelle *data_prob*. Benötigt als Parameter die Matrix mit den
        gezählten Kategorie-Kombinationen. (vc = df.value_counts([A, B])
    - _build_data_expect(vc: DataFrame)
        Erstellt die Hilfstabelle *data_expect*. Benötigt als Parameter die Matrix mit den
        gezählten Kategorie-Kombinationen. (vc = df.value_counts([A, B])
    - diagram()
        Visualisiert die MAD-Verteilung als Histogramm und zeigt den
        Schwellenwert für Ausreißer als vertikale Linie an. DIe Originaldaten werden nicht visualisiert.

    Parameter
    ---------
    - data : pd.DataFrame
        Die zu prüfenden Datenpunkte, sie müssen alle vom Typ *int* oder *float* sein.
    - threshold : float, optional
        Grenzwert für die MAD-Summen. Bei Überschreitung ist es ein Ausreißer.
        Der Wert muss von Hand gesetzt werden, vorher Daten analysieren! (Standard: 8.0)

    Beispiel
    -------
    ```
    cxc = CategorialXCategorial(["A", "B", "C"], ["X", "Y", "Z"], threshold_count=1000, threshold_expectation=0.5)
    cxc.marginal_probabilities()
    cxc.diagram()
    ```
    """

    def __init__(self, data: pd.DataFrame, threshold: float = 8.0):
        super().__init__(data)

        self.data = self.data.astype(float)
        self.mad_scores = None

        self.threshold = threshold

    @staticmethod
    def _calc_mad(values: list[float]) -> list[float]:
        mad = MAD(values)
        mad.mad()
        return mad.data[NAME_MAD]

    def mad_sum(self) -> None:
        pdf = self.data
        mad_scores = pd.DataFrame()
        for index, col_name in enumerate(pdf.columns):
            mad_scores[NAME_MAD + f"_{index}"] = self._calc_mad(
                self.data[col_name].to_list()
            )

        # Falsche Werte ersetzen, hier Alternativen beachten
        mad_scores = mad_scores.replace({np.nan: 0.0, np.inf: 1.0, -np.inf: 0.0})
        transform = MinMaxScaler()
        mad_scores[mad_scores.columns] = transform.fit_transform(mad_scores)

        mad_scores[NAME_MAD_SUM] = mad_scores.sum(axis=1)
        self.mad_scores = mad_scores
        pdf[NAME_IS_OUTLIER] = mad_scores[NAME_MAD_SUM] > self.threshold
        self.data = pdf

        logger.success(
            f"Numerical MAD Summe Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        mad_scores = self.mad_scores
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))

        axes.hist(mad_scores[NAME_MAD_SUM], bins="auto", alpha=0.7, color="steelblue")
        xmin, xmax = axes.get_xlim()
        xmax = max(self.threshold, xmax)

        axes.set_xlabel(NAME_MAD)
        axes.set_ylabel("anzahl")
        axes.set_title(
            f"Thresholdbereich Spalte: {NAME_MAD_SUM} ({len(self.outliers)})"
        )
        axes.axvline(self.threshold, color="red", linestyle="-.", linewidth=2)
        axes.axvspan(self.threshold, xmax, color="red", alpha=0.3)

        plt.show()


if __name__ == "__main__":
    import outlier_detection.files as f
    from pathlib import Path

    path = Path(f.__file__).parent.joinpath("segment.csv").absolute()
    data = pd.read_csv(path, header=0)

    nf = NumericalFullDataFrame(data)
    nf.mad_sum()
    nf.diagram()
