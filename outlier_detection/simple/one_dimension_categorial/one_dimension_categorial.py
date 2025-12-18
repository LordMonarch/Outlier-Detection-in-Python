"""
1D-Categorical: Outlier-Detection mittels Cumulative Summe.

Findet Ausreißer in 1-Dimensionalen kategorischen Daten.
Die Daten können beliebigen Typ haben, egal ob int oder str.
Es werden komplette Kategorien als Ausreißer entfernt.

Funktionsweise
1. Zählt erst die Anzahl der Datensätze jeder Kategorie.
2. Bildet dann die Prozentuale cumulative Summe. D.h. die Anteile aller Datensätze werden bus 1.0 aufsummiert.
3. Wendet den Grenzwert auf die Summe an. Entfernt dabei jeweils komplette Kategorien.
"""

import loguru
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER

NAME_CUM_SUM = "cum_sum"
logger = loguru.logger


class OneDimCategorial(Detection):
    """
    1D-Categorical: Outlier-Detection mittels Cumulative Summe.

    Diese Klasse erweitert die Detection-Klasse und bietet Methoden zur
    Identifikation von Ausreißern basierend auf der Berechnung der cumulativen Summe
    sowie zur Visualisierung der Ergebnisse.
    Der Grenzwert wird an den Konstruktor übergeben.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_DATA: str
        Name der Spalten, die die Kategorien enthält. Sie wird auf Ausreißer untersucht.
    - NAME_CUM_SUM : str
        Name der Spalte, in der die berechneten Cumulative Summe gespeichert wird.
    - NAME_IS_OUTLIER : str
        Name der Spalte, ist es ein Ausreißer?
    - cumm_frac: DataFrame
        Enthält die Daten mit zugehöriger cumulative Summe.
    - threshold: float
        Der Grenzwert ab wann es ein Ausreißer ist.

    Methoden
    -------
    - cum_sum()
        Berechnet cumultative Summe für die Daten, identifiziert seltene Werte
        als Ausreißer basierend auf dem gegebenen Threshold und markiert diese.
    - diagram()
        Visualisiert die Datenverteilung als Histogramm und hebt die identifizierten
        Ausreißer als vertikale Linien hervor.

    Parameter
    ---------
    - data : list of int or float
        Die zu prüfenden Datenpunkte.
    - threshold : float, optional
        Grenzwert für Ausreißer (Standard: 0.5, möglich ist auch 1.0).

    Beispiel
    -------
    ```
    odc = OneDimCategorial(data=[1, 2, 2, 3, 100, 101], threshold=0.5)
    odc.cum_sum()
    odc.diagram()
    ```
    """

    def __init__(self, data: list[str], threshold: float = 0.05):
        super().__init__(data)
        self.data[NAME_DATA] = self.data[NAME_DATA].astype("category")
        self.threshold = threshold
        self.cumm_frac = None

    def cum_sum(self) -> None:
        pdf = self.data

        # Gets the count of each unique value
        vc = pdf[NAME_DATA].value_counts()
        vc = vc.sort_values()
        # Gets the cumulative count of each unique value
        cumm_frac = pd.DataFrame({NAME_CUM_SUM: np.cumsum(vc.values) / len(pdf)})
        cumm_frac[NAME_DATA] = vc.index
        self.cumm_frac = cumm_frac

        pdf = pdf.merge(cumm_frac, on=NAME_DATA, how="inner")
        pdf[NAME_IS_OUTLIER] = pdf[NAME_CUM_SUM] < self.threshold
        self.data = pdf

        logger.success(
            f"Categorial cummulative Summe Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        pdf = self.data
        cumm_frac = self.cumm_frac

        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        s = sns.barplot(
            x=cumm_frac[NAME_DATA],
            y=cumm_frac[NAME_CUM_SUM],
            order=cumm_frac[NAME_DATA],
            color="blue",
            ax=axes[0],
        )
        axes[0].set_xlabel(NAME_DATA)
        axes[0].set_ylabel(NAME_CUM_SUM)
        axes[0].set_title("Thresholdbereich")

        axes[0].axvline(
            len(self.outliers.unique()) - 0.5,
            color="red",
            linestyle="-.",
            linewidth=2,
        )
        axes[0].axvspan(len(self.outliers.unique()) - 0.5, -0.5, color="red", alpha=0.3)

        axes[1].hist(pdf[NAME_DATA], bins="auto", alpha=0.7, color="steelblue")
        axes[1].set_xlabel(NAME_DATA)
        axes[1].set_ylabel("anzahl")
        axes[1].set_title(f"Original Daten mit Ausreißern ({len(self.outliers)})")
        for outlier in self.outliers:
            plt.axvline(outlier, color="red", linestyle="-.", linewidth=2)

        plt.show()


if __name__ == "__main__":
    import outlier_detection.files as f
    from pathlib import Path

    path = Path(f.__file__).parent.joinpath("SpeedDating.csv").absolute()
    d = pd.read_csv(path, header=0)
    odc = OneDimCategorial(d["age"].to_list())
    odc.cum_sum()
    print(odc.data)
    odc.diagram()
