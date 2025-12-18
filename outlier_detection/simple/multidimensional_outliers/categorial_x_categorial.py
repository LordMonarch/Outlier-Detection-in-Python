"""
Categorical X Categorial: Outlier-Detection für zwei Categorial Spalten mittels Randwahrscheinlichkeit.

Findet Ausreißer in 2-Dimensionalen kategorischen Daten.
Die Daten können beliebigen Typ haben, egal ob int oder str.
Es werden komplette Kategorien als Ausreißer entfernt.

Die Tabelle wird umgewandelt in Hilfstabellen. Diese hat als Zeilen eine Kategorie und als Index die andere Kategorie.
Dadurch werden alle Kombinationen abgebildet.

Funktionsweise:
1. Zählt erst die Anzahl der Datensätze jeder vorhanden Kategorie-Kombination.
2. Ermittel für jede Kombination die Eintrittswahrscheinlichkeit bzw. Erwartungswert mit der Formel:
(Summe(Kategorie in Spalte) / Datensatz_gesamt) * (Summe(Kategorie in Zeile) / Datensatz_gesamt) * Datensatz_gesamt
3. Ermittel die Abweichung vom Erwartungswert:
Teile paarweise Matrix aus 1. durch Matrix aus 2.
4. Prüfe den Grenzwert der Matrix 1. mit der Anzahl der Datensätze
und den Grenzwert der Matrix 3. mit der Abweichung vom Erwartungswert.
Liegt eine Kategorie-Kombination unter **beiden** Grenzwerten handelt es sich um einen Ausreißer.
"""

import loguru
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from outlier_detection.base import DetectionPaired
from outlier_detection.columns import (
    NAME_DATA,
    NAME_OTHER,
    NAME_SUM,
    NAME_COUNT,
    NAME_PROBABILITY,
    NAME_IS_OUTLIER,
)

NAME_CUM_SUM = "cum_sum"
logger = loguru.logger


class CategorialXCategorial(DetectionPaired):
    """
    Categorical X Categorial: Outlier-Detection für zwei Categorial Spalten mittels Randwahrscheinlichkeit.

    Diese Klasse erweitert die DetectionPaired-Klasse und bietet Methoden zur
    Identifikation von Ausreißern basierend auf der Berechnung der Randwahrscheinlichkeit
    sowie zur Visualisierung der Ergebnisse.
    Die beiden Grenzwerte werden an den Konstruktor übergeben.

    Eigenschaften / Konstanten
    --------------------------
    - NAME_DATA: str
        Name der Spalten, die die Kategorien enthält. Sie wird auf Ausreißer untersucht.
    - NAME_CUM_SUM : str
        Name der Spalte, in der die berechneten Cumulative Summe gespeichert wird.
    - NAME_IS_OUTLIER : str
        Name der Spalte, ist es ein Ausreißer?
    - NAME_SUM : str
        Name der Spalten und Zeilensummen in den Hilfstabellen, bei der ausgabe über die Methode *print_sum_df(df)*
    - NAME_COUNT : str
        Name der Spalte, in der die Anzahl der Kategorie-Kombination steht
    - NAME_PROBABILITY : str
        Name der Spalte, für die Abweichung vom Erwartungswert **nicht des Erwartungswertes!**
    - data_count : DataFrame
        Hilfstabelle mit der Anzahl der Kategorie-Kombinationen.
    - data_prob : DataFrame
        Hilfstabelle mit dem Erwartungswert der Kategorie-Kombinationen.
    - data_expect : DataFrame
        Hilfstabelle mit der Abweichung vom Erwartungswert der Kategorie-Kombination.
        (Tatsächliche Eintrittswahrscheinlichkeit)
    - threshold_count : int
        Grenzwert für die Hilfstabelle *data_count*. Bei Unterschreitung ist es ein Ausreißer.
    - threshold_expectation : float
        Grenzwert für die Hilfstabelle *data_expectation*. Bei Unterschreitung ist es ein Ausreißer.

    Methoden
    -------
    - print_sum_df(df: pd.DataFrame)
        Das übergebene DataFrame wird um alle Zeilen und Spalten Summen sowie die Gesamtsumme ergänzt.
        Verändert das Dataframe nicht. Hat keinen Rückgabewert, führt nur den *print(df)* aus.
    - marginal_probabilities()
        Berechnet Randwahrscheinlichkeit für die Daten, identifiziert seltene Werte
        als Ausreißer basierend auf den gegebenen Thresholds: *threshold_count* und *threshold_expectation*
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
        Visualisiert die Datenverteilung mit zwei HeatMaps. Die erste basiert auf der Anzahl,
        die zweite auf der Abweichung vom Erwartungswert.
        Hebt die unterschreitung der Grenzwerte hervor.
        Ausreißer sind alle die in beiden Fällen die Grenzwerte unterschreiten.

    Parameter
    ---------
    - data : list of str
        Die zu prüfenden Datenpunkte der einen Kategorie.
    - other_data : list of str
        Die zu prüfenden Datenpunkte der anderen Kategorie.
    - threshold_count : int, optional
        Grenzwert für Ausreißer der Hilfstabelle *data_count* (Standard: 1000,
        muss von Hand nach Prüfung der Daten gesetzt werden).
    - threshold_expectation Grenzwert für die Hilfstabelle *data_expectation*. (Standard: 0.5,
        Wert sollte deutlich unter 1.0 liegen)

    Beispiel
    -------
    ```
    cxc = CategorialXCategorial(["A", "B", "C"], ["X", "Y", "Z"], threshold_count=1000, threshold_expectation=0.5)
    cxc.marginal_probabilities()
    cxc.diagram()
    ```
    """

    def __init__(
        self,
        data: list[str],
        other_data: list[str],
        threshold_count: int = 1000,
        threshold_expectation: float = 0.5,
    ):
        super().__init__(data, other_data)

        self.data[NAME_DATA] = self.data[NAME_DATA].astype("category")
        self.data[NAME_OTHER] = self.data[NAME_OTHER].astype("category")

        self.threshold_count = threshold_count
        self.threshold_expectation = threshold_expectation

        self.data_count = None
        self.data_prob = None
        self.data_expect = None

    @staticmethod
    def print_sum_df(df) -> None:
        df = df.copy()
        df[NAME_SUM] = df.sum(axis=1)
        df.loc[NAME_SUM] = df.sum(axis=0)
        print(df.astype(float).round(2))

    def _build_data_count(self, vc: pd.DataFrame) -> None:
        df = self.data.copy()

        pdf = pd.DataFrame(
            columns=df[NAME_DATA].unique(), index=df[NAME_OTHER].unique()
        )
        for row in vc.index:
            pdf.loc[row[1], row[0]] = vc.loc[(row[0], row[1])]
        pdf.columns.name = NAME_DATA
        pdf.index.name = NAME_OTHER
        self.data_count = pdf

    def _build_data_prob(self, vc: pd.DataFrame) -> None:
        df = self.data_count.copy()
        count = df.values.sum()

        pdf = df.copy()
        for row in vc.index:
            pdf.loc[row[1], row[0]] = (
                (df.loc[row[1]].sum() / count) * (df[row[0]].sum() / count) * count
            )
        self.data_prob = pdf

    def _build_data_expect(self) -> None:
        self.data_expect = self.data_count / self.data_prob

    def marginal_probabilities(self) -> None:
        pdf = self.data
        vc = pdf.value_counts([NAME_DATA, NAME_OTHER])
        self._build_data_count(vc)
        self._build_data_prob(vc)
        self._build_data_expect()

        # self.print_sum_df(self.data_count)
        # self.print_sum_df(self.data_prob)
        # self.print_sum_df(self.data_expect)

        # Zählmatrix zusammenbauen
        df_count = self.data_count.copy()
        df_count[NAME_OTHER] = df_count.index
        df_count = df_count.reset_index(drop=True)
        df_count = df_count.melt(
            id_vars=NAME_OTHER,
            var_name=NAME_DATA,
            value_name=NAME_COUNT,
            ignore_index=False,
        )

        # Wahrscheinlichkeitsmatrix zusammenbauen
        df_expect = self.data_expect.copy()
        df_expect[NAME_OTHER] = df_expect.index
        df_expect = df_expect.reset_index(drop=True)
        df_expect = df_expect.melt(
            id_vars=NAME_OTHER, var_name=NAME_DATA, value_name=NAME_PROBABILITY
        )

        pdf = pdf.merge(df_count, on=[NAME_DATA, NAME_OTHER], how="inner")
        pdf = pdf.merge(df_expect, on=[NAME_DATA, NAME_OTHER], how="inner")

        pdf[NAME_IS_OUTLIER] = (pdf[NAME_COUNT] < self.threshold_count) & (
            pdf[NAME_PROBABILITY] < self.threshold_expectation
        )
        pdf = pdf.drop_duplicates()
        self.data = self.data.merge(pdf, on=[NAME_DATA, NAME_OTHER], how="inner")

        logger.success(
            f"Categorial cummulative Summe Outliers gefunden threshold_cunt={self.threshold_count}, threshold_prob={self.threshold_expectation}: {len(self.outliers)}/{len(self.data)}"
        )

    def diagram(self) -> None:
        df_count = self.data_count.astype(int)
        df_expect = self.data_expect.astype(float).round(2)
        fig, axes = plt.subplots(1, 2, figsize=(21, 7))
        s1 = sns.heatmap(
            df_count,
            annot=True,
            vmin=0,
            vmax=self.threshold_count,
            cmap="Reds_r",
            ax=axes[0],
        )
        axes[0].set_xlabel(NAME_DATA)
        axes[0].set_ylabel(NAME_OTHER)
        axes[0].set_title(f"Threshold für Anzahl ({len(self.outliers)})")
        axes[0].axis("equal")

        s2 = sns.heatmap(
            df_expect,
            annot=True,
            vmin=0.0,
            vmax=self.threshold_expectation,
            cmap="Reds_r",
            ax=axes[1],
        )
        axes[1].set_xlabel(NAME_DATA)
        axes[1].set_ylabel(NAME_OTHER)
        axes[1].set_title(f"Threshold für Erwartungswert ({len(self.outliers)})")
        axes[1].axis("equal")

        plt.show()


if __name__ == "__main__":
    import outlier_detection.files as f

    data = f.categorial_x_categorial()

    cxc = CategorialXCategorial(data[NAME_DATA].to_list(), data[NAME_OTHER].to_list())
    cxc.marginal_probabilities()
    print(cxc.data)
    cxc.diagram()
