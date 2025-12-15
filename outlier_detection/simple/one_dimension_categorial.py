import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from outlier_detection.base import Detection
from outlier_detection.columns import NAME_DATA

NAME_CUM_SUM = "cum_sum"

class OneDimCategorial(Detection):
    def __init__(self, data: list[str], threshold: float = 0.05):
        super().__init__(data)
        self.data[NAME_DATA] = self.data[NAME_DATA].astype("category")
        self.threshold = threshold
        self.counter = None

    def calculate(self) -> None:
        pdf = self.data

        # Gets the count of each unique value
        vc = pdf[NAME_DATA].value_counts()
        vc = vc.sort_values()
        # Gets the cumulative count of each unique value
        cumm_frac = np.cumsum(vc.values) / len(pdf)
        print(cumm_frac)
        print(vc.values)
        # Finds the values with low cumulative counts
        num_rare_vals = np.where(cumm_frac < self.threshold)[0].max()
        cut_off = vc.values[::-1][num_rare_vals]
        #min_count = vc[cut_off]
        print(num_rare_vals)
        #logger.success(
        #    f"IQR Outliers gefunden threshold={self.threshold}: {len(self.outliers)}/{len(self.data)}"
        #)

        fix, ax = plt.subplots(figsize=(10, 2))
        s = sns.barplot(x=vc.index, y=vc.values, order=vc.index, color="blue")
        s.axvline(len(vc), -num_rare_vals - 0.5)
        plt.show()

    def diagram(self) -> None:
        pass


if __name__ == "__main__":
    import outlier_detection.files as f
    from pathlib import Path

    path = Path(f.__file__).parent.joinpath("SpeedDating.csv").absolute()
    print(path)
    d = pd.read_csv(path, header=0)
    odc = OneDimCategorial(d["age"].to_list())
    odc.calculate()
    print(d)
