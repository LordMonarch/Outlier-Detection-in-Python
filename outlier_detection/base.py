from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from outlier_detection.columns import NAME_DATA, NAME_IS_OUTLIER, NAME_OTHER


class Detection(ABC):
    def __init__(self, data: list[Any]):
        if issubclass(type(data), list):
            self.data = pd.DataFrame({NAME_DATA: data})

        else:
            raise TypeError(
                f"Daten müssen als Liste übergeben werden. Aktuell: {type(data)}"
            )

    @property
    def outliers(self) -> pd.Series:
        if NAME_IS_OUTLIER not in self.data.columns:
            raise ValueError("Die Detection wurde noch nicht gestartet!")

        filtered = self.data[self.data[NAME_IS_OUTLIER]]
        return filtered[NAME_DATA]

    @property
    def without_outliers(self) -> pd.Series:
        print(self.data.columns)
        if NAME_IS_OUTLIER not in self.data.columns:
            raise ValueError("Die Detection wurde noch nicht gestartet!")

        filtered = self.data[~self.data[NAME_IS_OUTLIER]]
        return filtered[NAME_DATA]

    @abstractmethod
    def diagram(self) -> None: ...


class DetectionFullDataFrame(Detection):
    @abstractmethod
    def diagram(self) -> None: ...

    def __init__(self, data: pd.DataFrame):
        if issubclass(type(data), pd.DataFrame):
            super().__init__([])
            self.data = data
        else:
            raise TypeError(
                f"Daten müssen als DataFrame übergeben werden. Aktuell: {type(data)}"
            )

    @property
    def outliers(self) -> pd.DataFrame:
        if NAME_IS_OUTLIER not in self.data.columns:
            raise ValueError("Die Detection wurde noch nicht gestartet!")

        filtered = self.data[self.data[NAME_IS_OUTLIER]]
        return filtered

    @property
    def without_outliers(self) -> pd.DataFrame:
        print(self.data.columns)
        if NAME_IS_OUTLIER not in self.data.columns:
            raise ValueError("Die Detection wurde noch nicht gestartet!")

        filtered = self.data[~self.data[NAME_IS_OUTLIER]]
        return filtered


class DetectionPaired(Detection):
    def __init__(self, data: list[Any], other_data: list[Any]):
        if issubclass(type(data), list) and issubclass(type(other_data), list):
            super().__init__([])
            self.data = pd.DataFrame({NAME_DATA: data, NAME_OTHER: other_data})

        else:
            raise TypeError(
                f"Daten müssen als Liste übergeben werden. Aktuell: data={type(data)} und other_data={type(other_data)}"
            )

    @abstractmethod
    def diagram(self) -> None: ...
