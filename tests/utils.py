import pandas as pd
import pytest

from outlier_detection.utils import abs_diff, mean, median, quantile, std_dev


class TestUtils:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

    def test_abs_diff(self, sample_df):
        df = abs_diff(sample_df.copy(), "A", "B", "diff")
        expected = pd.Series([4, 2, 0, 2, 4], name="diff")
        pd.testing.assert_series_equal(df["diff"], expected)

    def test_mean(self, sample_df):
        df = mean(sample_df.copy(), "A", "mean_A")
        expected_value = sample_df["A"].mean()
        # Prüfe, ob jede Zeile den Mittelwert enthält
        assert all(df["mean_A"] == expected_value)

    def test_median(self, sample_df):
        df = median(sample_df.copy(), "A", "median_A")
        expected_value = sample_df["A"].median()
        assert all(df["median_A"] == expected_value)

    def test_quantile(self, sample_df):
        q25 = quantile(sample_df, "A", 0.25)
        expected_q25 = sample_df["A"].quantile(0.25)
        assert q25 == expected_q25

    def test_std_dev(self, sample_df):
        df = std_dev(sample_df.copy(), "A", "std_A")
        expected_value = sample_df["A"].std()
        assert all(df["std_A"] == expected_value)
