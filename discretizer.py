from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class Discretizer(KBinsDiscretizer):

    def __init__(self, n_bins=5):
        super().__init__(n_bins, encode='ordinal', strategy='uniform')

    def _new_stat_column(self):
        return [0 for _ in range(self.n_bins)]

    def transform_and_stat(self, x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        transformed: np.ndarray = super().transform(x)
        stat_data = {column: self._new_stat_column() for column in x.columns}
        for row in transformed:
            for bin_number, column_name in zip(row, x.columns):
                stat_data[column_name][bin_number] += 1

        return pd.DataFrame(transformed), pd.DataFrame(stat_data)
