import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

PreprocessorType = Optional[Callable[[pd.DataFrame], pd.DataFrame]]


class AbstractDataSplitter(ABC):
    preprocessor: PreprocessorType

    def __init__(self, preprocessor: PreprocessorType = None):
        self.preprocessor = preprocessor

    @abstractmethod
    def load_source(self) -> pd.DataFrame:
        raise NotImplementedError('Concrete DataSplitter must implement load_source for data loading')

    def split(self, train_ratio: float, valid_ratio: float, test_ratio: float) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.load_source()
        if self.preprocessor:
            df = self.preprocessor(df)
        total = train_ratio + valid_ratio + test_ratio
        first_round_part1, rest_total = train_ratio / total, total - train_ratio
        train_df, rest_df = train_test_split(df, train_size=first_round_part1)
        second_round_part1 = valid_ratio / rest_total
        valid_df, test_df = train_test_split(rest_df, train_size=second_round_part1)
        return train_df, valid_df, test_df


class FileDataSplitter(AbstractDataSplitter):
    source_filename: str
    train_filename: str
    valid_filename: str
    test_filename: str

    def __init__(self, source, train='train.csv', valid='valid.csv', test='test.csv',
                 preprocessor: PreprocessorType = None):
        super().__init__(preprocessor)
        self.source_filename = source
        self.train_filename = train
        self.valid_filename = valid
        self.test_filename = test

    def load_source(self) -> pd.DataFrame:
        return pd.read_csv(self.source_filename)

    def store(self, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
        if self.source_filename in [self.train_filename, self.valid_filename, self.test_filename]:
            bak_filename = Path(f'{self.source_filename}.bak')
            if bak_filename.is_file():
                os.remove(str(bak_filename))
            os.rename(self.source_filename, f'{self.source_filename}.bak')
        train.to_csv(self.train_filename)
        valid.to_csv(self.valid_filename)
        test.to_csv(self.test_filename)

    def split(self, train_ratio: float, valid_ratio: float, test_ratio: float) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, valid_df, test_df = super().split(train_ratio, valid_ratio, test_ratio)
        self.store(train_df, valid_df, test_df)
        return train_df, valid_df, test_df


class DataFrameSplitter(AbstractDataSplitter):
    source_df: pd.DataFrame

    def __init__(self, source_df: pd.DataFrame, preprocessor: PreprocessorType = None):
        super().__init__(preprocessor)
        self.source_df = source_df

    def load_source(self) -> pd.DataFrame:
        return self.source_df
