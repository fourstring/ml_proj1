from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from discretizer import Discretizer
from splitter import AbstractDataSplitter


class AbstractClassifier(ABC):
    train_set: pd.DataFrame
    valid_set: pd.DataFrame
    test_set: pd.DataFrame
    train_labels: pd.Series
    valid_labels: pd.Series
    test_labels: pd.Series
    classes: List[int]
    name: str
    data_source: AbstractDataSplitter
    scaler: MinMaxScaler
    _prepared: False

    def __init__(self, name: str, data_source: AbstractDataSplitter):
        self.name = name
        self.data_source = data_source
        self.data_source.preprocessor = self.preprocessing

    def _prepared_hook(self):
        assert self._prepared

    @abstractmethod
    def labeling(self, data: pd.DataFrame) -> pd.Series:
        pass

    def normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaler.fit(data)
        self.scaler = scaler
        return pd.DataFrame(scaler.transform(data))

    @abstractmethod
    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predicate(self, data_vector: pd.Series, *args, **kwargs) -> np.float64:
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def classify(self, to_classify: pd.DataFrame, *args, **kwargs):
        pass

    @abstractmethod
    def report(self):
        pass


class AbstractNaiveBayesClassifier(AbstractClassifier, ABC):
    @dataclass
    class NaiveBayesParam:
        count: int = 0
        class_prob: np.float64 = 0.0
        features_prob: Optional[pd.DataFrame] = None

    params: Dict[int, NaiveBayesParam]
    discretizer: Discretizer

    def __init__(self, name: str, data_source: AbstractDataSplitter):
        super().__init__(name, data_source)
        self.params = {}
        self.discretizer = Discretizer(n_bins=10)

    def prepare(self):
        self.train_set, self.valid_set, self.test_set = self.data_source.split(0.8, 0.1, 0.1)
        self.train_set = pd.concat([self.train_set, self.valid_set])
        self.train_labels = self.train_set.pop('label')
        self.test_labels = self.test_set.pop('label')
        self._prepared = True

    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = self.labeling(data)
        normalized = self.normalization(data)
        ret = pd.DataFrame(self.discretizer.fit_transform(normalized))
        ret['label'] = labels
        return ret

    def train(self, *args, **kwargs):
        self._prepared_hook()
        feature_counters = {}
        for c in self.classes:
            self.params[c] = self.NaiveBayesParam()
            feature_counters[c] = {column_name: [0 for _ in range(self.discretizer.n_bins)] for column_name in
                                   self.train_set.columns}
        for (_, row), label in zip(self.train_set.iterrows(), self.train_labels):
            self.params[label].count += 1
            for bin_number, column_name in zip(row, self.train_set.columns):
                feature_counters[label][column_name][int(bin_number)] += 1

        total = self.train_set.shape[0]
        for c in self.classes:
            param = self.params[c]
            param.class_prob = param.count / total
            class_feature_params = feature_counters[c]
            for column_name in class_feature_params:
                class_feature_params[column_name] = list(
                    map(lambda x: x / param.count, class_feature_params[column_name]))
            param.features_prob = pd.DataFrame(class_feature_params)

    def predicate(self, data_vector: pd.Series, *args, **kwargs) -> Tuple[np.float64, int]:
        max_prob = np.float64(0.0)
        max_prob_class = -1
        discretized_vec = pd.Series(self.discretizer.transform(data_vector.values.reshape(1, -1))[0],
                                    index=data_vector.index)
        for c in self.classes:
            param = self.params[c]
            prob = np.float64(param.class_prob)
            features_prob = param.features_prob
            for column_name in features_prob.columns:
                prob *= features_prob[column_name][int(discretized_vec[column_name])]
            if prob > max_prob:
                max_prob = prob
                max_prob_class = c

        return max_prob, max_prob_class

    def test(self, *args, **kwargs) -> pd.Series:
        self._prepared_hook()
        return self.classify(self.test_set)

    def classify(self, to_classify: pd.DataFrame, *args, **kwargs) -> pd.Series:
        self._prepared_hook()
        return pd.Series(self.predicate(row) for _, row in to_classify.iterrows())

    def report(self):
        test_result = self.test()
        stat_counters = [
            [0, 0],
            [0, 0]
        ]
        for (_, predicate_label), test_label in zip(test_result, self.test_labels):
            stat_counters[predicate_label][test_label] += 1
        total = self.test_set.shape[0]
        tp = stat_counters[1][1]
        tn = stat_counters[0][0]
        fp = stat_counters[1][0]
        fn = stat_counters[0][1]
        acc = (tp + tn) / (tp + tn + fp + fn) * 100
        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        f1 = 2 / (1 / precision + 1 / recall)
        tpr = tp / total * 100
        tnr = tn / total * 100
        fpr = fp / total * 100
        fnr = fn / total * 100
        return pd.Series(
            {'Name': self.name, 'TP Rate': tpr, 'FP Rate': fpr, 'TN Rate': tnr, 'FN Rate': fnr, 'Accuracy': acc,
             'Precision': precision, 'Recall': recall, 'F1': f1})


class MobileDataSetMixin:
    def labeling(self, data: pd.DataFrame) -> pd.Series:
        labels = data['price_range'].apply(lambda x: 0 if x in [0, 1] else 1)
        self.classes = [0, 1]
        data.pop('price_range')
        return labels


class MobileNaiveBayes(MobileDataSetMixin, AbstractNaiveBayesClassifier):
    pass


class AbstractGradientDescentModel(AbstractClassifier, ABC):
    params: np.ndarray
    learning_rate: float
    movement_rate: float

    @dataclass
    class TrainReport:
        batch_size: int
        epoch: int
        train_loss: List[float] = field(default_factory=lambda: [])
        valid_loss: List[float] = field(default_factory=lambda: [])

    train_report: TrainReport

    def __init__(self, name: str, learning_rate: float, data_source: AbstractDataSplitter, movement_rate: float = 0.1):
        super().__init__(name, data_source)
        self.learning_rate = learning_rate
        self.movement_rate = movement_rate

    @abstractmethod
    def gradient(self, data: pd.DataFrame, data_labels: pd.Series, cur_params: np.ndarray) -> np.ndarray:
        pass

    def descent(self, data: pd.DataFrame, data_labels: pd.Series, cur_params: np.ndarray,
                last_movement: np.float64) -> Tuple[np.ndarray, np.float64]:
        movement = -self.gradient(data, data_labels,
                                  cur_params) * self.learning_rate - self.movement_rate * last_movement
        return cur_params + movement, movement

    @abstractmethod
    def loss(self, data: pd.DataFrame, data_labels: pd.Series, cur_params: np.ndarray) -> np.float64:
        pass

    @abstractmethod
    def model(self, data_vector: pd.Series, params: np.ndarray) -> np.float64:
        pass

    def split_df(self, df: pd.DataFrame, batch_size: int):
        return np.array_split(df, int((df.shape[0] + batch_size - 1) / batch_size))

    def split_1d(self, ary, batch_size: int):
        return np.array_split(ary, int((len(ary) + batch_size - 1) / batch_size))

    def train(self, batch_size: int = 100, valid_loss_threshold: Optional[float] = None,
              max_epoch: Optional[int] = None,
              *args, **kwargs) -> TrainReport:
        threshold_enabled = valid_loss_threshold is not None
        epoch_enabled = max_epoch is not None
        assert threshold_enabled or epoch_enabled, 'You must specify one of ' \
                                                   'valid_loss_threshold and max_epoch at least '
        cur_params = self.params.copy()
        report = self.TrainReport(batch_size, 0)
        while True:
            report.epoch += 1
            epoch_train_loss = 0.0
            epoch_valid_loss = 0.0
            last_movement = np.float64(0.0)
            for batch, batch_label in zip(self.split_df(self.train_set, batch_size),
                                          self.split_1d(self.train_labels, batch_size)):
                cur_params, last_movement = self.descent(batch, batch_label, cur_params, last_movement)
                epoch_train_loss += self.loss(batch, batch_label, cur_params)
            for valid_batch, valid_batch_label in zip(self.split_df(self.valid_set, batch_size),
                                                      self.split_1d(self.valid_labels, batch_size)):
                epoch_valid_loss += self.loss(valid_batch, valid_batch_label, cur_params)
            epoch_train_loss = epoch_train_loss / self.train_set.shape[0]
            epoch_valid_loss = epoch_valid_loss / self.valid_set.shape[0]
            report.train_loss.append(epoch_train_loss)
            report.valid_loss.append(epoch_valid_loss)
            print(f'Epoch: {report.epoch},Train Loss: {epoch_train_loss:.2f},Valid loss: {epoch_valid_loss:.2f}')
            if threshold_enabled and epoch_valid_loss <= valid_loss_threshold:
                self.train_report = report
                self.params = cur_params
                return report
            if epoch_enabled and report.epoch == max_epoch:
                self.train_report = report
                self.params = cur_params
                return report

    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = self.labeling(data)
        normalized = self.normalization(data)
        normalized['label'] = labels
        normalized.insert(len(normalized.columns), len(normalized.columns) - 1, 1)
        return normalized

    def prepare(self):
        self.train_set, self.valid_set, self.test_set = self.data_source.split(0.8, 0.1, 0.1)
        self.train_labels = self.train_set.pop('label')
        self.valid_labels = self.valid_set.pop('label')
        self.test_labels = self.test_set.pop('label')
        self.params = np.random.randn(len(self.train_set.columns))
        self._prepared = True

    def predicate(self, data_vector: pd.Series, threshold: float = 0.5, *args, **kwargs) -> Tuple[np.float64, int]:
        prob = self.model(data_vector, self.params)
        return prob, 1 if prob >= threshold else 0

    def test(self, threshold: float = 0.5, *args, **kwargs) -> pd.Series:
        return self.classify(self.test_set, threshold)

    def classify(self, to_classify: pd.DataFrame, threshold: float = 0.5, *args, **kwargs) -> pd.Series:
        return pd.Series(self.predicate(row, threshold) for _, row in to_classify.iterrows())

    def report(self):
        test_result = [prob for prob, _ in self.test()]
        fpr, tpr, threshold = roc_curve(self.test_labels, test_result)
        auc = roc_auc_score(self.test_labels, test_result)
        plt.figure()
        plt.subplot(211)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC of {self.name}')
        plt.legend(loc="lower right")

        plt.subplot(212)
        epoch = np.arange(1, self.train_report.epoch + 1)
        plt.plot(epoch, self.train_report.train_loss, color='darkorange',
                 lw=lw, label='Loss on train set')
        plt.plot(epoch, self.train_report.valid_loss, color='navy',
                 lw=lw, label='Loss on valid set')
        plt.xlim([0.0, self.train_report.epoch + 1])
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title(f'Loss curve of {self.name}')
        plt.legend(loc="lower right")
        plt.show()


class MobileLogisticRegression(MobileDataSetMixin, AbstractGradientDescentModel):
    def gradient(self, data: pd.DataFrame, data_labels: pd.Series, cur_params: np.ndarray) -> np.ndarray:
        grad_sum = sum(
            np.dot(X.T, self.logit(X, cur_params) - y) for (_, X), y in zip(data.iterrows(), data_labels))
        return grad_sum

    def loss(self, data: pd.DataFrame, data_labels: pd.Series, cur_params: np.ndarray) -> float:
        loss_sum = np.float64(0.0)
        for (_, X), y in zip(data.iterrows(), data_labels):
            logit = self.logit(X, cur_params)
            loss_sum += -y * np.log(logit) - (1 - y) * np.log(1 - logit)
        return loss_sum

    def model(self, data_vector: pd.Series, params: np.ndarray) -> np.float64:
        return self.logit(data_vector, params)

    def logit(self, data_vector: pd.Series, params: np.ndarray) -> np.float64:
        return 1 / (1 + np.exp(-np.dot(data_vector.values.T, params)))


class AbstractSVMClassifier(AbstractClassifier, ABC):
    svm: LinearSVC

    def __init__(self, name: str, data_source: AbstractDataSplitter):
        super().__init__(name, data_source)
        self.svm = LinearSVC()

    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = self.labeling(data)
        normalized = self.normalization(data)
        normalized['label'] = labels
        normalized.insert(len(normalized.columns), len(normalized.columns) - 1, 1)
        return normalized

    def prepare(self):
        self.train_set, self.valid_set, self.test_set = self.data_source.split(0.8, 0.1, 0.1)
        self.train_set = pd.concat([self.train_set, self.valid_set])
        self.train_labels = self.train_set.pop('label')
        self.test_labels = self.test_set.pop('label')
        self._prepared = True

    def train(self, *args, **kwargs):
        self.svm.fit(self.train_set, self.train_labels)

    def predicate(self, data_vector: pd.Series, *args, **kwargs) -> np.float64:
        pass

    def test(self, *args, **kwargs):
        return self.classify(self.test_set)

    def classify(self, to_classify: pd.DataFrame, *args, **kwargs):
        return self.svm.predict(to_classify)

    def report(self):
        plot_roc_curve(self.svm, self.test_set, self.test_labels)
        plt.show()


class MobileSVM(MobileDataSetMixin, AbstractSVMClassifier):
    pass
