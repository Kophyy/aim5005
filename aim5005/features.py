import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO:  
        return (x-self.minimum)/diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return.
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Calculate the mean and standard deviation for each feature.
        """
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        # Optional: To avoid division by zero, replace zeros in std with 1.
        self.std[self.std == 0] = 1

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standard Scale the given vector: (x - mean) / std.
        """
        x = self._check_is_array(x)
        return (x - self.mean) / self.std

    def fit_transform(self, x: list) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def _check_is_array(self, y: np.ndarray) -> np.ndarray:
        """
        Try to convert y to a np.ndarray if it's not already one and return.
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        return y

    def fit(self, y: np.ndarray) -> None:
        """
        Fit the label encoder by determining the unique classes in y.
        """
        y = self._check_is_array(y)
        self.classes_ = np.unique(y)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform the input labels into numeric values based on the fitted classes.
        """
        y = self._check_is_array(y)
        label_map = {label: idx for idx, label in enumerate(self.classes_)}
        return np.array([label_map[label] for label in y])

    def fit_transform(self, y: list) -> np.ndarray:
        """
        Fit to data, then transform it into numeric labels.
        """
        y = self._check_is_array(y)
        self.fit(y)
        return self.transform(y)