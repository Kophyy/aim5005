from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase
#from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

### DO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return minimum values [-1., 2.] " 
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]])
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]])
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    # TODO: 
    # test for custom StandardScaler
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)  # Add this line to calculate result
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
    
    # test for LabelEncoder
    def test_label_encoder_african_countries(self):
        """
        Custom test: Verify that LabelEncoder correctly maps African countries labels to numeric values.
        Given an input list of labels, np.unique will sort them, so:
          "Kenya" -> 0, "Nigeria" -> 1, "South Africa" -> 2.
        For input ["Nigeria", "South Africa", "Nigeria", "Kenya"], we expect [1, 2, 1, 0].
        """
        encoder = LabelEncoder()
        data = ["Nigeria", "South Africa", "Nigeria", "Kenya"]
        expected = np.array([1, 2, 1, 0])
        result = encoder.fit_transform(data)
        assert (result == expected).all(), f"Label encoder did not produce expected result. Got {result}, expected {expected}"

    def test_label_encoder_african_countries_single_value(self):
        """
        Custom test: Verify that LabelEncoder correctly maps a single African country label to numeric value.
        Given an input list of labels, np.unique will sort them, so:
          "Kenya" -> 0, "Nigeria" -> 1, "South Africa" -> 2.
        For input ["Kenya"], we expect [0].
        """
        encoder = LabelEncoder()
        data = ["Kenya"]
        expected = np.array([0])
        result = encoder.fit_transform(data)
        assert (result == expected).all(), f"Label encoder did not produce expected result. Got {result}, expected {expected}"

if __name__ == '__main__':
    unittest.main()