
import numpy as np 
import pandas as pd
import pytest
import os
from mock import mock_open
# import lightgbm as lgb

from src.processing import data_helpers


class TestGetTrainTestSplit: 
	"""
	This class Testis a test suite for the get_train_test function.
	"""

	def test_throws_error_when_no_data_supplied(self): 
		with pytest.raises(Exception, match="Both samples and features must be included."):
			data_helpers.get_train_test_split()

	def test_get_train_test_split(self, mocker): 
		"""
		This is a base test of the data heleprs that create a 
		train test split based on either a raw data df or a train
		and test df
		"""

		n_features = 5
		n_samples = 4
		
		#x shaped 3 rows, 5 cols
		x = np.array([
			[1,2,3,4,5],
			[4,5,6,7,8],
			[9,8,7,6,5],
			[9,8,7,6,5]
		])
		y = np.array([6, 7, 8, 9])
		train_test_split_input = pd.DataFrame({
			0: [1, 4, 9, 9],
			1: [2, 5, 8, 8],
			2: [3, 6, 7, 7],
			3: [4, 7, 6, 6],
			4: [5, 8, 5, 5],
			5: [6, 7, 8, 8]
		})

		datasets_mock = mocker.patch(
			"sklearn.datasets.make_regression",
			return_value = (x, y)
		)
		sklearn_mock = mocker.patch(
			"sklearn.model_selection.train_test_split", 
			return_value = (
				train_test_split_input.loc[:1], 
				train_test_split_input.loc[2:]
			)
		)


		actual_output = data_helpers.get_train_test_split(
			n_samples = n_samples,
			n_features = n_features
		)


		expected_df_train_x_mat = train_test_split_input.loc[:1][range(5)]
		expected_df_train_y_vec = train_test_split_input.loc[:1][5]
		expected_df_test_x_mat  = train_test_split_input.loc[2:][range(5)]
		expected_df_test_y_vec  = train_test_split_input.loc[2:][5]


		pd.testing.assert_frame_equal(actual_output[0], expected_df_train_x_mat)
		pd.testing.assert_frame_equal(actual_output[1], expected_df_test_x_mat)
		pd.testing.assert_series_equal(actual_output[2], expected_df_train_y_vec)
		pd.testing.assert_series_equal(actual_output[3], expected_df_test_y_vec)
		datasets_mock.assert_called_once_with(n_samples, n_features)
		sklearn_mock.assert_called_once() # called w/ dataframe. ambiguous in pytest
	

	def test_get_train_test_split_with_dataframe_no_feature(self, mocker): 
		"""
		This is a test of the data heleprs that creates a 
		train test split based on a raw data df
		"""
		input_df = pd.DataFrame({
			0: [1, 4, 9, 9],
			1: [2, 5, 8, 8],
			2: [3, 6, 7, 7],
			3: [4, 7, 6, 6],
			4: [5, 8, 5, 5],
			5: [6, 7, 8, 9]
		})
		feature_name = 5

		sklearn_mock = mocker.patch(
			"sklearn.model_selection.train_test_split", 
			return_value = (
				input_df.loc[:1], 
				input_df.loc[2:]
			)
		)

		actual_output = data_helpers.get_train_test_split(
			df = input_df
		)

		expected_df_train = input_df.loc[:1]
		expected_df_test  = input_df.loc[2:]

		pd.testing.assert_frame_equal(actual_output[0], expected_df_train)
		pd.testing.assert_frame_equal(actual_output[1], expected_df_test)

	def test_fails_get_x_y_split_with_train_test_dfs(self, mocker): 
		"""
		This is a test of the data heleprs that creates a 
		train test split based on a raw data df
		"""
		input_df = pd.DataFrame({
			0: [1, 4, 9, 9],
			1: [2, 5, 8, 8],
			2: [3, 6, 7, 7],
			3: [4, 7, 6, 6],
			4: [5, 8, 5, 5],
			5: [6, 7, 8, 9]
		})

		with pytest.raises(Exception, match="No feature name for dataframes"):
			actual_output = data_helpers.get_train_test_split(
				df_train = input_df.loc[:1],
				df_test = input_df.loc[2:]
		)
