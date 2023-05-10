import numpy as np 
import pandas as pd
import pytest
import os
from mock import MagicMock, patch, mock_open, call

from src.preprocessing import outlier_helpers


class TestExtractOutlierSamples: 
	"""
	This class tests the remove outliers function
	"""
	def test_removes_outliers_with_more_than_3sds(self): 
		"""
		This function returns true for outliers greater than 3 stds
		"""
		input_series = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 6, 12, 36])
		n_sds = 3

		actual_output = outlier_helpers.extract_outlier_samples(input_series, n_sds)

		expected_output = pd.Series([False, False, False, False, False, False,
									 False, False, False, False, False, True])

		pd.testing.assert_series_equal(actual_output, expected_output)


	def test_removes_nothing_when_no_outputliers(self): 
		"""
		This function removes no outliers because there are none! 
		"""
		input_series = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
		n_sds = 3

		actual_output = outlier_helpers.extract_outlier_samples(input_series, n_sds)

		expected_output = pd.Series([False, False, False, False, False, False,
									False, False, False, False, False, False])

		pd.testing.assert_series_equal(actual_output, expected_output)

class TestRemoveOutliers:
	"""
	this class removes outliers for each column of a dataframe. 
	"""
	def test_removes_outliers_in_dataframe(self, mocker):
		"""
		This function tests whether the input df will properly find the 
		rows with outliers. 
		"""
		input_df = pd.DataFrame({
			'a': [1,2,1,2,1,2,1,2,1,40],
			'b': [40,1,2,1,2,1,2,1,2,1]
		})
		n_stds = 3

		mocker.patch(
			"src.preprocessing.outlier_helpers.extract_outlier_samples",
			side_effect = [
				pd.Series([
			 		False, False, False, False, False, False, False, False, False, True
				]),
				pd.Series([
					True, False, False, False, False, False, False, False, False, False
				])
			]
		)
		
		actual_series, actual_df = outlier_helpers.create_outlier_sample_rows(input_df, n_stds)
		print("WHAT")
		print(type(actual_series))
		print(type(actual_df))
		print(actual_series, actual_df)
		expected_series_output = pd.Series([
			True,
			False,
			False,
			False,
			False,
			False,
			False,
			False,
			False,
			True,
		])
		expected_dataframe_output = pd.DataFrame({
			'a': [
				False, False, False, False, False, False, False, False, False, True
			],
			'b': [
				True, False, False, False, False, False, False, False, False, False
			]
		})

		pd.testing.assert_series_equal(actual_series, expected_series_output)
		pd.testing.assert_frame_equal(actual_df, expected_dataframe_output)
		
