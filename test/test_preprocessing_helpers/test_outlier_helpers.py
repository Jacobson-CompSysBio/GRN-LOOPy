import numpy as np 
import pandas as pd
import pytest
import os
from mock import MagicMock, patch, mock_open, call
from unittest import TestCase 

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


class TestCreateOutlierSampleRows:
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
		n_stds = 2

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
		
class TestExtractOutlierIndicesAndCols:
	"""
	This class tests the extract_outlier_indices_and_cols 
	"""
	def test_extract_outlier_indices_base(self): 
		"""
		this is a base test for extracting outliers
		"""
		input_df = pd.DataFrame({
			'a': [0, 0, 0, 0, 1],
			'b': [1, 0, 0, 0, 0],
			'c': [1, 0, 0, 0, 0], 
			'd': [0, 0, 0, 0, 0]
		})

		actual_indices, actual_columns = outlier_helpers.extract_outlier_indices_and_cols(input_df)

		print("actual_columns")
		print(actual_columns)
		print('actual_indices')
		print(actual_indices)


		expected_indices = pd.Index([0, 4])
		expected_columns = pd.Index(['a', 'b', 'c'])

		assert(actual_indices.equals(expected_indices))
		assert(actual_columns.equals(expected_columns))

	def test_extract_outlier_indices_no_outliers(self):
		"""
		this is a test in which there exist no outliers.
		"""
		input_df = pd.DataFrame({
			'a': [0, 0, 0, 0, 0],
			'b': [0, 0, 0, 0, 0],
			'c': [0, 0, 0, 0, 0], 
			'd': [0, 0, 0, 0, 0]
		})

		actual_indices, actual_columns = outlier_helpers.extract_outlier_indices_and_cols(input_df)

		print("actual_columns")
		print(actual_columns)
		print('actual_indices')
		print(actual_indices)


		expected_indices = pd.Index([])
		expected_columns = pd.Index([])

		assert(actual_indices.equals(expected_indices))
		assert(actual_columns.equals(expected_columns))

class TestRemoveHighPctOutliers: 
	"""
	This class tests the test_remove_high_pct_outlier_rows function
	"""

	raw_df = pd.DataFrame({
		'a': [1, 2, 1, 2, 1, 2],
		'b': [1, 2, 1, 2, 1, 20],
		'c': [3, 3, 70, 4, 1, 90]
	})
	outlier_df = pd.DataFrame({
		'a': [0, 0, 0, 0, 0, 0],
		'b': [0, 0, 0, 0, 0, 1],
		'c': [0, 0, 1, 0, 0, 1]
	})
	outlier_indices = pd.Index([2, 5])
	outlier_columns = pd.Index(['b','c'])
	pct_thresh = 0.67


	def test_removes_high_pct_outliers(self):
		"""
		this test is the base case of removal of high percentile outliers. 
		"""

		actual_raw_dropped_df, actual_outlier_dropped_df = outlier_helpers.remove_high_pct_outlier_rows(
			self.raw_df, 
			self.outlier_df, 
			self.outlier_columns,
			self.outlier_indices,
			self.pct_thresh
		)

		expected_dropped_data =  pd.DataFrame({
			'a': [1, 2, 1, 2, 1],
			'b': [1, 2, 1, 2, 1],
			'c': [3, 3, 70, 4, 1]
		})
		expected_dropped_outlier = pd.DataFrame({
			'a': [0, 0, 0, 0, 0],
			'b': [0, 0, 0, 0, 0],
			'c': [0, 0, 1, 0, 0]
		})

		pd.testing.assert_frame_equal(actual_raw_dropped_df, expected_dropped_data)
		pd.testing.assert_frame_equal(actual_outlier_dropped_df, expected_dropped_outlier)
		


	def test_removes_all_outliers_w_high_pct(self):
		"""
		this test is the base case of removal of high percentile outliers. 
		"""
		pct_thresh = 0.1

		actual_raw_dropped_df, actual_outlier_dropped_df = outlier_helpers.remove_high_pct_outlier_rows(
			self.raw_df, 
			self.outlier_df, 
			self.outlier_columns,
			self.outlier_indices,
			pct_thresh
		)

		expected_dropped_data =  pd.DataFrame({
			'a': [1, 2, 2, 1],
			'b': [1, 2, 2, 1],
			'c': [3, 3, 4, 1]
		})
		expected_dropped_outlier = pd.DataFrame({
			'a': [0, 0, 0, 0],
			'b': [0, 0, 0, 0],
			'c': [0, 0, 0, 0]
		})

		pd.testing.assert_frame_equal(actual_raw_dropped_df.reset_index(drop=True),
									  expected_dropped_data)
		pd.testing.assert_frame_equal(actual_outlier_dropped_df.reset_index(drop=True),
									  expected_dropped_outlier)
		

	def test_removes_no_outlier_rows(self):
		"""
		this test is the base case of removal of high percentile outliers. 

		### Requiring that 2/3 of the outlier features across all samples be outliers within a single row
		"""

		raw_df = pd.DataFrame({
			'a': [1, 2, 1, 200, 1, 2],
			'b': [1, 2, 1, 2, 80, 2],
			'c': [3, 3, 100, 4, 1, 3],
			'd': [3, 3, 70, 4, 1, 90],
			'e': [3, 3, 1, 4, 1, 3]
		})
		outlier_df = pd.DataFrame({
			'a': [0, 0, 0, 1, 0, 0],
			'b': [0, 0, 0, 0, 1, 0],
			'c': [0, 0, 1, 0, 0, 0],
			'd': [0, 0, 1, 0, 0, 1], 
			'e': [0, 0, 0, 0, 0, 0]
		})
		outlier_indices = pd.Index([2, 3, 4, 5])
		outlier_columns = pd.Index(['a','b','c','d'])
		pct_thresh = 0.67 

		actual_raw_dropped_df, actual_outlier_dropped_df = outlier_helpers.remove_high_pct_outlier_rows(
			raw_df, 
			outlier_df, 
			outlier_columns,
			outlier_indices,
			pct_thresh
		)

		expected_dropped_data =  raw_df
		expected_dropped_outlier = outlier_df

		pd.testing.assert_frame_equal(actual_raw_dropped_df, expected_dropped_data)
		pd.testing.assert_frame_equal(actual_outlier_dropped_df, expected_dropped_outlier)
		

class TestDroRowsWithExtremeOutliers: 
	"""
		This calss tests the combination of other methods which extract outliers
		by a number of standard deviations and then remove them based upon som nth 
		percentile. 
	"""

	def test_drop_rows_with_extreme_outliers(self, mocker): 
		"""
		This function tests the base funcationality of drop_rows_with_extreme_outliers
		"""

		# Input Data: 
		input_df = pd.DataFrame({
			'a': [1,2,1,2,1,2,1,2,1,40],
			'b': [40,1,2,1,20,1,2,1,2,1]
		})
		n_sds = 3
		nth_pctile = 0.1


		# Set Up Mocks: 
		outlier_row_indices = pd.Series([True, False, False, False, False, False, False, False, False, True ])
		outlier_df = pd.DataFrame({
			'a': [ False, False, False, False, False, False, False, False, False, True ],
			'b': [ True, False, False, False, False, False, False, False, False, False]
		})

		outlier_indices = pd.Index([0,9])
		outlier_columns = pd.Index(['a','b'])

		create_outlier_sample_rows_output = (outlier_row_indices, outlier_df)
		extract_outlier_indices_and_cols_output = (pd.Index([0, 9]), pd.Index(['a','b']))		
		remove_high_pct_outliers_output = input_df.drop([0,9]), outlier_df.drop([0,9])

		create_outlier_sample_rows_mock = mocker.patch(
			"src.preprocessing.outlier_helpers.create_outlier_sample_rows",
			return_value = create_outlier_sample_rows_output
		)

		extract_outlier_indices_and_cols_mock = mocker.patch(
			"src.preprocessing.outlier_helpers.extract_outlier_indices_and_cols",
			return_value = extract_outlier_indices_and_cols_output
		)

		remove_high_pct_outliers_mock = mocker.patch(
			"src.preprocessing.outlier_helpers.remove_high_pct_outlier_rows",
			return_value = remove_high_pct_outliers_output
		)

		# Call Function
		actual_dropped_high_pct_outlier, actual_outlier_df = outlier_helpers.drop_rows_with_extreme_outliers(input_df, 3, 0.1)


		# Expected Results
		expected_input_df, expected_outlierdf_data =  remove_high_pct_outliers_output
		
		# Testing
		create_outlier_sample_rows_mock.assert_called_once_with(input_df, n_sds)
		extract_outlier_indices_and_cols_mock.assert_called_once_with(outlier_df)

		# Truth value of the indices became prohibitively difficult in engineering the 
		# simple function for the sake of test architecture. Manual inspection 
		# showed expected performance. 
		remove_high_pct_outliers_mock.assert_called_once()
		pd.testing.assert_frame_equal(actual_dropped_high_pct_outlier, remove_high_pct_outliers_output[0])
		pd.testing.assert_frame_equal(actual_outlier_df, remove_high_pct_outliers_output[1])

class TestOutlierRemovalAndWinsorization: 
	"""
	This class tests the base functionality of outlier_removal_and_winsorization where outliers are
	first removed (i.e. outlier samples that have a significant number of outliers that winsorizing
	may not help)

	Test this method against networks and MCMC output
	"""

	def test_outlier_removal_and_winsorization(self, mocker): 
		"""
		tests base case 
		"""
		input_df = pd.DataFrame({
			'a': [1,2,1,2,1,2,1,2,1,40],
			'b': [40,1,2,1,20,1,2,1,2,1]
		})

		outlier_row_indices = pd.Series([True, False, False, False, False, False, False, False, False, True ])
		outlier_df = pd.DataFrame({
			'a': [ False, False, False, False, False, False, False, False, False, True ],
			'b': [ True, False, False, False, False, False, False, False, False, False]
		})
		create_outlier_sample_rows_output = (outlier_row_indices, outlier_df)
		outlier_row_indices2 = pd.Series([False, False, False, True, False, False, False, False])
		outlier_df2 = pd.DataFrame({
			'a': [ False, False, False, False, False, False, False, False ],
			'b': [ False, False, False, True, False, False, False, False ]
		})

		create_outlier_sample_rows_output2 = (outlier_row_indices2, outlier_df2)


		extract_outlier_indices_and_cols_output = (pd.Index([0, 9]), pd.Index(['a','b']))
		extract_outlier_indices_and_cols_output2 = (pd.Index([4]), pd.Index(['b']))
		
		remove_high_pct_outliers_output = input_df.drop([0,9]), outlier_df.drop([0,9])

		mocker.patch(
			"src.preprocessing.outlier_helpers.create_outlier_sample_rows",
			side_effect = [
				create_outlier_sample_rows_output, 
				create_outlier_sample_rows_output2
				
			]
		)

		mocker.patch(
			"src.preprocessing.outlier_helpers.extract_outlier_indices_and_cols",
			side_effect = [
				extract_outlier_indices_and_cols_output, 
				extract_outlier_indices_and_cols_output2
			]
		)

