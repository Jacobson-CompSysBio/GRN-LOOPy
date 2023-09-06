import numpy as np 
import pandas as pd
import pytest
import os
from mock import mock_open


from src.preprocessing import correlation_helpers

base_data = {
					"a": [0, 1, 1, 0],
					"b": [1, 0, 0, 1],
					"c": [0, 1, 1, 1]
				}
base_dataframe = pd.DataFrame(data= base_data)
threshold = 0.95

correlation_matrix_data = {
	"a": [	1,	   -1,  0.5773],
	"b": [   -1,		1, -0.5773],
	"c": [0.5773, -0.5773,	   1]
}
correlation_df = pd.DataFrame(correlation_matrix_data)
correlation_df.index = correlation_df.columns

upper_right_corr_df_data = {
			"a": [   0.0,	 0.0, 0.0],
			"b": [  -1.0,	 0.0, 0.0],
			"c": [0.5773, -0.5773, 0.0]
		}
upper_right_corr_df = pd.DataFrame(upper_right_corr_df_data)
upper_right_corr_df.index = upper_right_corr_df.columns

stacked_corr_df = pd.DataFrame({
	'from': ['a'],
	'to':   ['b'],
	'corr': [ -1.0]
})

class TestCorrelateData: 
	"""
	This class Testis a test suite for the Correlate_data function.
	"""

	def test_correlate_data_base(self): 
		"""
		This is a base test of def correlate_data. 
		"""

		input_df = base_dataframe

		actual_output = correlation_helpers.correlate_data(input_df, has_index_col=False)

		expected_output = correlation_df

		pd.testing.assert_frame_equal(actual_output, expected_output, atol=1e-4)
	
	def test_correlate_data_with_index_col(self): 
		"""
		This is a base test of def correlate_data. 
		"""

		input_df = base_dataframe
		input_df['index'] = list(input_df.index)
		input_df = input_df[ ['index', 'a', 'b', 'c'] ]

		actual_output = correlation_helpers.correlate_data(input_df, has_index_col=True)
		expected_output = correlation_df

		pd.testing.assert_frame_equal(actual_output, expected_output, atol=1e-4)


class TestExtractCorrelatesToUpperRight: 
	"""
	This class Testis a test suite for the Extract_correlates_to_upper_right function.
	"""

	def test_extract_correlates_to_upper_right_base(self): 
		"""
		This is a base test of def extract_correlates_to_upper_right. 
		it turns the upper right triangle of a correlation without the diagonal

		"""

		input_df = correlation_df

		actual_output = correlation_helpers.extract_correlates_to_upper_right(input_df)

		expected_output = upper_right_corr_df

		pd.testing.assert_frame_equal(actual_output, expected_output, atol=1e-4)



class TestStackData: 
	"""
	This class Testis a test suite for the Stack_data function.
	"""

	def test_stack_data_base(self): 
		"""
		This is a base test of def stack_data. 

		stack data takes a dataframe and a threshold at which all corrlated
		values are extracted to then be saved to file.
		"""
		
		input_df = upper_right_corr_df
		input_thresh = threshold

		actual_output = correlation_helpers.stack_data(input_df, input_thresh)

		expected_output = stacked_corr_df

		pd.testing.assert_frame_equal(actual_output, expected_output)


class TestCreateCorrelationList: 
	"""
	This class Testis a test suite for the Create_correlation_list function.
	"""

	def test_create_correlation_list_base(self, mocker): 
		"""
		This is a base test of def create_correlation_list. 

		This function reads in a file, and then calls
			- correlate data
			- extract_correlates_to_upper_right
			- stack_data

		The function then generates the outfile name and returns the dataframe. 
		"""

		file_name = "./test/test_data/test_input_networks/test_net_noidx.tsv"
		has_indices = False
		corr_thresh = threshold
		save_corr = False

		correlate_mock = mocker.patch(
			"src.preprocessing.correlation_helpers.correlate_data",
			return_value = correlation_df
		)
		extraction_mock = mocker.patch(
			"src.preprocessing.correlation_helpers.extract_correlates_to_upper_right",
			return_value = upper_right_corr_df
		)
		stack_mock  = mocker.patch(
			"src.preprocessing.correlation_helpers.stack_data",
			return_value = stacked_corr_df
		)
		to_csv_mock = mocker.patch.object(stacked_corr_df, "to_csv")

		actual_output = correlation_helpers.create_correlation_list(file_name, has_indices, corr_thresh, save_corr)

		expected_corr_call_object = pd.read_csv(file_name, sep='\t')
		expected_output = stacked_corr_df
		expected_output_save_file = './test/test_data/test_input_networks/test_net_noidx_correlation_over_0.95.tsv'

		## pytest does not like asserts with dfs
		pd.testing.assert_frame_equal(actual_output, expected_output)

		pd.testing.assert_frame_equal(expected_corr_call_object, correlate_mock.call_args[0][0])
		assert(has_indices == correlate_mock.call_args[0][1])

		pd.testing.assert_frame_equal(correlation_df, extraction_mock.call_args[0][0])

		pd.testing.assert_frame_equal(upper_right_corr_df, stack_mock.call_args[0][0])
		assert(corr_thresh == stack_mock.call_args[0][1])

		to_csv_mock.assert_not_called()


	def test_create_correlation_list_base_save_corr(self, mocker): 
		"""
		This is a base test of def create_correlation_list. 

		This function reads in a file, and then calls
			- correlate data
			- extract_correlates_to_upper_right
			- stack_data

		The function then generates the outfile name and returns the dataframe. 
		"""

		file_name = "./test/test_data/test_input_networks/test_net_noidx.tsv"
		has_indices = False
		corr_thresh = threshold
		save_corr = True

		correlate_mock = mocker.patch(
			"src.preprocessing.correlation_helpers.correlate_data",
			return_value = correlation_df
		)
		extraction_mock = mocker.patch(
			"src.preprocessing.correlation_helpers.extract_correlates_to_upper_right",
			return_value = upper_right_corr_df
		)
		stack_mock  = mocker.patch(
			"src.preprocessing.correlation_helpers.stack_data",
			return_value = stacked_corr_df
		)
		to_csv_mock = mocker.patch.object(stacked_corr_df, "to_csv")

		actual_output = correlation_helpers.create_correlation_list(file_name, has_indices, corr_thresh, save_corr)

		expected_corr_call_object = pd.read_csv(file_name, sep='\t')
		expected_output = stacked_corr_df
		expected_output_save_file = './test/test_data/test_input_networks/test_net_noidx_correlation_over_0.95.tsv'

		## pytest does not like asserts with dfs
		pd.testing.assert_frame_equal(actual_output, expected_output)

		pd.testing.assert_frame_equal(expected_corr_call_object, correlate_mock.call_args[0][0])
		assert(has_indices == correlate_mock.call_args[0][1])

		pd.testing.assert_frame_equal(correlation_df, extraction_mock.call_args[0][0])

		pd.testing.assert_frame_equal(upper_right_corr_df, stack_mock.call_args[0][0])
		assert(corr_thresh == stack_mock.call_args[0][1])

		to_csv_mock.assert_called_once_with(expected_output_save_file, sep='\t', index=None)


