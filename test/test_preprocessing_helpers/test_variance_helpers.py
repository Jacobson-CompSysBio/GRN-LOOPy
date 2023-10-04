import numpy as np 
import pandas as pd
import pytest
import os
from mock import MagicMock, patch, mock_open, call


from src.preprocessing import variance_helpers


data = {
					"a": [0, 1, 2, 3, 2, 1, 3, 0, 3], # sum = 15  
					"b": [0, 0, 0, 0, 0, 0, 0, 0, 0], # sum =  0
					"c": [2, 1, 2, 3, 1, 3, 2, 3, 3], # sum = 20 
					"d": [1, 0, 0, 1, 0, 1, 1, 0, 1]  # sum =  5
				}
dataframe = pd.DataFrame(data= data)
threshold = 0.1

class TestNormalizeData: 
	"""
	This test class tests normalizing data within a dataframe.
	"""

	data = {
		"a": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		"b": [0, 10, 0, 10, 0, 10, 0, 10, 0, 10],
		"c": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
		'd': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		'e': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
	}
	df = pd.DataFrame(data)

	## TODO: write normalization tests. 

class TestRemoveLowVariance:
	"""
	This class tests the remove low variance function
	""" 



	def test_remove_low_variance_features(self): 
		"""
		This function tests 'remove_low_var' with ideal parameters.
		"""
		

		expected_output_columns = ['a','c','d']
		expected_output = dataframe[ expected_output_columns ]

		actual_output = variance_helpers.remove_low_variance_features(dataframe, threshold)

		pd.testing.assert_frame_equal(expected_output, actual_output)
		
	def test_remove_low_variance_features_savemeta(self, mocker): 
		"""
		This function tests 'remove_low_var' with ideal parameters.
		"""
	
		expected_output_columns = ['a','c','d']
		expected_output = dataframe[ expected_output_columns ]

		open_mock = mocker.patch("builtins.open", new_callable=mock_open())

		actual_output = variance_helpers.remove_low_variance_features(dataframe, threshold, True, 'path')


		print(open_mock.call_args_list)

		open_mock.assert_called_with('path/meta.txt', 'a')

		pd.testing.assert_frame_equal(expected_output, actual_output)
		

	def test_remove_low_variance_throws_error(self):
		"""
		This test asserts that remove_low_var throws an error
		when there exist no columns within the low variance dataframe
		"""

		data_all_zero = {
			"a": [ 0, 0, 0, 0, 0],
			"b": [ 0, 0, 0, 0, 0],
			"c": [ 0, 0, 0, 0, 0],
			"d": [ 0, 0, 0, 0, 0]
		}

		dataframe_no_var = pd.DataFrame(data=data_all_zero)


		with pytest.raises(RuntimeError, match="remove_low_variance removed all features from dataframe."):
			variance_helpers.remove_low_variance_features(dataframe_no_var, threshold)

class TestWriteHighVarToFile: 
	"""
	This class tests the function write_high_var_to_file and
	 writes the high var dataframe to the file and returns the new filename.
	"""

	def test_write_high_var_to_file(self, mocker): 
		"""
		This is the base happy path test for wrote_high_var_to_file.
		"""
		input_df = dataframe
		variance_thresh = threshold
		filename = "some/path/df.tsv"
		has_index_col= False
		
		to_csv_mock = mocker.patch.object(input_df, "to_csv")

		actual_output = variance_helpers.write_high_var_to_file(input_df, variance_thresh, filename,  has_index_col)
		expected_output = f"some/path/df_ge{variance_thresh}variance.tsv"

		assert(actual_output == expected_output)
		to_csv_mock.assert_called_once_with(expected_output, sep="\t", index = has_index_col)

	def test_write_high_var_to_file_with_index(self, mocker): 
		"""
		This is the base happy path test for wrote_high_var_to_file.
		"""
		input_df = dataframe
		variance_thresh = threshold
		filename = "some/path/df.tsv"
		has_index_col= True
		
		to_csv_mock = mocker.patch.object(input_df, "to_csv")

		actual_output = variance_helpers.write_high_var_to_file(input_df, variance_thresh, filename,  has_index_col)
		expected_output = f"some/path/df_ge{variance_thresh}variance.tsv"

		assert(actual_output == expected_output)
		to_csv_mock.assert_called_once_with(expected_output, sep="\t", index=has_index_col)


class TestRemoveLowVarAndSave:
	"""
	This class tests the function that reads a given files, removes the low variance
	and saves the data to a new file by calling `
		- remove_low_variance_features
		- write_high_var_to_file
	"""

	def test_remove_low_var_and_save_no_index_tsv(self, mocker): 
		"""
		This test calls remove_low_var_and_save with a dataframe that has no indices
		"""

		input_network_path = "./test/test_data/test_raw_input/test_net_no_header.tsv"
		input_df = pd.read_csv(input_network_path, sep='\t')
		has_index_col = False
		print_meta = True
		meta_path = os.path.dirname(input_network_path)

		# Returning the dataframe object not because of any logic but to ensure there
		# is a separate df to assert against
		rlv_mock = mocker.patch(
			"src.preprocessing.variance_helpers.remove_low_variance_features",
			return_value = input_df
		)

		write_to_file_mock = mocker.patch(
			"src.preprocessing.variance_helpers.write_high_var_to_file",
			return_value = "dataframe"
		)

		variance_helpers.remove_low_var_and_save(input_network_path, threshold, has_index_col)

		## pytest does not like asserts with dfs
		# rlv_mock.assert_called_once_with(input_df, threshold)
		pd.testing.assert_frame_equal(input_df, rlv_mock.call_args[0][0])
		assert(threshold == rlv_mock.call_args[0][1])
		assert(print_meta == rlv_mock.call_args[0][2])
		assert(meta_path == rlv_mock.call_args[0][3])


		# write_to_file_mock.assert_called_once_with(dataframe, threshold, input_network_path)
		pd.testing.assert_frame_equal(input_df, write_to_file_mock.call_args[0][0])
		assert(threshold ==  write_to_file_mock.call_args[0][1])
		assert(input_network_path ==  write_to_file_mock.call_args[0][2])

	def test_remove_low_var_and_save_with_index_tsv(self, mocker): 
		"""
		This test calls remove_low_var_and_save with a dataframe that has indices
		"""

		input_network_path = "./test/test_data/test_raw_input/test_net_w_header_indexed.tsv"
		input_df = pd.read_csv(input_network_path, sep='\t', index_col=0)
		has_index_col = True
	

		# Returning the dataframe object not because of any logic but to ensure there
		# is a separate df to assert against
		rlv_mock = mocker.patch(
			"src.preprocessing.variance_helpers.remove_low_variance_features",
			return_value = input_df
		)
		write_to_file_mock = mocker.patch(
			"src.preprocessing.variance_helpers.write_high_var_to_file",
			return_value = "dataframe"
		)

		variance_helpers.remove_low_var_and_save(input_network_path, threshold, has_index_col)
	
		## pytest does not like asserts with dfs
		# rlv_mock.assert_called_once_with(input_df, threshold)
		pd.testing.assert_frame_equal(input_df, rlv_mock.call_args[0][0])
		assert(threshold == rlv_mock.call_args[0][1])

		# write_to_file_mock.assert_called_once_with(dataframe, threshold, input_network_path)
		pd.testing.assert_frame_equal(input_df, write_to_file_mock.call_args[0][0])
		assert(threshold ==  write_to_file_mock.call_args[0][1])
		assert(input_network_path ==  write_to_file_mock.call_args[0][2])

	def test_remove_low_var_and_save_with_csv(self, mocker): 
		"""
		This test calls remove_low_var_and_save with a dataframe that has indices
		"""
		
		input_network_path = "./test/test_data/test_raw_input/test_net_w_header.tsv"
		sep='\t'
		input_df = pd.read_csv(input_network_path, sep=sep)
		has_index_col = False
	

		# Returning the dataframe object not because of any logic but to ensure there
		# is a separate df to assert against
		rlv_mock = mocker.patch(
			"src.preprocessing.variance_helpers.remove_low_variance_features",
			return_value = input_df
		)
		write_to_file_mock = mocker.patch(
			"src.preprocessing.variance_helpers.write_high_var_to_file",
			return_value = "dataframe"
		)

		variance_helpers.remove_low_var_and_save(input_network_path, threshold, has_index_col, sep)
	
		## pytest does not like asserts with dfs
		# rlv_mock.assert_called_once_with(input_df, threshold)
		pd.testing.assert_frame_equal(input_df, rlv_mock.call_args[0][0])
		assert(threshold == rlv_mock.call_args[0][1])

		# write_to_file_mock.assert_called_once_with(dataframe, threshold, input_network_path)
		pd.testing.assert_frame_equal(input_df, write_to_file_mock.call_args[0][0])
		assert(threshold ==  write_to_file_mock.call_args[0][1])
		assert(input_network_path ==  write_to_file_mock.call_args[0][2])

