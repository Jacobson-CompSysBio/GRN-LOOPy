import pandas as pd
import pytest

from src.postprocessing import network_thresholding_helpers

class TestThresholdEdgelist: 
	"""
	This class test is a test suite for threshld_edgelist

	Total Net:                Thresh net:
	 'c', 'f', 0.8            'c', 'f', 0.8 
	 'k', 'f', 0.75    top    'k', 'f', 0.75
	 'g', 'e', 0.7    thresh  'g', 'e', 0.7 
	 'd', 'b', 0.65    0.4    'd', 'b', 0.65
	 'f', 'a', 0.6     =>
	 'b', 'f', 0.5
	 'c', 'b', 0.4
	 'g', 'd', 0.25
	 'a', 'b', 0.2
	 'd', 'e', 0.1
	"""


	sorted_edgelist = pd.DataFrame({
		"from":   ['c',   'k', 'g',  'd', 'f',  'b', 'c', 'g',  'a',   'd' ],
		"to":     ['f',   'f', 'e',  'b', 'a',  'f', 'b', 'd',  'b',   'e' ],
		"weight": [ 0.8, 0.75,  0.7, 0.65, 0.6, 0.5, 0.4, 0.25, 0.2,  0.1 ]
	})

	def test_threhsold_edgelist(self): 
		"""
		This is a base test of create model to ensure that the 
		default model is chosen
		"""

		actual_output = network_thresholding_helpers.threshold_edgelist(
				sorted_data = self.sorted_edgelist, 
				top_pct = 0.4)

		expected_output =  pd.DataFrame({
			"from":   ['c',   'k', 'g',  'd'],
			"to":     ['f',   'f', 'e',  'b'],
			"weight": [ 0.8, 0.75,  0.7, 0.65]
		})

		pd.testing.assert_frame_equal(actual_output, expected_output)

	def test_threhsold_edgelist_one_pct(self): 
		"""
		This is a base test of create model to ensure that the 
		default model is chosen
		""" 

		actual_output = network_thresholding_helpers.threshold_edgelist(
				sorted_data = self.sorted_edgelist, 
				top_pct = 0.1)

		expected_output =  pd.DataFrame({
			"from":   ['c'],
			"to":     ['f'],
			"weight": [ 0.8]
		})

		pd.testing.assert_frame_equal(actual_output, expected_output)

		

class TestSortDataFrame: 
	"""
	This class test is a test suite for threshld_edgelist

	Total Net:                Thresh net:
	 'c', 'f', 0.8            'c', 'f', 0.8 
	 'k', 'f', 0.75    top    'k', 'f', 0.75
	 'g', 'e', 0.7    thresh  'g', 'e', 0.7 
	 'd', 'b', 0.65    0.4    'd', 'b', 0.65
	 'f', 'a', 0.6     =>
	 'b', 'f', 0.5
	 'c', 'b', 0.4
	 'g', 'd', 0.25
	 'a', 'b', 0.2
	 'd', 'e', 0.1
	"""

	base_edgelist = pd.DataFrame({
		"from":   [ 'a', 'c', 'b', 'd', 'g', 'g', 'f', 'c', 'd', 'k'],
		"to":     [ 'b', 'b', 'f', 'e', 'd', 'e', 'a', 'f', 'b', 'f'],
		"weight": [ 0.2, 0.4, 0.5, 0.1, 0.25, 0.7, 0.6, 0.8, 0.65, 0.75]
	})


	sorted_edgelist = pd.DataFrame({
		"from":   ['c',   'k', 'g',  'd', 'f',  'b', 'c', 'g',  'a',   'd' ],
		"to":     ['f',   'f', 'e',  'b', 'a',  'f', 'b', 'd',  'b',   'e' ],
		"weight": [ 0.8, 0.75,  0.7, 0.65, 0.6, 0.5, 0.4, 0.25, 0.2,  0.1 ]
	})

	def test_sort_data_frame(self): 
		"""
		This is a base test of create model to ensure that the 
		default model is chosen
		"""
		input_df = self.base_edgelist

		actual_output = network_thresholding_helpers.sort_edgelist(input_df)

		pd.testing.assert_frame_equal(actual_output, self.sorted_edgelist)

	def test_sort_edgelist_by_colname(self): 
		"""
		This is a base test of create model to ensure that the 
		default model is chosen
		""" 
		input_df = self.base_edgelist
		column_name = 'weight'
		actual_output = network_thresholding_helpers.sort_edgelist(input_df, sort_column=column_name)

		pd.testing.assert_frame_equal(actual_output, self.sorted_edgelist)

	
	def test_sort_edgelist_with_extra_columns(self): 
		"""
		This is a base test of create model to ensure that the 
		default model is chosen
		""" 
		input_df = self.base_edgelist
		input_df['extra_column'] = 'F'
		actual_output = network_thresholding_helpers.sort_edgelist(input_df)

		expected_output = self.sorted_edgelist
		expected_output['extra_column'] = 'F'

		pd.testing.assert_frame_equal(actual_output, expected_output)

	