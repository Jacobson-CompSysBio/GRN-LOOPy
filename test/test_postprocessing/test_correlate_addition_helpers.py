import numpy as np 
import pandas as pd
import pytest
import os
from mock import MagicMock, patch, mock_open, call
from unittest import TestCase 

from src.postprocessing import correlate_addition_helpers


class TestCreateNonRepDF: 
	"""
	This class tests the create non rep df function
	"""
	def test_creates_non_rep_df(self): 
		"""
		This function tests the test_create_non_rep_df by creating a 
		dataframe where all non-representative values are added back. 
		"""
		input_network_edgelist = pd.read_csv('./test/test_data/test_edgelists/test_net_noidx.csv')
		input_network_edgelist = input_network_edgelist.sort_values(by='weight',  ascending=False).reset_index(drop=True)
		sliced_input = input_network_edgelist.loc[:5]

		correlated_data_df = pd.read_csv('./test/test_data/test_rep_maps/test_rep_map.tsv', sep='\t')

		actual_output = correlate_addition_helpers.create_non_rep_df(
				sliced_input,
				correlated_data_df
		)

		expected_output = pd.DataFrame({
			"from":   ["P",    "H",  "E",   "X",   "G",  "P",  "C", "C"  ],
			"to":     ["E",    "X",  "P",   "H",   "X",  "C",  "M", "N"  ],
			"weight": [ 0.95, 0.95, 0.87,  0.87,  0.75,  0.6,  0.3, 0.3  ]
		})
		pd.testing.assert_frame_equal(actual_output, expected_output)


	def test_creates_no_non_rep_df(self): 
		"""
		This function tests the test_create_non_rep_df by warning the user 
		"""
		input_network_edgelist = pd.read_csv('./test/test_data/test_edgelists/test_net_noidx.csv')
		input_network_edgelist = input_network_edgelist.sort_values(by='weight',  ascending=False).reset_index(drop=True)
		sliced_input = input_network_edgelist.loc[:5]

		correlated_data_df = pd.read_csv('./test/test_data/test_rep_maps/test_rep_map_mno.tsv', sep='\t')
		expected_output = None

		base_warn = "No representatives from supplied file exist in network.\n"
		suggestion = "This could be caused by using the wrong representative map file\n"
		suggestion2 = " or setting a threshold too high. "

		with pytest.warns(UserWarning, match =f"{base_warn} {suggestion} {suggestion2}"):
			actual_output = correlate_addition_helpers.create_non_rep_df(
					sliced_input,
					correlated_data_df
			)
			
			assert actual_output == expected_output

class TestAddCorrelatesBackToDF:
	"""
	This class tests the addition of the non-representative data to the base dataframe
	"""
	input_network_edgelist = pd.read_csv('./test/test_data/test_edgelists/test_net_noidx.csv')
	input_network_edgelist = input_network_edgelist.sort_values(by='weight',  ascending=False).reset_index(drop=True)
	input_correlate_path = './test/test_data/test_rep_maps/test_rep_map.tsv'
	correlate_dataframe = pd.read_csv(input_correlate_path, sep='\t')

	def test_add_correlates_back_to_df(self, mocker): 
		sliced_input = self.input_network_edgelist.loc[:3]
		mocked_create_non_rep_df_output = pd.DataFrame({
			"from":   [ "X"],
			"to":     [ "Y"],
			"weight": [ 23 ]
		})

		correlate_df_mock = mocker.patch(
			"src.postprocessing.correlate_addition_helpers.create_non_rep_df",
			return_value = mocked_create_non_rep_df_output
		)

		actual_output = correlate_addition_helpers.add_correlates_back_to_df(
			sliced_input, 
			self.input_correlate_path
		)

		expected_output = pd.DataFrame({
			"from":   ["H",   "E",   "G",  "H", "X" ],
			"to":     ["E",   "H",   "E",  "C", "Y" ],
			"weight": [ 0.95,0.87,  0.75,  0.6,  23 ]
		})
		
		pd.testing.assert_frame_equal(actual_output, expected_output)
		pd.testing.assert_frame_equal(correlate_df_mock.call_args[0][0], sliced_input)
		pd.testing.assert_frame_equal(correlate_df_mock.call_args[0][1], self.correlate_dataframe)
	
	def test_add_correlates_back_to_df_no_correlates(self, mocker): 
		sliced_input = self.input_network_edgelist.loc[:3]

		mocked_create_non_rep_df_output = None
		correlate_df_mock = mocker.patch(
			"src.postprocessing.correlate_addition_helpers.create_non_rep_df",
			return_value = mocked_create_non_rep_df_output
		)

		actual_output = correlate_addition_helpers.add_correlates_back_to_df(
			sliced_input,
			self.input_correlate_path
		)

		expected_output = pd.DataFrame({
			"from":   ["H",   "E",   "G",  "H"],
			"to":     ["E",   "H",   "E",  "C"],
			"weight": [ 0.95,0.87,  0.75,  0.6]
		})
		
		pd.testing.assert_frame_equal(actual_output, expected_output)
		pd.testing.assert_frame_equal(correlate_df_mock.call_args[0][0], sliced_input)
		pd.testing.assert_frame_equal(correlate_df_mock.call_args[0][1], self.correlate_dataframe)
