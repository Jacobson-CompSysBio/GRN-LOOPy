import numpy as np 
import pandas as pd
import pytest
import os
from mock import MagicMock, patch, Mock, call
from unittest import TestCase 

from src.processing import run_model


class TestRunModel: 
	"""
	This class tests the run_model functionality
	"""

	class BaseModel:
		model_name = None
		feature_importances_ = [3, 5, 1, 2] # arbitrary FI values
		def fit(): 
			return 0
		def predict(): 
			return 0


	def test_run_model(self, mocker): 
		"""
		This function tests the test_create_non_rep_df by creating a 
		dataframe where all non-representative values are added back. 
		"""
		mock_prediction_output = pd.Series([1,2])

		model = self.BaseModel()
		model.fit = Mock()
		model.predict = Mock(return_value=mock_prediction_output)

		data = pd.read_csv(
			"./test/test_data/test_raw_input/test_net_w_header.tsv",
			sep='\t'
		)
		train = data.loc[:3]
		test = data.loc[3:]
		
		x_cols = ['a', 'b','c','d']
		y_col = 'e'

		eval_set = False
		device = 'cpu'

		time_mock = mocker.patch(
			'time.time',
			return_value = 1
		)
		r2_mock = mocker.patch(
			'sklearn.metrics.r2_score', 
			return_value = 0.5
		)

		actual_output = run_model.run_model(
			model,
			train,
			test,
			x_cols,
			y_col,
			eval_set, 
			device
		)

		expected_output ={
			'device': device,
			'train_time': 0, # 1 - 1 = 0
			'r2': 0.5,
			'feature_imps': "3,5,1,2",
			'features': 'a,b,c,d',
			'eval_set': eval_set
		}
		
		model_args_position, model_args_param  = model.fit.call_args

		assert actual_output == expected_output
		time_mock.assert_has_calls([call(), call()]) # called twice
		pd.testing.assert_frame_equal( model_args_position[0], train[x_cols] )
		pd.testing.assert_series_equal( model_args_position[1], train[y_col] )
		
		assert model_args_param == {}
		pd.testing.assert_frame_equal( model.predict.call_args[0][0], test[x_cols])
		pd.testing.assert_series_equal( r2_mock.call_args[0][0], mock_prediction_output)
		pd.testing.assert_series_equal( r2_mock.call_args[0][1], test[y_col])

	def test_run_model_w_eval_gpu(self, mocker): 
		"""
		This function tests the test_create_non_rep_df by creating a 
		dataframe where all non-representative values are added back. 
		"""
		mock_prediction_output = pd.Series([1,2])

		model = self.BaseModel()
		model.fit = Mock()
		model.predict = Mock(return_value=mock_prediction_output)

		data = pd.read_csv(
			"./test/test_data/test_raw_input/test_net_w_header.tsv",
			sep='\t'
		)
		train = data.loc[:3]
		test = data.loc[3:]
		
		x_cols = ['a', 'b','d', 'e']
		y_col = 'c'

		eval_set = True
		device = 'gpu'

		time_mock = mocker.patch(
			'time.time',
			return_value = 1
		)
		r2_mock = mocker.patch(
			'sklearn.metrics.r2_score', 
			return_value = 0.5
		)

		actual_output = run_model.run_model(
			model,
			train,
			test,
			x_cols,
			y_col,
			eval_set, 
			device
		)

		expected_output ={
			'device': device,
			'train_time': 0, # 1 - 1 = 0
			'r2': 0.5,
			'feature_imps': "3,5,1,2",
			'features': 'a,b,d,e',
			'eval_set': eval_set
		}


		model_args_position, model_args_param  = model.fit.call_args

		assert actual_output == expected_output
		time_mock.assert_has_calls([call(), call()]) # called twice
		pd.testing.assert_frame_equal( model_args_position[0], train[x_cols] )
		pd.testing.assert_series_equal( model_args_position[1], train[y_col] )
		pd.testing.assert_frame_equal( model_args_param['eval_set'][0], test[x_cols] )
		pd.testing.assert_series_equal( model_args_param['eval_set'][1], test[y_col] )
		pd.testing.assert_frame_equal( model.predict.call_args[0][0], test[x_cols])
		pd.testing.assert_series_equal( r2_mock.call_args[0][0], mock_prediction_output)
		pd.testing.assert_series_equal( r2_mock.call_args[0][1], test[y_col])