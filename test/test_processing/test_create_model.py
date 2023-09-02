import numpy as np 
import pandas as pd
import pytest
import os
from mock import mock_open
# import lightgbm as lgb

from src.processing import create_model


class TestCreateModel: 
	"""
	This class Testis a test suite for the Create Model function.
	"""

	def test_create_correlation_list_base(self, mocker): 
		"""
		This is a base test of create model to ensure that the 
		default model is chosen
		"""
		expected_output = {"model": "this_is_our_model"}

		lgb_mock = mocker.patch(
			"lightgbm.LGBMRegressor",
			return_value = expected_output
		)

		actual_output = create_model.create_model()

		lgb_mock.assert_called_once_with(
			boosting_type= 'gbdt',
			objective= 	 'regression', 
			learning_rate= 	 0.1,
			n_estimators= 	 100,
			num_leaves= 	 31,
			max_depth= 	 -1,
			verbose= 	 1,
			random_state= 	 42,
			device='cpu' 
		)

	def test_create_correlation_list_base(self, mocker): 
		"""
		This is a base test of create model to ensure a specifici
		 model is chosen
		"""
		expected_output = {"model": "this_is_our_model2"}

		lgb_mock = mocker.patch(
			"lightgbm.LGBMRegressor",
			return_value = expected_output
		)

		expected_learning_rate = 25
		expected_device = 'gpu'
		actual_output = create_model.create_model(
			learning_rate=expected_learning_rate,
			device= expected_device, 
			gpu_device_id=7
		)

		lgb_mock.assert_called_once_with(
			boosting_type= 'gbdt',
			objective= 	 'regression', 
			learning_rate= 	expected_learning_rate,
			n_estimators= 	 100,
			num_leaves= 	 31,
			max_depth= 	 -1,
			verbose= 	 1,
			random_state= 	 42,
			device=expected_device,
			gpu_device_id = 7
		)
