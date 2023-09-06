import networkx as nx
import igraph
import numpy as np 
import pandas as pd
import pytest
import os
from mock import mock_open


from src.preprocessing import network_helpers

class TestConvertDFToNetwork: 
	"""
	This class tests the function `convert_df_to_network`

		from  to  corr			E──┐
		A	 B   1				  │	   F
		A	 C   1		   D──────B────C──┘
		B	 C   1   =====>		 │	│ 
		B	 D   1				  │	│
		B	 E   1				  A────┘
		C	 F   1 
	
   		   a  b  c  d  e  f
		a  0  1  1  0  0  0
		b  1  0  1  1  1  0
		c  1  1  0  0  0  1
		d  0  1  0  0  0  0
		e  0  1  0  0  0  0	
		f  0  0  1  0  0  0
	"""

	def test_converts_df_to_network(self): 
		"""
		This tests converting a network
		"""
		adj_array = np.array([
			np.array([0, 1, 1, 0, 0, 0]),
			np.array([1, 0, 1, 1, 1, 0]),
			np.array([1, 1, 0, 0, 0, 1]),
			np.array([0, 1, 0, 0, 0, 0]),
			np.array([0, 1, 0, 0, 0, 0]),
			np.array([0, 0, 1, 0, 0, 0])
		])
		nodes = ['A','B','C', 'D', 'E','F']
		
		input_dict = {
			"from" : ["A", "A", "B", "B", "B", "C"],
			"to" :   ["B", "C", "C", "D", "E", "F"],
			"corr":  [1, 1, 1, 1, 1, 1]
		}
		input_df = pd.DataFrame(input_dict)

		actual_output = network_helpers.convert_df_to_network(input_df)

		expected_output = nx.Graph()
		expected_output.add_edges_from([
			["A" , "B"],
			["A" , "C"],
			["B" , "C"],
			["B" , "D"],
			["B" , "E"],
			["C" , "F"]
		])

		assert(nx.utils.graphs_equal(actual_output, expected_output))

class TestCreateRepresentativeSet: 
	"""
	This class tests the create_representative_set method which 
	takes a NX network object, takes a list of all connected components, 
	and then chooses the first element as a representative for that
	connected component.
	"""

	def creates_a_representative_set(self): 
		"""
		This test is the base case and creates a rep set. 
		
		Input is a disconnected network filled with cliques. 
		┌────┐
		B──C │	 G
		│ /│ │	/│   ===> Representatives: {B, E}
		│/ │ │   / │
		A──D─┘  E──F
		"""

		disconnected_cliques = nx.Graph()
		disconnected_cliques.add_edges_from([
				['A', 'B'],
				['A', 'C'],
				['A', 'D'],
				['B', 'C'],
				['B', 'D'],
				['C', 'D'],
				['E', 'F'],
				['E', 'G'],
				['G', 'F']
		])

		actual_representatives, actual_representative_map = network_helpers.create_representative_set(disconnected_cliques)

		expected_representatives = ['C', 'E']
		expected_representative_map = {
			'D':'C',
			'B':'C',
			'A':'C',
			'G':'E',
			'F':'E'
		}

		assert(actual_representative_map == expected_representative_map)
		assert(actual_representatives == expected_representatives)

class TestWriteRepresentativeMapToFile: 
	"""
	This class tests the write_representative_map_to_file method which 
	takes an element map and writs the keys and values to a tsv
	"""
	def test_writes_map_to_file(self, mocker): 
		"""
		This function tests whether the map gets written to file
		"""

		file_name = './test_rep_map.tsv'
		rep_map = {
			'a': 'f',
			'b': 'f',
			'c': 'f',
			'd': 'f',
			'e': 'f',
			'h': 'l',
			'i': 'l',
			'j': 'l',
			'k': 'l'
		}
		
		expected_rep_map_df = pd.DataFrame({
			'non_representative': list(rep_map.keys()),
			'representative': list(rep_map.values())
		})

		data_frame_mock = mocker.patch("pandas.DataFrame", return_value=expected_rep_map_df)
		to_csv_mock = mocker.patch.object(expected_rep_map_df, "to_csv")

		network_helpers.write_representative_map_to_file(rep_map, file_name)
		
		data_frame_mock.assert_called_once_with({
			'non_representative': list(rep_map.keys()),
			'representative': list(rep_map.values())
		})
		to_csv_mock.assert_called_once_with(file_name, sep='\t', index=None)


class TestEXtractRepresentativesAndSaveToFile: 
	"""
	This class tests the function extract_representatives_and_save_to_files which 
	either takes a correlation dataframe, or reads one ine, creates a representative
	set via `create_representative_set`, then writes to file with `write_representative_map_to_file`
	"""
	as_df = pd.DataFrame({
		'source': ['A', 'A', 'A', 'B', 'B', 'C', 'E', 'E', 'G'],
		'target': ['B', 'C', 'D', 'C', 'D', 'D', 'F', 'G', 'F'],
		'weight': [1, 1, 1, 1, 1, 1, 1, 1, 1]
	})
	as_graph = nx.Graph()
	as_graph.add_edges_from([
			['A', 'B'],
			['A', 'C'],
			['A', 'D'],
			['B', 'C'],
			['B', 'D'],
			['C', 'D'],
			['E', 'F'],
			['E', 'G'],
			['G', 'F']
	])
	representatives = ['C', 'E']
	rep_map={
		'D':'C',
		'B':'C',
		'A':'C',
		'G':'E',
		'F':'E'
	}
	non_representatives = ['B', 'A', 'D', 'F', 'G']
	base_outfile_non_rep_name = 'nonrep_to_representative_map.tsv'


	def test_throws_error_when_no_df_or_file_given(self, mocker): 
		with pytest.raises(Exception, match="No dataframe or filepath included"):
			network_helpers.extract_representatives_and_save_to_files()

	def test_calculates_from_supplied_correlation_df(self, mocker): 
		"""
		This function tests whether or not a file is read in and 
		the proper pipeline of commands are followed
		""" 
		input_df_file = 'file.txt'
		input_df = self.as_df

		read_csv_mock = mocker.patch(
			'pandas.read_csv',
			return_value = input_df
		)
		convert_mock = mocker.patch(
			'src.preprocessing.network_helpers.convert_df_to_network',
			return_value = self.as_graph
		)
		create_representative_mock = mocker.patch(
			'src.preprocessing.network_helpers.create_representative_set',
			return_value = (self.representatives, self.rep_map)
		)
		write_representative_map_mock = mocker.patch(
			'src.preprocessing.network_helpers.write_representative_map_to_file',
		)
		
		actual_output = network_helpers.extract_representatives_and_save_to_files(df_filepath = input_df_file)
		
		read_csv_mock.assert_called_once_with(input_df_file, sep='\t')
		convert_mock.assert_called_once()
		pd.testing.assert_frame_equal(convert_mock.call_args[0][0], input_df)

		create_representative_mock.assert_called_once_with(self.as_graph)

		write_representative_map_mock.assert_called_once_with(self.rep_map, self.base_outfile_non_rep_name)


	def test_calculates_from_supplied_correlation_file(self, mocker): 
		"""
		This function tests whether or not a file is read in and 
		the proper pipeline of commands are followed
		""" 
		input_df = self.as_df

		convert_mock = mocker.patch(
			'src.preprocessing.network_helpers.convert_df_to_network',
			return_value = self.as_graph
		)
		create_representative_mock = mocker.patch(
			'src.preprocessing.network_helpers.create_representative_set',
			return_value = (self.representatives, self.rep_map)
		)
		write_representative_map_mock = mocker.patch(
			'src.preprocessing.network_helpers.write_representative_map_to_file',
		)
		
		actual_output = network_helpers.extract_representatives_and_save_to_files(df = self.as_df)
		
		convert_mock.assert_called_once()
		pd.testing.assert_frame_equal(convert_mock.call_args[0][0], input_df)

		create_representative_mock.assert_called_once_with(self.as_graph)

		write_representative_map_mock.assert_called_once_with(self.rep_map, self.base_outfile_non_rep_name)


	def test_calculates_from_supplied_correlation_with_originaldatafile(self, mocker): 
		"""
		This function tests whether or not a file is read in and 
		the proper pipeline of commands are followed
		""" 
		input_df = self.as_df
		original_data_file = "original.tsv"

		convert_mock = mocker.patch(
			'src.preprocessing.network_helpers.convert_df_to_network',
			return_value = self.as_graph
		)
		create_representative_mock = mocker.patch(
			'src.preprocessing.network_helpers.create_representative_set',
			return_value = (self.representatives, self.rep_map)
		)
		write_representative_map_mock = mocker.patch(
			'src.preprocessing.network_helpers.write_representative_map_to_file',
		)
		
		actual_output = network_helpers.extract_representatives_and_save_to_files(df = self.as_df, original_data_file=original_data_file)
		
		expected_file_name = f"original_nonrep_to_rep_map.tsv"

		convert_mock.assert_called_once()
		pd.testing.assert_frame_equal(convert_mock.call_args[0][0], input_df)

		create_representative_mock.assert_called_once_with(self.as_graph)

		write_representative_map_mock.assert_called_once_with(self.rep_map, expected_file_name)

	def test_calculates_from_supplied_correlation_with_outfile(self, mocker): 
		"""
		This function tests whether or not a file is read in and 
		the proper pipeline of commands are followed
		""" 
		input_df = self.as_df
		outfile_name = "outfile.tsv"

		convert_mock = mocker.patch(
			'src.preprocessing.network_helpers.convert_df_to_network',
			return_value = self.as_graph
		)
		create_representative_mock = mocker.patch(
			'src.preprocessing.network_helpers.create_representative_set',
			return_value = (self.representatives, self.rep_map)
		)
		write_representative_map_mock = mocker.patch(
			'src.preprocessing.network_helpers.write_representative_map_to_file',
		)
		
		actual_output = network_helpers.extract_representatives_and_save_to_files(df = self.as_df, outfile_name=outfile_name)
		
		expected_file_name = outfile_name

		convert_mock.assert_called_once()
		pd.testing.assert_frame_equal(convert_mock.call_args[0][0], input_df)

		create_representative_mock.assert_called_once_with(self.as_graph)

		write_representative_map_mock.assert_called_once_with(self.rep_map, expected_file_name)


# class TestConvertDirectedToUndirected: 
# 	"""
# 	This class tests the function `convert_directed_to_undirected`

		
# 		to	from	weight		   E──┐
# 		e	b		1   			  ▼
# 		d	b		1	 	   D─────►B◄──►C◄──►F
# 		c	b		0.7  			  │	▲
# 		b	a		1	 			  ▼	│
# 		b	c		0.5   			  A◄───┘
# 		c	a		0.25
# 		a	c		0.9
# 		c	f		0.3
# 		f	c		0.8
	
#			   a	b	c	 d  e  f			a	b	c	 d  e  f   
# 		a  0	1	0.25  0  0  0		 a  0	1	0.9   0  0  0   
# 		b  0	0	0.7   1  1  0		 b  1	0	0.7   1  1  0   
# 		c  0.9  0.5  0	 0  0  0.8 ==>   c  0.9  0.7  0	 0  0  0.8 
# 		d  0	0	0	 0  0  0		 d  0	1	0	 0  0  0   
# 		e  0	0	0	 0  0  0		 e  0	1	0	 0  0  0   
# 		f  0	0	0.3   0  0  0		 f  0	0	0.8   0  0  0   
# 	"""

# 	def test_converts_directed_to_undirected(self): 
# 		"""
# 		This tests converting a network
# 		and maintains the highest edge weight
# 		"""


# 		adj_array =  igraph.Matrix([
# 			[ 0,	  0, 0.9, 0, 0, 0  ], 
# 			[ 1,	  0, 0.5, 0, 0, 0  ],
# 			[ 0.25, 0.7,   0, 0, 0, 0.3],
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  0, 0.8, 0, 0, 0  ] 
# 		])

# 		g1 = igraph.Graph.Weighted_Adjacency(adj_array)
# 		print("INPUT VALUE ", g1)
# 		actual_output = network_helpers.convert_directed_to_undirected(g1)
# 		print("ACTUAL OUTPUT: ", actual_output)
# 		expected_adj_array =  igraph.Matrix([
# 			[ 0,	  1, 0.9, 0, 0, 0  ], 
# 			[ 1,	  0, 0.7, 1, 1, 0  ],
# 			[ 0.9 , 0.7,   0, 0, 0, 0.8],
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  0, 0.8, 0, 0, 0  ] 
# 		])

# 		actual_weighted_adj = actual_output.get_adjacency(attribute='weight')

# 		assert(actual_weighted_adj, expected_adj_array)

# 	def test_converts_directed_to_undirected_with_names(self): 
# 		"""
# 		This tests converting a network
# 		and maintains the highest edge weight
# 		"""


# 		adj_array =  igraph.Matrix([
# 			[ 0,	  0, 0.9, 0, 0, 0  ], 
# 			[ 1,	  0, 0.5, 0, 0, 0  ],
# 			[ 0.25, 0.7,   0, 0, 0, 0.3],
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  0, 0.8, 0, 0, 0  ] 
# 		])

# 		g1 = igraph.Graph.Weighted_Adjacency(adj_array)
# 		print("INPUT VALUE ", g1)
# 		actual_output = network_helpers.convert_directed_to_undirected(g1)
# 		print("ACTUAL OUTPUT: ", actual_output)
# 		expected_adj_array =  igraph.Matrix([
# 			[ 0,	  1, 0.9, 0, 0, 0  ], 
# 			[ 1,	  0, 0.7, 1, 1, 0  ],
# 			[ 0.9 , 0.7,   0, 0, 0, 0.8],
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  1,   0, 0, 0, 0  ], 
# 			[ 0,	  0, 0.8, 0, 0, 0  ] 
# 		])

# 		actual_weighted_adj = actual_output.get_adjacency(attribute='weight')

# 		assert(actual_weighted_adj, expected_adj_array)


