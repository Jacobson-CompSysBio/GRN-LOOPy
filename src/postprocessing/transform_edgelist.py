import igraph as ig
import pandas as pd

def transform_edgelist_to_undirected(df: pd.DataFrame): 
	edges = [tuple(x) for x in df.values]
	g = ig.Graph.TupleList(edges, edge_attrs=['weight'], directed=True)
	undirected_g = g.as_undirected(combine_edges=max)
	
	undirected_df = undirected_g.get_edge_dataframe()
	undirected_df_vert = undirected_g.get_vertex_dataframe()
	undirected_df['source'].replace(undirected_df_vert['name'], inplace=True)
	undirected_df['target'].replace(undirected_df_vert['name'], inplace=True)
	return undirected_df