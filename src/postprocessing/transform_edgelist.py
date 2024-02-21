import igraph as ig
import pandas as pd

def transform_edgelist_to_undirected(df: pd.DataFrame): 
	edges = [tuple(x) for x in df.values]
	print("converting edges to tuple", flush=True)
	g = ig.Graph.TupleList(edges, edge_attrs=['weight'], directed=True)
	print("converting to undirected", flush=True)
	g = g.as_undirected(combine_edges=max)
	
	print("converting graph to dataframe", flush=True)
	undirected_df = g.get_edge_dataframe()
	print("converting vertices to df", flush=True)
	undirected_df_vert = g.get_vertex_dataframe()
	print("updating source nodes", flush=True)
	del(g)
	undirected_df['source'].replace(undirected_df_vert['name'], inplace=True)
	print("updating target nodes", flush=True)
	undirected_df['target'].replace(undirected_df_vert['name'], inplace=True)
	return undirected_df