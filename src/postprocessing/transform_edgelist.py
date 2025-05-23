import igraph as ig
import pandas as pd
from pandarallel import pandarallel

def replace_index_with_name(x_index, idx_to_name_df):
	return idx_to_name_df.loc[x_index]['name']

def transform_edgelist_to_undirected(df: pd.DataFrame):
	pandarallel.initialize()
	edges = [tuple(x) for x in df.values]
	print("converting edges to tuple", flush=True)
	g = ig.Graph.TupleList(edges, edge_attrs=['weight'], directed=True)
	print("converting to undirected", flush=True)
	g = g.as_undirected(combine_edges=max)

	print("converting graph to dataframe", flush=True)
	undirected_df = g.get_edge_dataframe()
	print("converting vertices to df", flush=True)
	undirected_df_vert = g.get_vertex_dataframe()
	del(g)

	print("updating source nodes", flush=True)
	undirected_df['source'] = undirected_df['source'].parallel_apply(replace_index_with_name, args=(undirected_df_vert,) )#lambda x: undirected_df_vert.loc[x]['name'])
	print("updating target nodes", flush=True)
	undirected_df['target'] = undirected_df['target'].parallel_apply(replace_index_with_name, args=(undirected_df_vert,) )#lambda x: undirected_df_vert.loc[x]['name'])

	return undirected_df
