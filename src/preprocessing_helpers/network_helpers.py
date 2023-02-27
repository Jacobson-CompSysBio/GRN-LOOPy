import pandas as pd
import networkx as nx
import random

### NETWORK Functions: 
def convert_df_to_network(df):
    """
    This function converts a stacked dataframe from a DF to an NX network object.
    """
    df.columns = ["source", "target", "weight"] 
    network = nx.from_pandas_edgelist(df)
    
    return network

def create_representative_set(network): 
    random.seed(42)
    """
    This function takes a NX network object, takes a list of all connected components, 
    and then chooses the first element as a representative for that connected component.
    """
    S = list(nx.connected_components(network))
    representatives = []
    element_map = {}
    for subgraph in S:
        representative = list(subgraph)[0]
        representatives.append(representative)

        # Get the remainder of elements within S1: 
        for element in subgraph:
            if element == representative: 
                continue

            element_map[element] = representative

    return representatives, element_map

def write_representative_map_to_file(element_map, outfile_name): 
    """
    This function writes the representatives to file. 
    """
    print("Writing to file", outfile_name) 
    non_representative_column = list(element_map.keys())
    representative_column = list(element_map.values())
    
    df = pd.DataFrame({"non_representative": non_representative_column,
                  "representative": representative_column})
    print(df.head())
    df.to_csv(outfile_name, sep='\t', index=None)
    
                            
def extract_representatives_and_save_to_files(df=None, df_filepath=None, original_data_file=None, outfile_name = None, delim='\t'): 
    """
    This takes in a dataframe edgelist or filepath to a dataframe edgelist, 
    converts the dataframe to a networkx network, extracts the representative
    set, and then saves all saved representations to file.
    """
    if df is None and df_filepath is None: 
        raise Exception("No dataframe or filepath included") 
    
    df = df if df is not None else pd.read_csv(df_filepath, sep=delim)
    
    network = convert_df_to_network(df)
    representatives, representative_map = create_representative_set(network) 
    all_nodes = network.nodes
    non_representatives = list(filter( lambda x: x not in representatives, all_nodes))
    
    if outfile_name is None and original_data_file is None: 
        output_name = "nonrep_to_representative_map.tsv"
    elif outfile_name is None and original_data_file is not None: 
        base_name = '.'.join(original_data_file.split('.')[0:-1])
        output_name = f"{base_name}_nonrep_to_rep_map.tsv"
    else: 
        output_name = outfile_name
    
    write_representative_map_to_file(representative_map, output_name)
    
    return representatives, non_representatives

def remove_representatives_from_main_dataset_and_save(raw_data_file, non_representatives): 
    """
    This function reads in the raw data and removes the non_representative elements.
    """
    df = pd.read_csv(raw_data_file, sep="\t")
    
    filtered_cols = list(filter(lambda x: x not in non_representatives, df.columns))
    filtered_df = df[ filtered_cols ]
    
    base_name = '.'.join(raw_data_file.split('.')[0:-1])
    
    outfile_name = f"{base_name}_no_correlated_data.tsv"

    filtered_df.to_csv(outfile_name, sep='\t', index=False)
