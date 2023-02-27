import pandas as pd

def remove_low_variance_features(raw_df, variance_thresh):
    """
    This function removes features with variance that
    falls under the threshold variance_thresh
    """
    
    threshold_mask = raw_df.var() >= variance_thresh
    
    return raw_df[ threshold_mask ]

def write_high_var_to_file(high_var_df, variance_thresh, filename):
    """
    This function writes the high var dataframe to file and then
    returns the new file name. 
    """
    base_name = '.'.join(filename.split('.')[0:-1])
    
    outfile_name = f"{base_name}_ge{variance_thresh}variance.tsv"
    
    high_var_df.to_csv(outfile_name, sep='\t', index=None)
    
    return outfile_name

def remove_low_var_and_save(input_file, variance_thresh, has_index_col):
    """
    This function combines the two above functions and reorders the dataframe
    to ensure that the original index columns are not lost.
    """
    raw_data = pd.read_csv(input_file, sep='\t')
    
    index_col_vars = {}
    if has_index_col: 
        # ASSUMES INDEX COL IS FIRST COL
        colnames = raw_data.columns
        index_col_vars['non_index_cols'] = colnames[1:]
        index_col_vars['index_col'] = colnames[0]
        index_col_vars['index_col_values'] = raw_data[ colnames[0] ]
        raw_data = raw_data[index_col_vars['non_index_cols']]

    high_var_df = remove_low_variance_features(raw_data, variance_thresh)

    if has_index_col: 
        high_var_df[ index_col_vars['index_col'] ] = index_col_vars['index_col_values']
        new_columns = [ *index_col_vars['index_col'], *index_col_vars['non_index_cols'] ]
        high_var_df = high_var_df[ new_columns ]

    outfile_name = write_high_var_to_file(high_var_df, variance_thresh, input_file)

    return outfile_name
