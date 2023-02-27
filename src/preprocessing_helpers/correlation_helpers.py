import numpy as np
import pandas as pd


### CORRELATION Functions
def correlate_data(df, has_index_col):
    """
    Correlates data. Depending on whether an index column exists,
    the first column is removed for correlation.

    This assumes entire matrix is numerical (minus the index column).
    """
    columns_for_correlation = df.columns[1:] if has_index_col else df
    return df[ columns_for_correlation ].corr()

def extract_correlates_to_upper_right(df):
    """
    Removes the duplicated values of the correlation matrix and
    additionally removes the diagonal of the matrix.

    this assumes the diagonal has all 1s
    """
    df = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
    return df - np.eye(len(df))

def stack_data(df, threshold):
    """
    This stacks data into a 1 to 1 of the correlation matrix.
    """
    df = df.stack().reset_index()
    df = df[ df[0] > threshold ]
    return df

def create_correlation_list(filename, has_indices, corr_thresh, save_corr):
    """
    This function assumes files are tsvs, correlates data, and thresholds the
    final correlation data. 

    Data are then saved to file. 
    """
    df = pd.read_csv(filename, sep='\t')
    df = correlate_data(df, has_indices)

    df = extract_correlates_to_upper_right(df)


    df = stack_data(df, corr_thresh)

    base_name = '.'.join(filename.split('.')[0:-1])

    outfile_name = f"{base_name}_correlation_over_{corr_thresh}.tsv"

    if save_corr: 
        df.to_csv(outfile_name, sep='\t', index=None)
        print(f"Correlation data saved @ {outfile_name}")
    
    return df

