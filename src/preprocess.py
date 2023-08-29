import argparse

from preprocessing.correlation_helpers import create_correlation_list
from preprocessing.network_helpers import (
    extract_representatives_and_save_to_files,
    remove_representatives_from_main_dataset_and_save)
from preprocessing.variance_helpers import remove_low_var_and_save

def get_arguments():
    """
    Extracts command line arguments. 
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--infile', type=str, dest='infile', required=True,
                        help='the base input file dataframe')
    parser.add_argument('--has_indices', dest='has_indices', action='store_true',
                       help='signifies that dataset does not have indices')
    parser.add_argument('--corr_thresh', dest='corr_thresh', action='store', default=0.95,
                        help='the threshold at which to cut off values. Default 0.95')
    parser.add_argument('--save_corr', dest='save_corr', action='store_true',
                        help='saves the correlation data to file')
    parser.add_argument('--remove_high_corr', dest='remove_high_corr', action='store_true',
                        help='removes highly correlated values from the dataset.')
    parser.add_argument('--cv_thresh', dest='cv_thresh', action='store', default=0.05,
                        help='the minimal threshold of a coefficient of variance to keep: mu/sigma')
    parser.add_argument('--remove_low_variance', dest='remove_low_variance', action='store_true',
                        help="removes low variance elements using the cv_thresh from data and saves.")
    parser.add_argument('--outfile', dest='outfile', action='store', default='preprocessed.tsv',
                        help='the base name for the output files. Default is preprocessed.tsv')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='prints verbosely')

    return parser.parse_args()


def main():
    """
    Main Function
    """
    args = get_arguments()

    input_file = args.infile
    has_indices = args.has_indices
    remove_high_corr = args.remove_high_corr
    corr_thresh = args.corr_thresh
    cv_thresh = args.cv_thresh
    remove_low_variance = args.remove_low_variance
    save_corr = args.save_corr
    verbose = args.verbose


    if verbose:
        print(f"""Arguments Supplied:
        input_file: {input_file}
        has_indices: {has_indices}
        corr_thresh: {corr_thresh}
        save_corr: {save_corr}
        verbose: {verbose}
        """)

    if verbose: 
        print("Removing Low Variance Values")
    

    if remove_low_variance:
        # save low variance, then use that low variance file as base input.
        input_file = remove_low_var_and_save(input_file, cv_thresh, has_indices)


    if verbose: 
        print("Creating Correlation List.")

    if remove_high_corr:
        stacked_corr_data = create_correlation_list(input_file, has_indices, corr_thresh, save_corr)

        if verbose: 
            print("Extracting Representatives.") 

        representatives, non_representatives = extract_representatives_and_save_to_files(
            df = stacked_corr_data,
            original_data_file = input_file
        ) 

        if verbose: 
            print("Saving Dataset with Nonrepresentatives Removed")

        # TODO: Change the name! we're removing nonreps
        remove_representatives_from_main_dataset_and_save(input_file, non_representatives) 
if __name__ == "__main__":
    main()