import glob
import os

def run_amber(path=None, cami_high=True, cami_medium=False, cami_airways=False):
    print("Running")
    path_dataset = '/mnt/d/datasets'
    results_path = '/mnt/c/Users/kuluk/Documents/Github/MI108F20_Binning/Embedding_cluster_tests/unnormalized_embedding'
    path = results_path

    directory_of_files = os.path.join(results_path, "*binning_results.tsv")
    labels = glob.glob(directory_of_files)
    print(labels)
    labels = [label.split('/')[-1].split('binning')[0] for label in labels]
    paths_to_results = [os.path.abspath(x) for x in glob.glob(directory_of_files)]
    if cami_high:
        gold_standard_file = os.path.abspath(os.path.join(path_dataset, 'cami_high', 'ground_truth_with_length.tsv'))
        unique_common_file = os.path.abspath(os.path.join( path_dataset, 'cami_high','unique_common.tsv'))
    elif cami_airways:
        gold_standard_file = os.path.abspath(os.path.join(path_dataset, 'cami_airways', 'ground_truth_with_length.tsv'))
        outdir_with_circular = os.path.join(path, 'amber_with_circular')
        command_amber_with_circular = f'amber.py -g {gold_standard_file} -l "{", ".join(labels)}" {" ".join(paths_to_results)} -o {outdir_with_circular}'
        os.system(command_amber_with_circular)
        return

    else:
        return
    outdir_with_circular = os.path.join(path, 'amber_with_circular')
    outdir_without_circular = os.path.join(path, 'amber_without_circular')
    command_amber_without_circular = f'amber.py -g {gold_standard_file} -l "{", ".join(labels)}" -r {unique_common_file} -k "circular element" {" ".join(paths_to_results)} -o {outdir_without_circular}'
    command_amber_with_circular = f'amber.py -g {gold_standard_file} -l "{", ".join(labels)}" {" ".join(paths_to_results)} -o {outdir_with_circular}'
    os.system(command_amber_with_circular)
    os.system(command_amber_without_circular)


if __name__ == '__main__':
    run_amber()