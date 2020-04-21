
import argparse
import os
from collections import defaultdict
from Bio import SeqIO


def main():
    args = handle_input_arguments()

    cont_file_path = args.contigs
    bin_folder_path = args.output_folder
    bin_file = args.bin_file

    bin_to_contig = defaultdict(list)

    # Create target Directory if don't exist
    if not os.path.exists(bin_folder_path):
        os.mkdir(bin_folder_path)
        print("Directory ", bin_folder_path, " Created ")
    else:
        print("Directory ", bin_folder_path, " already exists")

    # create entry in dict for all bins : binid -> list of contig ids
    with open(bin_file) as file:
        next(file)
        next(file)
        next(file)
        next(file)

        for line in file:
            line.replace('\n', '')
            content = line.split('\t')
            bin_id = content[1]
            contig_id = content[0]

            # handles both new bin ids and already existing contig ids
            bin_to_contig[bin_id].append(contig_id)

    # SeqIO creates dict of all contigs in contig file.
    contig_record_dict = SeqIO.to_dict(SeqIO.parse(cont_file_path, "fasta"))

    for k,v in contig_record_dict.items():
        print(f'key:{k} \n\n')
        print(f'value: {v}\n\n')
        print(f'type: {type(v)}')
        break

    print(f'contig_records len:  {len(contig_record_dict)}')
    print(f'bin_dict len:  {len(bin_to_contig)}')
    print(list(contig_record_dict.keys())[-5:])

    # for each bin id - create a file
    for bin_id, contig_list in bin_to_contig.items():
        record_list = []
        for contig in contig_list:
            record_list.append(contig_record_dict[contig])
        # write to file
        SeqIO.write(record_list, f'{bin_folder_path}/checkM_binID_{bin_id}.faa', 'fasta')

        '''
    for bin_id, contig_ids in contig_record_dict.items():

        # Get contig ids of a single bin
        bin_records = []
        for contig_id in contig_ids:
            bin_records.append(contig_id)

        # write bin to file following input path
        SeqIO.write(bin_records, f'{bin_folder_path}/checkM_{bin_file}_binID_{bin_id}.faa', 'fasta')
    '''


def handle_input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--contigs", required=True, help="Path to fasta file with contigs")

    parser.add_argument("-o", "--output_folder", required=True, help="Path to folder with output bin files")

    parser.add_argument("-b", "--bin_file", required=True, help="Path to bin file that maps contig ids to bin ids")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
