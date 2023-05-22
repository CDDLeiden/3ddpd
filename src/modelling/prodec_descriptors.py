import os
import pandas as pd
import shutil

import Bio
import Bio.SeqIO as Bio_SeqIO

import prodec
from prodec import *
from papyrus_scripts.reader import read_protein_set

def get_target_data(PAPYRUS_DIR, PAPYRUS_VERSION, target_ids):
    """
    Extract target data from Papyrus
    :param PAPYRUS_DIR: Location to Papyrus download
    :param PAPYRUS_VERSION: Papyrus version
    :param target_ids: List of targets of interest as defined by target_id in Papyrus
    """
    # Read target sequences
    protein_data = read_protein_set(version=PAPYRUS_VERSION, source_path=PAPYRUS_DIR)
    # Filter protein data for our targets of interest based on accession code
    targets = protein_data[protein_data.target_id.isin(target_ids)]

    return targets

def calculate_msa(target_dataframe, output_dir):
    """
    Calculate MSA in fasta format for a dataframe of targets
    :param target_dataframe: Dataframe with Papyrus target data
    :param output_dir: Directory to write MSA file
    """
    # Create object with sequences and descriptions
    records = []
    for index, row in target_dataframe.reset_index(drop=True).iterrows():
        records.append(
            Bio_SeqIO.SeqRecord(
                seq=Bio.Seq.Seq(row["Sequence"]),
                id=str(index),
                name=row["target_id"],
                description=" ".join([row["UniProtID"], row["Organism"], row["Classification"]]),
            )
        )
    sequences_path = os.path.join(output_dir, "sequences.fasta")
    # Write sequences as .fasta file
    _ = Bio_SeqIO.write(records, sequences_path, "fasta")

    # Calculate MSA for proteins of interest
    # NOTE: ClustalO needs to be installed and it is only available on Linux
    os.system('!clustalo - i sequences.fasta - t Protein - o msa.fasta - -outfmt = fa')
    # move fasta file to output directory (by default it is written to the working directory)
    alignment_file = os.path.join(output_dir, 'benchmark_msa.fasta')
    shutil.move('msa.fasta', alignment_file)

    # Parse aligned sequences from fasta file
    aligned_sequences = [str(seq.seq) for seq in Bio.SeqIO.parse(alignment_file, "fasta")]

    return aligned_sequences

def calculate_protein_descriptor(target_dataframe, aligned_sequences, protein_descriptor):
    """
    Calculate protein descriptor of choice for aligned proteins of interest

    Parameters
    ----------
    target_dataframe : pandas.DataFrame
        Pandas dataframe with information about targets of interest
    aligned_sequences : list
        List of aligned sequences read from fasta file produced with Clustal Omega
    protein_descriptor : str
        Protein descriptor label as described in ProDEC

    Returns
    -------
    pandas.DataFrame
        Dataset with accession and features for the protein descriptor of interest for the targets in the input
    """
    # Get protein descriptor from ProDEC
    prodec_descriptor = prodec.ProteinDescriptors().get_descriptor(protein_descriptor)

    # Calculate descriptor features for aligned sequences of interest
    protein_features = prodec_descriptor.pandas_get(aligned_sequences)

    # Insert protein labels in the obtained features
    protein_features.insert(0, "target_id", target_dataframe.target_id.reset_index(drop=True))

    return protein_features

def write_prodec_protein_descriptors(PAPYRUS_DIR, PAPYRUS_VERSION, target_dict, protein_descriptors, output_dir):
    """
    Calculates and writes protein descriptors using ProDEC and an on-the-fly MSA
    :param PAPYRUS_DIR:
    :param PAPYRUS_VERSION:
    :param target_dict:
    :param protein_descriptors:
    :param output_dir:
    :return:
    """
    # Define target IDs for mapping in Papyrus
    target_ids = [f'{accession}_WT' for accession in target_dict.values()]

    # Read target data
    target_dataframe = get_target_data(PAPYRUS_DIR, PAPYRUS_VERSION, target_ids)

    # Calculate or parse MSA
    alignment_file = os.path.join(output_dir, 'benchmark_msa.fasta')
    if os.path.exists(alignment_file):
        # Parse aligned sequences from fasta file
        aligned_sequences = [str(seq.seq) for seq in Bio.SeqIO.parse(alignment_file, "fasta")]
    else:
        aligned_sequences = calculate_msa(target_dataframe, output_dir)

    # Calculate protein descriptors of interest
    for protein_descriptor in protein_descriptors:
        protein_descriptor_alias = protein_descriptor.replace(" ", '_')

        descriptor = calculate_protein_descriptor(target_dataframe, aligned_sequences, protein_descriptor)

        # Write descriptor dataframe to file for modelling
        descriptor.to_csv(f'protein_descriptor_{protein_descriptor_alias}.txt', sep='\t')





