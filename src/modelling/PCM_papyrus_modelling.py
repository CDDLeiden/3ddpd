# Import dependencies
import os

import joblib
import pandas as pd

import Bio.SeqIO as Bio_SeqIO
import prodec
import papyrus_scripts
from papyrus_scripts.modelling import qsar, pcm
from papyrus_scripts.preprocess import keep_accession, keep_quality, keep_protein_class, consume_chunks
from papyrus_scripts.reader import read_papyrus, read_protein_set
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


def format_3ddpd_entry(input_dir, output_dir, descriptor_name, target_dict):
    """
    Format 3DDPD dataframe to be used in Papyrus modelling scripts.
    The 3DDPD dataframe is saved with the first column as "entry", but Papyrus modelling takes target_id
    A Uniprot name-accession dictionary is needed
    :param input_dir: Path to file containing 3DDPD dataframe
    :param output_dir: Path to directory to write formated descriptors
    :param descriptor_name: Name of the 3DDPD
    :param target_dict: Disctionary mapping Uniprot names to Accession codes
    """
    descriptor_df = pd.read_csv(f'{input_dir}/{descriptor_name}.txt', sep=' ')

    descriptor_df['entry'] = descriptor_df['entry'].apply(lambda x: target_dict[f'{x.split("_")[1]}_human'] + '_WT')
    descriptor_df.rename(columns={'entry': 'target_id'}, inplace=True)
    descriptor_df.to_csv(f'{output_dir}/{descriptor_name}.txt', sep='\t', index=False)


def concatenate_descriptors(descriptors_dir: str, classical_descriptors: list, dynamic_descriptors: list):
    """
    Concatenate classical descriptors and 3DDPDs to test synergistic effect
    """
    merge_names = []
    # Read classical descriptors
    for desc in classical_descriptors:
        df_classical = pd.read_csv(os.path.join(descriptors_dir, f'protein_descriptor_{desc}.txt'), sep='\t',
                                   index_col=0)
        # Read 3DDPDs
        for dyn_desc in dynamic_descriptors:
            df_dynamic = pd.read_csv(os.path.join(descriptors_dir, f'{dyn_desc}.txt'), sep='\t')

            # Concatenate descriptors
            df_merge = pd.merge(df_classical, df_dynamic, on='target_id')

            # Define new descriptor name
            merge_name = f'{dyn_desc}_{desc}.txt'
            merge_names.append(
                merge_name.replace('.txt', ''))  # Append to list to be able to print them and copy them when needed

            # Write file with concatenated descriptors
            df_merge.to_csv(os.path.join(descriptors_dir, merge_name), sep='\t', index=False)

    print('Concatenated descriptors created:', merge_names)

def create_datasets(papyrus_dir: str,
                    papyrus_version: str,
                    target_dict: dict,
                    output_dir: str):
    """
    Create bioactivity and protein dataset from Papyrus for targets of interest
    """
    # Check if bioactivity file already exists
    bioactivity_file = f'{output_dir}/papyrus-{papyrus_version}_targets_bioactivity.txt'
    if os.path.isfile(bioactivity_file):
        bioactivity_data = pd.read_csv(bioactivity_file, sep='\t', index_col=0)
    else:
        
        # Read downloaded Papyrus dataset in chunks, as it does not fit in memory
        CHUNKSIZE = 100000
        data = read_papyrus(version=papyrus_version, chunksize=CHUNKSIZE, source_path=papyrus_dir)
        # Create filter for targets of interest
        target_accession_list = target_dict.values()
        filter = keep_accession(data, target_accession_list)
        # Iterate through chunks and apply the filter defined
        bioactivity_data = consume_chunks(filter, total=-(-papyrus_scripts.utils.IO.get_num_rows_in_file('bioactivities', False) // CHUNKSIZE))
        # Add Activity_class column with threshold at pchembl_value = 6.5 
        bioactivity_data['Activity_class'] = bioactivity_data['pchembl_value_Mean'].apply(lambda x: 0 if x < 6.5 else 1)
        # Write out file
        bioactivity_data.to_csv(bioactivity_file, sep='\t')
        
    # Check if protein file already exists
    protein_file = f'{output_dir}/papyrus-{papyrus_version}_targets.txt'
    if os.path.isfile(protein_file):
        targets_data = pd.read_csv(protein_file, sep='\t', index_col=0)
    else:
        # Define target IDs for mapping in Papyrus
        target_ids = [f'{accession}_WT' for accession in target_dict.values()]
        # Read target sequences 
        protein_data = papyrus_scripts.read_protein_set(version=papyrus_version, source_path=papyrus_dir)
        # Filter protein data for our targets of interest based on accession code
        targets_data = protein_data[protein_data.target_id.isin(target_ids)]
        # Write out file
        targets_data.to_csv(protein_file, sep='\t')

    return bioactivity_data,targets_data
    
def get_protein_descriptor(targets_data: pd.DataFrame,
                           msa_file: str,
                           descriptor_name: str,
                           output_dir: str):
    """
    Calculate or read path to ProDEC protein descriptor for targets of interest
    """
    # Check if file already exists
    descriptor_tag = descriptor_name.replace(' ', '_')
    descriptor_file = f'{output_dir}/protein_descriptors/protein_descriptor_{descriptor_tag}.txt'

    if os.path.isfile(descriptor_file):
        print(f'Reading protein descriptor {descriptor_name}')
    else:
        print(f'Calculating protein descriptor {descriptor_name}')
        # Read aligned sequences from MSA file
        aligned_sequences = [str(seq.seq) for seq in Bio_SeqIO.parse(msa_file, "fasta")]
        # Get protein descriptor from ProDEC
        prodec_descriptor = prodec.ProteinDescriptors().get_descriptor(descriptor_name)
        # Calculate descriptor features for aligned sequences of interest
        protein_features = prodec_descriptor.pandas_get(aligned_sequences)
        # Insert protein labels in the obtained features
        protein_features.insert(0, "target_id", targets_data.target_id.reset_index(drop=True))
        # Write out file
        protein_features.to_csv(descriptor_file, sep='\t')

    return descriptor_file
    

def build_qsar_models(papyrus_dir: str,
                     dataset: pd.DataFrame,
                     split_types: list, 
                     split_options: list, 
                     output_dir: str):
    """
    Train and validate base QSAR regression and classification models for all splitting methods.
    Use 10 different random seeds
    """       
    # Split options 
    for split_by, split_option in zip(split_types, split_options):
        if split_by == 'Year':
            year = split_option
            test_set_size = None
        elif split_by == 'random':
            test_set_size = split_option
            year = None 

        # Model type
        for model_type in ['regression', 'classification']:
            if model_type == 'regression':
                model = RandomForestRegressor(verbose=0)
                stratify = False
            else:
                model = RandomForestClassifier(verbose=0)
                stratify = True

            # Seeds
            for seed_n,seed in enumerate([1234, 2345, 3456, 4567, 5678, 6879, 7890, 8901, 9012, 9999]):

                try:
                    output = f'{output_dir}/QSAR_results_{model_type}_ECFP_{split_by}_{seed_n}.tsv'
                    modelname = f'{output_dir}/QSAR_model_{model_type}_ECFP_{split_by}_{seed_n}.joblib.xz'
                    # Train and validate QSAR models 
                    cls_results, cls_models = qsar(dataset,
                                                   split_by=split_by,
                                                   split_year=year,
                                                   random_state=seed,
                                                   descriptors='fingerprint',
                                                   descriptor_path=papyrus_dir,
                                                   verbose=True,
                                                   model=model,
                                                   stratify=stratify)
                    # Save validation results
                    cls_results.to_csv(output, sep='\t')
                    # Save models 
                    joblib.dump(cls_models, modelname, compress=('xz', 9), protocol=0)
                    
                except:
                    print('Skipping...')

def build_pcm_models(papyrus_dir: str,
                     dataset: pd.DataFrame,
                     split_types: list, 
                     split_options: list,
                     targets_data: pd.DataFrame,
                     msa_file: str,
                     protein_descriptors: list, 
                     output_dir: str):
    """
    Train and validate PCM regression and classification models for all combinations of descriptors and splitting methods possible.
    Use 10 different random seeds
    """
    # Split options
    for split_by, split_option in zip(split_types, split_options):
        if split_by == 'Year':
            year = split_option
            test_set_size = None
        elif split_by == 'random':
            test_set_size = split_option
            year = None 
        
        # Protein descriptors
        for protein_descriptor in protein_descriptors:
            if protein_descriptor == 'unirep':
                prot_descriptor_type = protein_descriptor
                descriptor_folder = papyrus_dir
            else:
                prot_descriptor_type = 'custom'
                if '3DDPD' in protein_descriptor:
                    descriptor_folder = f'{papyrus_dir}/protein_descriptors/{protein_descriptor}.txt'
                else:
                    descriptor_folder = get_protein_descriptor(targets_data,msa_file,protein_descriptor,papyrus_dir)
            
            # Model type
            for model_type in ['regression', 'classification']:
                if model_type == 'regression':
                    model = RandomForestRegressor(verbose=0)
                    stratify = False
                else:
                    model = RandomForestClassifier(verbose=0)
                    stratify = True

                # Seed
                for seed_n,seed in enumerate([1234, 2345, 3456, 4567, 5678, 6879, 7890, 8901, 9012, 9999]):
                    
                    try:
                        descriptor_tag = protein_descriptor.replace(' ', '_')
                        output = f'{output_dir}/PCM_results_{model_type}_ECFP_{descriptor_tag}_{split_by}_{seed_n}.tsv'
                        if not os.path.exists(output):
                            modelname = f'{output_dir}/PCM_results_{model_type}_ECFP_{descriptor_tag}_{split_by}_{seed_n}.joblib.xz'

                            # Train and validate PCM models
                            cls_results, cls_models = pcm(dataset,
                                                          split_by=split_by,
                                                          split_year=year,
                                                          random_state=seed,
                                                          mol_descriptors='fingerprint',
                                                          mol_descriptor_path=papyrus_dir,
                                                          prot_sequences_path=papyrus_dir,
                                                          prot_descriptors=prot_descriptor_type,
                                                          prot_descriptor_path=descriptor_folder,
                                                          verbose=True,
                                                          model=model,
                                                          stratify=stratify)
                            # Save validation results 
                            cls_results.to_csv(output, sep='\t')
                            # Save models 
                            joblib.dump(cls_models, modelname, compress=('xz', 9), protocol=0)
                        else:
                            print('Seed already trained. Skipping...')
                        
                    except:
                        print('Skipping...')


    
# if __name__ == '__main__':
#
#     # Define paths
#     PAPYRUS_DIR = '../PCM_modelling'
#     PAPYRUS_VERSION = '05.5'
#     msa_file =  f'{PAPYRUS_DIR}/msa.fasta'
#
#     # Define targets of interest
#     target_dict = {'5ht1b_human': 'P28222',
#                    '5ht2b_human': 'P41595',
#                    'aa1r_human': 'P30542',
#                    'aa2ar_human': 'P29274',
#                    'acm1_human': 'P11229',
#                    'acm2_human': 'P08172',
#                    'acm4_human': 'P08173',
#                    'adrb2_human': 'P07550',
#                    'agtr1_human': 'P30556',
#                    'ccr5_human': 'P51681',
#                    'cnr1_human': 'P21554',
#                    'cxcr4_human': 'P61073',
#                    'drd3_human': 'P35462',
#                    'ednrb_human': 'P24530',
#                    'ffar1_human': 'O14842',
#                    'hrh1_human': 'P35367',
#                    'lpar1_human': 'Q92633',
#                    'oprd_human': 'P41143',
#                    'oprk_human': 'P41145',
#                    'oprx_human': 'P41146',
#                    'ox1r_human': 'O43613',
#                    'ox2r_human': 'O43614',
#                    'p2ry1_human': 'P47900',
#                    'p2y12_human': 'Q9H244',
#                    'par1_human': 'P25116',
#                    's1pr1_human': 'P21453'
#                      }
#
#     # Define options for models
#     split_types = ['random', 'Year']
#     split_options = [0.2, 2013]
#
#     # Define protein descriptors for optimization and benchmark
#     protein_descriptors_benchmark = ['Zscale Hellberg', 'Zscale van Westen', 'STscale', 'MS-WHIM', 'PhysChem', 'unirep']
#     protein_descriptors_rs3ddpd = ['3DDPD_RS_all_f100_pc5_fs_aa', '3DDPD_RS_std_f10_pc5_fs_aa', '3DDPD_RS_std_f50_pc5_fs_aa', '3DDPD_RS_std_f100_pc3_fs_aa',
#                                    '3DDPD_RS_std_f100_pc5_fs_aa', '3DDPD_RS_std_f100_pc5_fs_rc', '3DDPD_RS_std_f100_pc5_g_aa', '3DDPD_RS_std_f100_pc7_fs_aa',
#                                    '3DDPD_RS_std_f100_pc10_fs_aa', '3DDPD_RS_std_f500_pc5_fs_aa']
#     protein_descriptors_ps3ddpd = ['3DDPD_PS_all_f100_pc95_fs_aa', '3DDPD_PS_std_f100_pc95_fs_aa', '3DDPD_PS_std_f100_pc99_f_rc', '3DDPD_PS_std_f100_pc99_fs_aa',
#                                    '3DDPD_PS_std_f100_pc99_fs_rc','3DDPD_PS_std_f100_pc99_g_rc', '3DDPD_PS_std_f100_pc99_i_rc', '3DDPD_PS_std_f100_pc99_sf_rc']
#
#     protein_descriptors = protein_descriptors_benchmark + protein_descriptors_rs3ddpd + protein_descriptors_ps3ddpd
#
#     # Synergistic descriptors generated after pointing out best performing 3DDPDs from the optimization phase
#     protein_descriptors_rs3ddpd_sinergy = ['3DDPD_RS_std_f100_pc5_fs_aa_ms','3DDPD_RS_std_f100_pc5_fs_aa_st','3DDPD_RS_std_f100_pc5_fs_aa_z3']
#     protein_descriptors_rs3ddpd_concat = ['3DDPD_RS_std_f100_pc5_fs_aa_MS-WHIM', '3DDPD_RS_std_f100_pc5_fs_aa_PhysChem', '3DDPD_RS_std_f100_pc5_fs_aa_STscale',
#                                           '3DDPD_RS_std_f100_pc5_fs_aa_Zscale_Hellberg', '3DDPD_RS_std_f100_pc5_fs_aa_Zscale_van_Westen']
#     protein_descriptors_ps3ddpd_concat = ['3DDPD_PS_all_f100_pc95_fs_aa_MS-WHIM', '3DDPD_PS_all_f100_pc95_fs_aa_PhysChem', '3DDPD_PS_all_f100_pc95_fs_aa_STscale',
#                                           '3DDPD_PS_all_f100_pc95_fs_aa_Zscale_Hellberg', '3DDPD_PS_all_f100_pc95_fs_aa_Zscale_van_Westen']
#
#     protein_descriptors_synergy = protein_descriptors_rs3ddpd_sinergy + protein_descriptors_rs3ddpd_concat + protein_descriptors_ps3ddpd_concat
#
#     # Create dataset
#     bioactivity_data, target_data = create_datasets(PAPYRUS_DIR, PAPYRUS_VERSION, target_dict, f'{PAPYRUS_DIR}/datasets')
#
#     # Build benchmark QSAR models
#     build_qsar_models(PAPYRUS_DIR, bioactivity_data, split_types, split_options, f'{PAPYRUS_DIR}/models/qsar')
#
#     # Build PCM models
#     build_pcm_models(PAPYRUS_DIR, bioactivity_data, split_types, split_options, target_data, msa_file, protein_descriptors, f'{PAPYRUS_DIR}/models/pcm')

    
    
    
    
