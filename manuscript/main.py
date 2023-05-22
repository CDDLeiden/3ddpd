from .utils import get_directories
from src.md_3ddpd.utils import plot_3ddpd_dendrogram,plot_rmsf_dendrogram
from src.md_3ddpd import MD_analysis,StructBindingPocket,DynDescriptor
from src.modelling.prodec_descriptors import write_prodec_protein_descriptors
from src.modelling.PCM_papyrus_modelling import *
from src.modelling.PCM_papyrus_modelling_analysis import *

########################################################################################################################
# 0. Set directories and define entries of interest for the manuscript
md_dir, md_analysis_dir, desc_dir, desc_dir_dend, pcm_dir, PAPYRUS_DIR = get_directories('./directories.json')

entries_gpcrmd_wt = '87_5ht1b_wt_1,92_5ht2b_wt_1,165_aa1r_wt_1,49_aa2ar_wt_1,154_acm1_wt_1,111_acm2_wt_1,157_acm4_wt_1,11_adrb2_wt_1,189_agtr1_wt_1,118_ccr5_wt_1,163_cnr1_wt_1,101_cxcr4_wt_1,105_drd3_wt_1,158_ednrb_wt_1,75_ffar1_wt_1,108_hrh1_wt_1,184_lpar1_wt_1,73_oprd_wt_1,59_oprk_wt_1,155_oprx_wt_1,186_ox1r_wt_1,91_ox2r_wt_1,179_p2ry1_wt_1,77_p2y12_wt_1,128_par1_wt_1,63_s1pr1_wt_1'

entries_gpcrmd_mutants = '165_aa1r_wt_1,49_aa2ar_wt_1,111_acm2_wt_1,11_adrb2_wt_1,118_ccr5_wt_1,' \
                         '49_aa2ar_S91A_1,49_aa2ar_S277A_1,49_aa2ar_M177A_1,49_aa2ar_N253A_1,49_aa2ar_L85A_1,49_aa2ar_L167A_1,49_aa2ar_N181A_1,49_aa2ar_I66A_1,49_aa2ar_Y271A_1,49_aa2ar_T88D_1,' \
                         '165_aa1r_T277A_1,165_aa1r_R296C_1,165_aa1r_R291C_1,' \
                         '11_adrb2_S204A_1,11_adrb2_N293L_1,11_adrb2_S203A_1,11_adrb2_D130N_1,11_adrb2_D79N_1,11_adrb2_V317A_1,' \
                         '118_ccr5_Y108A_1,' \
                         '111_acm2_D103E_1,111_acm2_D103N_1,111_acm2_V421L_1'

########################################################################################################################
#                                            A) WT G protein-coupled receptors
########################################################################################################################
# A.0 Extract GPCRdb class A pre-computed MSA for targets in the 3DDPD set
#-----------------------------------------------------------------------------------------------------------------------
# Define binding pocket options
bp_options = {
    'target_input': '3ddpd_set',
    'hierarchy': 'gpcrdbA',
    'species_input': 'Homo sapiens',
    'output_type': ['d'], # Full-sequence MSA
    'precision': 90,
    'allosteric': 0,
    'target_input_alias': '3ddpd_set',
    'output_dir': desc_dir
}

# Compute binding pocket and extract output
bp = StructBindingPocket.BindingPocket(**bp_options)
bp.get_output()

# NOTE: The generated MSA with the above code ('MSA_full_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json')
# is equivalent to the json file provided under 3ddpd/data/ as 3ddpd_MSA.json for the tutorial

########################################################################################################################
# A.1. Analyze MD trajectories
#-----------------------------------------------------------------------------------------------------------------------
# Define options for RMSF calculation and plotting
md_options_wt = {
    'MSA_file': os.path.join(desc_dir, 'MSA_full_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json'),
    'plot_align': True, # Align RMSF to GPCdb MSA of reference
    'plot_segments': True, # Plot TM domains as segments for reference
    'save_plot': True,
    'output_dir': md_analysis_dir
}

# Calculate and plot RMSF aligned to full sequence GPCRdb(A) MSA
wt_traj = MD_analysis.MDTrajectory(md_dir,entries_gpcrmd_wt,**md_options_wt)
wt_traj.plot_rmsf_2col(alias='3ddpd_wt')

########################################################################################################################
# A.2. Compute binding pocket selections for MD analysis and 3DDPD calculation
#-----------------------------------------------------------------------------------------------------------------------
# Define binding pocket options
bp_options = {
    'target_input': '3ddpd_set',
    'hierarchy': 'target',
    'species_input': 'Homo sapiens',
    'output_type': ['a','b','c','d','e'],
    'precision': 90,
    'allosteric': 0,
    'target_input_alias': '3ddpd_set',
    'output_dir': desc_dir
}
hierarchy_options = ['None','gpcrdbA','family','subfamily','target']

for hierarchy in hierarchy_options:
    # Add hierarchy to binding pocket options
    bp_options['hierarchy'] = hierarchy

    # Compute binding pocket and extract output
    bp = StructBindingPocket.BindingPocket(**bp_options)
    bp.get_output()

########################################################################################################################
# A.3. Calculate 3DDPDs
#-----------------------------------------------------------------------------------------------------------------------
# Define starting arguments for rs3DDPD optimization
rs3ddpd_options = {'frame_split': 100,
                   'user_flex': 'all',
                   'sel_atoms': 'all',
                   'sel_residues': False,
                   'pca_explain': 0.99,
                   'numberpc': 5}

ps3ddpd_options = {'frame_split': 100,
                   'user_flex': 'all',
                   'sel_atoms': 'all',
                   'sel_residues': False,
                   'pca_explain': 0.95}

bp_options = {
    'MSA_input_file': os.path.join(desc_dir,'MSA_full_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json'),
    'BP_input_file': os.path.join(desc_dir,'BP_MDtraj_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json')
}

# NOTE: Optimization steps were defined with sequential PCM performance checks
# Optimization step 1: Trajectory data
for user_flex in ['all', 'std']:
    DynDescriptor.rs3ddpd_generation(md_dir = md_dir,
                                     desc_dir = desc_dir,
                                     input_entries = entries_gpcrmd_wt,
                                     input_alias = '3ddpd',
                                     frame_split = rs3ddpd_options['frame_split'],
                                     user_flex = user_flex,
                                     sel_atoms = rs3ddpd_options['sel_atoms'],
                                     sel_residues = rs3ddpd_options['sel_residues'],
                                     pca_explain = rs3ddpd_options['pca_explain'],
                                     numberpc = rs3ddpd_options['numberpc'],
                                     other_desc = None,
                                     **bp_options)

    DynDescriptor.ps3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=ps3ddpd_options['frame_split'],
                                     user_flex=user_flex,
                                     sel_atoms=ps3ddpd_options['sel_atoms'],
                                     sel_residues=ps3ddpd_options['sel_residues'],
                                     pca_explain=ps3ddpd_options['pca_explain'],
                                     **bp_options)

# Replace Trajectory data option with best results (from PCM) for rest of optimization
rs3ddpd_options['user_flex'] = 'std'

# Optimization step 2: Frame split (rs3DDPD)
for frame_split in [10,50,100,500]:
    DynDescriptor.rs3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=frame_split,
                                     user_flex=rs3ddpd_options['user_flex'],
                                     sel_atoms=rs3ddpd_options['sel_atoms'],
                                     sel_residues=rs3ddpd_options['sel_residues'],
                                     pca_explain=rs3ddpd_options['pca_explain'],
                                     numberpc=rs3ddpd_options['numberpc'],
                                     other_desc=None,
                                     **bp_options)

# Optimization step 3: Residue PCA (rs3DDPD) / Atom PCA coverage (ps3DDPD)
for numberpc in [3,5,7,10]:
    DynDescriptor.rs3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=rs3ddpd_options['frame_split'],
                                     user_flex=rs3ddpd_options['user_flex'],
                                     sel_atoms=rs3ddpd_options['sel_atoms'],
                                     sel_residues=rs3ddpd_options['sel_residues'],
                                     pca_explain=rs3ddpd_options['pca_explain'],
                                     numberpc=numberpc,
                                     other_desc=None,
                                     **bp_options)

for pca_explain in [0.95,0.99]:
    DynDescriptor.ps3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=ps3ddpd_options['frame_split'],
                                     user_flex=ps3ddpd_options['user_flex'],
                                     sel_atoms=ps3ddpd_options['sel_atoms'],
                                     sel_residues=ps3ddpd_options['sel_residues'],
                                     pca_explain=pca_explain,
                                     **bp_options)

# Optimization step 4: Atom selection
for sel_atoms in ['all','nonC']:
    DynDescriptor.rs3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=rs3ddpd_options['frame_split'],
                                     user_flex=rs3ddpd_options['user_flex'],
                                     sel_atoms=sel_atoms,
                                     sel_residues=rs3ddpd_options['sel_residues'],
                                     pca_explain=rs3ddpd_options['pca_explain'],
                                     numberpc=rs3ddpd_options['numberpc'],
                                     other_desc=None,
                                     **bp_options)

    DynDescriptor.ps3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=ps3ddpd_options['frame_split'],
                                     user_flex=ps3ddpd_options['user_flex'],
                                     sel_atoms=sel_atoms,
                                     sel_residues=ps3ddpd_options['sel_residues'],
                                     pca_explain=ps3ddpd_options['pca_explain'],
                                     **bp_options)

# Optimization step 5: Residue selection
for hierarchy in ['gpcrdbA','family','subfamily','target']:
    bp_options = {
        'MSA_input_file': os.path.join(desc_dir,f'MSA_full_3ddpd_set_Homo '
                                                f'sapiens_{hierarchy}_precision90_allosteric0.json'),
        'BP_input_file': os.path.join(desc_dir,f'BP_MDtraj_3ddpd_set_Homo sapiens_'
                                               f'{hierarchy}_precision90_allosteric0.json')
    }
    DynDescriptor.rs3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=rs3ddpd_options['frame_split'],
                                     user_flex=rs3ddpd_options['user_flex'],
                                     sel_atoms=rs3ddpd_options['sel_atoms'],
                                     sel_residues=True,
                                     pca_explain=rs3ddpd_options['pca_explain'],
                                     numberpc=rs3ddpd_options['numberpc'],
                                     other_desc=None,
                                     **bp_options)

    DynDescriptor.ps3ddpd_generation(md_dir=md_dir,
                                     desc_dir=desc_dir,
                                     input_entries=entries_gpcrmd_wt,
                                     input_alias='3ddpd',
                                     frame_split=ps3ddpd_options['frame_split'],
                                     user_flex=ps3ddpd_options['user_flex'],
                                     sel_atoms=ps3ddpd_options['sel_atoms'],
                                     sel_residues=True,
                                     pca_explain=ps3ddpd_options['pca_explain'],
                                     **bp_options)


# Plot optimized descriptors (best performance on PCM)
for descriptor_name in ['3ddpd_set_3DDPD_RS_std_f100_pc5_fs_aa', '3ddpd_set_3DDPD_PS_all_f100_pc0.95_fs_aa']:
    descriptor = DynDescriptor.Descriptor(desc_dir,descriptor_name,desc_dir)
    descriptor.plot_all_entries(subset=None, title_var='target', save=True)

########################################################################################################################
# A.4 Calculate classical protein descriptors for benchmark
#-----------------------------------------------------------------------------------------------------------------------
PAPYRUS_VERSION = '05.5'

# Define mapping of Uniprot names to Accession (hard-coded because of issues with the Uniprot client)
target_dict = {'5ht1b_human': 'P28222',
               '5ht2b_human': 'P41595',
               'aa1r_human': 'P30542',
               'aa2ar_human': 'P29274',
               'acm1_human': 'P11229',
               'acm2_human': 'P08172',
               'acm4_human': 'P08173',
               'adrb2_human': 'P07550',
               'agtr1_human': 'P30556',
               'ccr5_human': 'P51681',
               'cnr1_human': 'P21554',
               'cxcr4_human': 'P61073',
               'drd3_human': 'P35462',
               'ednrb_human': 'P24530',
               'ffar1_human': 'O14842',
               'hrh1_human': 'P35367',
               'lpar1_human': 'Q92633',
               'oprd_human': 'P41143',
               'oprk_human': 'P41145',
               'oprx_human': 'P41146',
               'ox1r_human': 'O43613',
               'ox2r_human': 'O43614',
               'p2ry1_human': 'P47900',
               'p2y12_human': 'Q9H244',
               'par1_human': 'P25116',
               's1pr1_human': 'P21453'
                 }

write_prodec_protein_descriptors(PAPYRUS_DIR = PAPYRUS_DIR,
                                 PAPYRUS_VERSION = PAPYRUS_VERSION,
                                 target_dict = target_dict,
                                 protein_descriptors = ['Zscale van Westen', 'Zscale Hellberg', 'PhysChem',
                                                        'MS-WHIM', 'STscale'],
                                 output_dir=os.path.join(pcm_dir, 'protein_descriptors'))

########################################################################################################################
# A.5. Run PCM benchmark models
#-----------------------------------------------------------------------------------------------------------------------
# Define options for models
split_types = ['random', 'Year']
split_options = [0.2, 2013]

# Define protein descriptors for optimization and benchmark
protein_descriptors_benchmark = ['Zscale_Hellberg', 'Zscale_van_Westen', 'STscale', 'MS-WHIM', 'PhysChem', 'unirep']
protein_descriptors_rs3ddpd = ['3DDPD_RS_all_f100_pc5_fs_aa', '3DDPD_RS_std_f10_pc5_fs_aa',
                               '3DDPD_RS_std_f50_pc5_fs_aa', '3DDPD_RS_std_f100_pc3_fs_aa',
                               '3DDPD_RS_std_f100_pc5_fs_aa', '3DDPD_RS_std_f100_pc5_fs_rc',
                               '3DDPD_RS_std_f100_pc5_g_aa', '3DDPD_RS_std_f100_pc7_fs_aa',
                               '3DDPD_RS_std_f100_pc10_fs_aa', '3DDPD_RS_std_f500_pc5_fs_aa']
protein_descriptors_ps3ddpd = ['3DDPD_PS_all_f100_pc95_fs_aa', '3DDPD_PS_std_f100_pc95_fs_aa',
                               '3DDPD_PS_std_f100_pc99_f_rc', '3DDPD_PS_std_f100_pc99_fs_aa',
                               '3DDPD_PS_std_f100_pc99_fs_rc', '3DDPD_PS_std_f100_pc99_g_rc',
                               '3DDPD_PS_std_f100_pc99_i_rc', '3DDPD_PS_std_f100_pc99_sf_rc']

protein_descriptors = protein_descriptors_benchmark + protein_descriptors_rs3ddpd + protein_descriptors_ps3ddpd

# Synergistic descriptors generated after pointing out best performing 3DDPDs from the optimization phase
protein_descriptors_rs3ddpd_sinergy = ['3DDPD_RS_std_f100_pc5_fs_aa_ms', '3DDPD_RS_std_f100_pc5_fs_aa_st',
                                       '3DDPD_RS_std_f100_pc5_fs_aa_z3']
protein_descriptors_rs3ddpd_concat = ['3DDPD_RS_std_f100_pc5_fs_aa_MS-WHIM', '3DDPD_RS_std_f100_pc5_fs_aa_PhysChem',
                                      '3DDPD_RS_std_f100_pc5_fs_aa_STscale',
                                      '3DDPD_RS_std_f100_pc5_fs_aa_Zscale_Hellberg',
                                      '3DDPD_RS_std_f100_pc5_fs_aa_Zscale_van_Westen']
protein_descriptors_ps3ddpd_concat = ['3DDPD_PS_all_f100_pc95_fs_aa_MS-WHIM', '3DDPD_PS_all_f100_pc95_fs_aa_PhysChem',
                                      '3DDPD_PS_all_f100_pc95_fs_aa_STscale',
                                      '3DDPD_PS_all_f100_pc95_fs_aa_Zscale_Hellberg',
                                      '3DDPD_PS_all_f100_pc95_fs_aa_Zscale_van_Westen']

protein_descriptors_synergy = protein_descriptors_rs3ddpd_sinergy + protein_descriptors_rs3ddpd_concat + protein_descriptors_ps3ddpd_concat

# Format 3DDPD descriptors for Papyrus modelling
for descriptor_name in protein_descriptors_rs3ddpd + protein_descriptors_ps3ddpd:
    format_3ddpd_entry(desc_dir, os.path.join(pcm_dir, 'protein_descriptors'),descriptor_name,target_dict)

# Concatenate descriptors for synergy
classical_descriptors = ['MS-WHIM', 'PhysChem', 'STscale', 'Zscale_Hellberg', 'Zscale_van_Westen']
best_performing_3ddpds = ['3DDPD_RS_std_f100_pc5_fs_aa', '3DDPD_PS_all_f100_pc95_fs_aa']
concatenate_descriptors(os.path.join(pcm_dir, 'protein_descriptors'), classical_descriptors, best_performing_3ddpds)

# Create dataset
bioactivity_data, target_data = create_datasets(papyrus_dir = PAPYRUS_DIR,
                                                papyrus_version = PAPYRUS_VERSION,
                                                target_dict = target_dict,
                                                output_dir = f'{pcm_dir}/datasets')

# Build benchmark QSAR models
build_qsar_models(papyrus_dir = PAPYRUS_DIR,
                  dataset = bioactivity_data,
                  split_types = split_types,
                  split_options = split_options,
                  output_dir = f'{pcm_dir}/models/qsar')

# Build PCM models (optimization and benchmark)
build_pcm_models(papyrus_dir=PAPYRUS_DIR,
                 dataset=bioactivity_data,
                 split_types=split_types,
                 split_options=split_options,
                 targets_data=target_data,
                 msa_file=os.path.join(pcm_dir, 'descriptors', 'benchmark_msa.fasta'),
                 protein_descriptors=protein_descriptors,
                 output_dir=f'{pcm_dir}/models/pcm')

# Build PCM models (synergy)
build_pcm_models(papyrus_dir=PAPYRUS_DIR,
                 dataset=bioactivity_data,
                 split_types=split_types,
                 split_options=split_options,
                 targets_data=target_data,
                 msa_file=os.path.join(pcm_dir, 'descriptors', 'benchmark_msa.fasta'),
                 protein_descriptors=protein_descriptors_synergy,
                 output_dir=f'{pcm_dir}/models/pcm')

########################################################################################################################
# A.5. Analyze PCM results
#-----------------------------------------------------------------------------------------------------------------------
def get_results_to_plot(protein_descriptor_set):
    # Read and tabulate the results
    results = tabulate_results(os.path.join(pcm_dir, 'models'), protein_descriptor_set)
    # List data of all seeds for plotting
    QSAR, PCM, aggregated = list_all_seeds(results)
    # Return QSAR + PCM aggregated seed results for plotting
    return aggregated

# Plot performance for an individual metric OPTIMIZATION - (boxplots)
RS_optimization_strategy = {'Trajectory data': [('3DDPD_RS_all_f100_pc5_fs_aa', 'Coordinate'),('3DDPD_RS_std_f100_pc5_fs_aa', 'Rigidity')],
                            'Frame split': [('3DDPD_RS_std_f10_pc5_fs_aa','10 frames'),('3DDPD_RS_std_f50_pc5_fs_aa', '50 frames'),('3DDPD_RS_std_f100_pc5_fs_aa', '100 frames'),('3DDPD_RS_std_f500_pc5_fs_aa','500 frames')],
                            'Residue PCA': [ ('3DDPD_RS_std_f100_pc3_fs_aa', '3 PCs'),('3DDPD_RS_std_f100_pc5_fs_aa', '5 PCs'),('3DDPD_RS_std_f100_pc7_fs_aa', '7 PCs'),('3DDPD_RS_std_f100_pc10_fs_aa', '10 PCs')],
                            'Atom selection': [('3DDPD_RS_std_f100_pc5_fs_aa', 'All heavy atoms'),('3DDPD_RS_std_f100_pc5_fs_rc', 'Minus C')],
                            'Residue selection': [('3DDPD_RS_std_f100_pc5_fs_aa', 'Full sequence'),('3DDPD_RS_std_f100_pc5_g_aa','GPCRdb (class A)')]}
PS_optimization_strategy = {'Trajectory data': [('3DDPD_PS_all_f100_pc95_fs_aa', 'Coordinate'),('3DDPD_PS_std_f100_pc95_fs_aa', 'Rigidity')],
                            'Atom PCA coverage': [('3DDPD_PS_all_f100_pc95_fs_aa', '95% variance'),('3DDPD_PS_all_f100_pc99_fs_aa', '99% variance')],
                            'Atom selection': [('3DDPD_PS_all_f100_pc95_fs_aa', 'All heavy atoms'),('3DDPD_PS_all_f100_pc95_fs_rc', 'Minus C')],
                            'Residue selection': [('3DDPD_PS_all_f100_pc95_fs_aa', 'Full sequence'),('3DDPD_PS_all_f100_pc95_g_aa', 'GPCRdb (class A)'),('3DDPD_PS_all_f100_pc95_f_aa','GPCRdb family'),('3DDPD_PS_all_f100_pc95_sf_aa', 'GPCRdb subfamily'),('3DDPD_PS_all_f100_pc95_i_aa', 'Target')]}


plot_optimization_performance(get_results_to_plot(protein_descriptors_ps3ddpd), split='year',
                              desc_filter=protein_descriptors_ps3ddpd, optimization_strategy=PS_optimization_strategy,
                              desc_set='PS_optimization', variable = 'r', plot_significance=True, save=True,
                              output_format = 'svg',output_dir=os.path.join(pcm_dir,'analysis'))
plot_optimization_performance(get_results_to_plot(protein_descriptors_rs3ddpd), split='year',
                              desc_filter=protein_descriptors_rs3ddpd, optimization_strategy=RS_optimization_strategy,
                              desc_set='RS_optimization', variable = 'r', plot_significance=True, save=True,
                              output_format = 'svg', output_dir=os.path.join(pcm_dir,'analysis'))


# Plot benchmark performance heatmaps for optimized 3DDPDs (pair-wise statistical significance as calculated with an
# independent T test ) - Random and temporal split; classification and regression
protein_descriptors_MS = ['QSAR'] + protein_descriptors_benchmark + ['3DDPD_RS_std_f100_pc10_fs_aa'] + ['3DDPD_PS_all_f100_pc95_fs_aa']

for split_type,metric in [('random','MCC'),('random','r'),('year','MCC'),('year','r')]:
    stat_significance_heatmap(get_results_to_plot(protein_descriptors_MS), split_type, metric,
                              protein_descriptors_MS, desc_set='benchmark',save=True)

# Plot performance of combinations of descriptors (Temporal split, regression)
synergy_names = {'Zscale_Hellberg': 'Zscale Hellberg',
              'Zscale_van_Westen': 'Zscale van Westen',
              'STscale': 'STscale',
              'MS-WHIM': 'MS-WHIM',
              'PhysChem': 'PhysChem',
              'unirep': 'Unirep',
              '3DDPD_RS_std_f100_pc5_fs_aa': 'rs3DDPD',
              '3DDPD_PS_all_f100_pc95_fs_aa': 'ps3DDPD',
              '3DDPD_RS_std_f100_pc5_fs_aa_Zscale_van_Westen': 'rs3DDPD + Zscale vW',
              '3DDPD_PS_all_f100_pc95_fs_aa_Zscale_van_Westen': 'ps3DDPD + Zscale vW',
              '3DDPD_RS_std_f100_pc5_fs_aa_Zscale_Hellberg': 'rs3DDPD + Zscale H',
              '3DDPD_PS_all_f100_pc95_fs_aa_Zscale_Hellberg': 'ps3DDPD + Zscale H',
              '3DDPD_RS_std_f100_pc5_fs_aa_STscale': 'rs3DDPD + STscale',
              '3DDPD_PS_all_f100_pc95_fs_aa_STscale': 'ps3DDPD + STscale',
              '3DDPD_RS_std_f100_pc5_fs_aa_PhysChem': 'rs3DDPD + PhysChem',
              '3DDPD_PS_all_f100_pc95_fs_aa_PhysChem': 'ps3DDPD + PhysChem',
               '3DDPD_RS_std_f100_pc5_fs_aa_MS-WHIM': 'rs3DDPD + MS-WHIM',
              '3DDPD_PS_all_f100_pc95_fs_aa_MS-WHIM': 'ps3DDPD + MS-WHIM'

}

plot_performance(get_results_to_plot(protein_descriptors_synergy), split='year', desc_filter=protein_descriptors_synergy ,
                 desc_names=synergy_names, desc_set='synergy', variable = 'r', save=True, output_format='png',
                 output_dir=os.path.join(pcm_dir,'analysis'))

# Plot feature importance: Best performing rs3DDPD and ps3DDPD, also best combination (temporal split, regression)
plot_feature_importance(model_dir=os.path.join(pcm_dir, 'models', 'pcm'), model_type='regression',
                        protein_descriptor='3DDPD_RS_std_f100_pc5_fs_aa', split_by='Year', top=25, save=True,
                        output_dir=os.path.join(pcm_dir,'analysis'))
plot_feature_importance(model_dir=os.path.join(pcm_dir, 'models', 'pcm'), model_type='regression',
                        protein_descriptor='3DDPD_PS_all_f100_pc95_fs_aa', split_by='Year', top=25, save=True,
                        output_dir=os.path.join(pcm_dir,'analysis'))
plot_feature_importance(model_dir=os.path.join(pcm_dir, 'models', 'pcm'), model_type='regression',
                        protein_descriptor='3DDPD_RS_std_f100_pc5_fs_aa_Zscale_van_Westen', split_by='Year', top=25,
                        save=True,output_dir=os.path.join(pcm_dir,'analysis'))
plot_feature_importance(model_dir=os.path.join(pcm_dir, 'models', 'pcm'), model_type='regression',
                        protein_descriptor='3DDPD_PS_all_f100_pc95_fs_aa_PhysChem', split_by='Year', top=50,
                        save=True,output_dir=os.path.join(pcm_dir,'analysis'))
plot_feature_importance(model_dir=os.path.join(pcm_dir, 'models', 'pcm'), model_type='regression',
                        protein_descriptor='Zscale_van_Westen', split_by='Year', top=25, save=True,
                        output_dir=os.path.join(pcm_dir,'analysis'))

########################################################################################################################
#                                      B) Mutant G protein-coupled receptors
########################################################################################################################
# B.2. Analyze MD trajectories
#-----------------------------------------------------------------------------------------------------------------------
# Define RMSF calculation and plotting options
md_options_mut = {
    'MSA_file': os.path.join(desc_dir,'MSA_full_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json'),
    'normalize':True, # Normalize RMSF (i.e. substract from a reference)
    'normalize_entry':'wt_1', # Normalize the RMSF values respect to the WT RMSF of the first replicate
    'save_plot':True,
    'plot_align':True,
    'plot_segments':True,
    'plot_normalize':True,
    'plot_mutation':True,
    'output_dir': md_analysis_dir
}

# Calculate and plot RMSF aligned to full sequence GPCRdb(A) MSA
mut_traj = MD_analysis.MDTrajectory(md_dir,entries_gpcrmd_mutants,**md_options_mut)
mut_traj.plot_rmsf(alias='3ddpd_mut_1')

# Plot RMSF in dendrogram
md_options_mut_dendrogram_1 = {
    'normalize': False,
    'plot_align':True
}
md_options_mut_dendrogram_2 = {
    'normalize':True,
    'plot_align':True
}
plot_rmsf_dendrogram(entries_gpcrmd_mutants.split(','), md_analysis_dir, True, 'gpcr_mut_PANELS_1', **md_options_mut_dendrogram_1)
plot_rmsf_dendrogram(entries_gpcrmd_mutants.split(','), md_analysis_dir, True, 'gpcr_mut_PANELS_2', **md_options_mut_dendrogram_2)

########################################################################################################################
# B.3. Calculate 3DDPDs
#-----------------------------------------------------------------------------------------------------------------------
bp_options = {
    'MSA_input_file': os.path.join(desc_dir,'MSA_full_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json'),
    'BP_input_file': os.path.join(desc_dir,'BP_MDtraj_3ddpd_set_Homo sapiens_gpcrdbA_precision90_allosteric0.json')
}

DynDescriptor.rs3ddpd_generation(md_dir = md_dir,
                                 desc_dir = desc_dir,
                                 input_entries = entries_gpcrmd_mutants,
                                 input_alias = 'mut_set',
                                 frame_split = 100,
                                 user_flex = 'std',
                                 sel_atoms = 'all',
                                 sel_residues = False,
                                 pca_explain = 0.99,
                                 numberpc = 5,
                                 other_desc = None,
                                 **bp_options)

DynDescriptor.ps3ddpd_generation(md_dir = md_dir,
                                 desc_dir = desc_dir,
                                 input_entries = entries_gpcrmd_mutants,
                                 input_alias = 'mut_set',
                                 frame_split = 100,
                                 sel_atoms = 'all',
                                 sel_residues = False,
                                 pca_explain = 0.95,
                                 **bp_options)

for descriptor_name in ['mut_set_3DDPD_RS_std_f100_pc5_fs_aa', 'mut_set_3DDPD_PS_all_f100_pc0.95_fs_aa']:
    descriptor = DynDescriptor.Descriptor(desc_dir,descriptor_name,desc_dir)
    descriptor.plot_all_entries_one_col(subset=None, title_var='entry', save=True)
    plot_3ddpd_dendrogram(entries_gpcrmd_mutants.split(','),desc_dir_dend,descriptor_name,True)





