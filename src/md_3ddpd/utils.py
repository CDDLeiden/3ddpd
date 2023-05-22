import os
import glob
from pathlib import Path
import json
import shutil
from re import sub
import datetime

import pandas as pd
from requests import get

import mdtraj as md
from mdtraj import load
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
# from chembl_webresource_client.new_client import new_client

from .definitions import get_colors

######################################################################################################################
## Preprocessing
######################################################################################################################
def rename_gpcrmd_wt_files(gpcrmd_dir: str, list_targets: list = None):
    """
    Rename wt files downloaded from GPCRmd to make them suitable for the 3DDPD pipeline:
                    'gpcrmdID_TargetName_Mutation_(replicate).extension'

    The files for each target are stored in a folder with the target name (e.g. 'aa1r_human').
    Write out a README_rename.txt file with the original and new names.
    :param gpcrmd_dir: path to the directory containing target subdirectories
    :param list_targets: (optional) list of targets to rename files for
    :return: None
    """
    if list_targets is None:
        # Define targets available in the GPCRmd directory
        list_targets = [os.path.basename(x) for x in glob.glob(os.path.join(gpcrmd_dir, '*')) if '_human' in x]

    for target in list_targets:
        # Define files available for each target downloaded from GPCRmd
        # Trajectories from GPCRmd have 'trj' tag and systems 'dyn' tag
        files = [os.path.basename(x) for x in glob.glob(os.path.join(gpcrmd_dir, target, '*')) if (('trj' in x) or ('dyn' in x))]
        # Define tags for entry name
        target_name = target.replace('_human', '')
        gpcrmd_id = Path(files[0]).stem.split('_')[-1]
        replicate_ids = [Path(file).stem.split('_')[0] for file in files if '.xtc' in file]
        mutant = 'wt'

        entry_name = f'{gpcrmd_id}_{target_name}_{mutant}'

        # Define new names
        rename_dict = {}
        for file in files:
            if ('.psf' in file) or ('.pdb' in file):
                new_name = f'{entry_name}.{file.split(".")[-1]}'
            elif '.xtc' in file:
                for i,replicate_id in enumerate(replicate_ids):
                    if replicate_id in file:
                        new_name = f'{entry_name}_{str(i+1)}.{file.split(".")[-1]}'
            else:
                 new_name = file

            # Rename files
            os.rename(os.path.join(gpcrmd_dir,target,file),os.path.join(gpcrmd_dir,target,new_name))
            rename_dict[file] = new_name

        # Write file with changes to names
        file_rename = os.path.join(gpcrmd_dir, target, 'README_rename.txt')
        df = pd.DataFrame.from_dict(rename_dict.items())
        df.columns = ['gpcrmd_name', 'new_name']

        if os.path.isfile(file_rename):
            df.to_csv(file_rename, sep='\t', mode='a', header=False) # Append to existing file
        else:
            df.to_csv(file_rename, sep='\t') # Create new file


def rename_gpcrmd_mutant_files(gpcrmd_dir: str, gpcrmd_mutant_dir: str, replicate_ids: list = [1], list_targets: list = None):
    """
    Rename mutant files generated using GPCRmd pipeline to make them suitable for the 3DDPD pipeline:
                    'gpcrmdID_TargetName_Mutation_(replicate).extension'
    And move them to the main gpcrmd_dir file.

    The files for each target are stored in a folder with the following tree structure:
        gpcr_mutant_dir
        L___ simulation_output
             L___ production
                  L___ target_gpcrmd_apo
                       L___ MUTANT
                            L___ rep_1

    Write out a README_rename.txt file with the original and new names.
    :param gpcrmd_dir: path to the directory containing target subdirectories
    :param gpcrmd_mutant_dir: path to the top directory containing gpcrmd simulation pipeline output data
    :param list_targets: (optional) list of targets to rename files for
    :param replicate_ids: trajectory replicates to rename and move
    :return: None
    """
    production_dir = os.path.join(gpcrmd_mutant_dir, 'simulation_output', 'production')
    # Read dictionary that was used to describe gpcrmd inputs for mutant simulation
    with open(os.path.join(gpcrmd_mutant_dir, 'inputs.json')) as json_file:
        inputs_dict = json.load(json_file)
        inputs_pdbfiles = [x['pdbfile'] for x in inputs_dict]

    if list_targets is None:
        # Define targets available in the GPCRmd directory
        list_targets = [os.path.basename(x) for x in glob.glob(os.path.join(gpcrmd_dir, '*')) if '_human' in x]

    for target in list_targets:

        # Define mutants available for each target in the list
        mutants = [os.path.basename(x) for x in glob.glob(os.path.join(production_dir, target.replace('human', 'gpcrmd_apo'), '*'))]
        if len(mutants) > 0:

            # Define tags for entry name
            target_name = target.replace('_human', '')
            gpcrmd_id = [x.split("_")[-1].replace('.pdb', '') for x in inputs_pdbfiles if target_name in x][0]

            rename_dict = {}
            for mutant in mutants:
                entry_name = f'{gpcrmd_id}_{target_name}_{mutant}'

                # .psf file is the same for all replicates
                file_psf = 'structure.psf'
                new_name_psf = f'{entry_name}.psf'

                # Move coordinate file
                replicate_dir = f"rep_{str(replicate_ids[0])}"
                dir_psf = os.path.join(production_dir, target.replace("human", "gpcrmd_apo"), mutant, replicate_dir)
                if not os.path.isfile(os.path.join(gpcrmd_dir, target, new_name_psf)):
                    try:
                        shutil.copy(os.path.join(dir_psf, file_psf), os.path.join(gpcrmd_dir, target, new_name_psf))

                        rename_dict[
                            f'production/{target.replace("human", "gpcrmd_apo")}/{mutant}/{replicate_dir}/{file_psf}'] = new_name_psf
                    except FileNotFoundError:
                        print(f'PSF File {os.path.join(dir_psf, file_psf)} does not exists. Skipping...')
                else:
                    print(f'File {os.path.join(gpcrmd_dir, target, new_name_psf)} exists. Skipping...')

                for replicate in replicate_ids:
                    # Define new names
                    replicate_dir = f'rep_{str(replicate)}'

                    file_xtc = 'output.xtc' # In theory it should be output_wrapped.xtc but trajectory is broken then
                    new_name_xtc = f'{entry_name}_{str(replicate)}.xtc'

                    # Move trajectory file
                    dir_xtc = os.path.join(production_dir, target.replace("human", "gpcrmd_apo"), mutant, replicate_dir)
                    if not os.path.isfile(os.path.join(gpcrmd_dir, target, new_name_xtc)):
                        try:
                            shutil.copy(os.path.join(dir_xtc, file_xtc), os.path.join(gpcrmd_dir, target, new_name_xtc))

                            rename_dict[f'production/{target.replace("human", "gpcrmd_apo")}/{mutant}/{replicate_dir}/{file_xtc}'] = new_name_xtc
                        except FileNotFoundError:
                            print(f'XTC File {os.path.join(dir_xtc, file_xtc)} does not exists. Skipping...')

                    else:
                        print(f'File {os.path.join(gpcrmd_dir, target, new_name_xtc)} exists. Skipping...')

            # Write file with changes to names
            file_rename = os.path.join(gpcrmd_dir, target, 'README_rename.txt')
            df = pd.DataFrame.from_dict(rename_dict.items())
            if not df.shape[0] == 0: # Empty dictionary if all files have been skipped
                df.columns = ['gpcrmd_name', 'new_name']

                if os.path.isfile(file_rename):
                    df.to_csv(file_rename, sep='\t', mode='a', header=False)  # Append to existing file
                else:
                    df.to_csv(file_rename, sep='\t')  # Create new file

######################################################################################################################
## MDtraj
######################################################################################################################
def read_trajectory(user_pathw,entry,slice: int = None):
    """"
    Reads MD trajectories of format .xtc or .dtr
    """
    gpcrmd_id = entry.split('_')[0]
    target = f"{entry.split('_')[1]}_human"
    mutant = '_'.join(entry.split('_')[:-1])
    replicate = entry.split('_')[-1]

    topology_file = os.path.join(user_pathw, target, f'{mutant}.psf')

    trajectory_file = os.path.join(user_pathw, target, f'{entry}.xtc')

    if os.path.exists(trajectory_file):
        # Read GPCRmd files (.xtc)
        traj_full = load(trajectory_file, top=topology_file)

        # Keep only trajectory for heavy atoms as otherwise from the .gro format residues are read in duplicate
        # and in any case we do not calculate 3DDPDs from H atoms
        heavy = traj_full.topology.select('element != H')
        traj_heavy = traj_full.atom_slice(heavy)

        if slice is not None:
            # Slice first N from MD trajectory (e.g. 1000 frames (200 ns) to be comparable with Desmond trajectories)
            traj_slice = traj_heavy[:slice]
        else:
            traj_slice = traj_heavy

        return traj_slice

    else:
        # Read Desmond files (.dtr)
        topology_file = os.path.join(user_pathw, mutant, f'{mutant}.gro')
        trajectory_file = os.path.join(user_pathw, mutant, f'{mutant}_100_{replicate}',f'{mutant}_100_{replicate}_trj', 'clickme.dtr')
        traj_full = md.load_dtr(trajectory_file, top=topology_file)

        # Keep only trajectory for heavy atoms as otherwise from the .gro format residues are read in duplicate
        # and in any case we do not calculate 3DDPDs from H atoms
        heavy = traj_full.topology.select('element != H')
        traj_heavy = traj_full.atom_slice(heavy)

        if slice is not None:
            # Slice first N from MD trajectory
            traj_slice = traj_heavy[:slice]
        else:
            traj_slice = traj_heavy

        return traj_slice


######################################################################################################################
## GPCRdb
######################################################################################################################
class GPCRdb:
    """"
    Retrieves protein information from GPCRdb
    """

    def __init__(self, **kwargs):
        if 'uniprot_name' in kwargs:
            self.uniprot_name = kwargs['uniprot_name']
        if 'accession' in kwargs:
            self.accession = kwargs['accession']
        if 'pdb' in kwargs:
            self.pdb = kwargs['pdb']
        if 'family_slug' in kwargs:
            self.family_slug = kwargs['family_slug']
        if 'species' in kwargs:
            self.species = kwargs['species']
        if 'target_list' in kwargs:
            self.target_list = kwargs['target_list']

    def get_uniprot_name(self):
        """"
        Map accession code to Uniprot name
        """
        if self.accession:
            url = f'https://gpcrdb.org/services/protein/accession/{self.accession}/'
            prot_info = get(url).json()
            uniprot_name = prot_info['entry_name']

            return uniprot_name
        else:
            raise ValueError('Accession not provided')

    def get_accession(self):
        """"
        Map Uniprot name to accession code
        """
        if self.uniprot_name:
            url = f'https://gpcrdb.org/services/protein/{self.uniprot_name}/'
            prot_info = get(url).json()
            accession = prot_info['accession']

            return accession
        else:
            raise ValueError('Uniprot name not provided')

    def get_family_slug(self):
        """"
        Map Uniprot name to accession code
        """
        if self.uniprot_name:
            url = f'https://gpcrdb.org/services/protein/{self.uniprot_name}/'
            prot_info = get(url).json()
            family_slug = prot_info['family']

            return family_slug
        else:
            raise ValueError('Uniprot name not provided')

    def get_sequence(self):
        """"
        Map Uniprot name to protein sequence
        """
        if self.uniprot_name:
            url = f'https://gpcrdb.org/services/protein/{self.uniprot_name}/'
            prot_info = get(url).json()
            sequence = prot_info['sequence']

            return sequence
        else:
            raise ValueError('Uniprot name not provided')

    def get_gpcrdb_target(self):
        """"
        Map Uniprot name to GPCRdb target name (not always the same)
        """
        if self.uniprot_name:
            url = f'https://gpcrdb.org/services/protein/{self.uniprot_name}/'
            prot_info = get(url).json()
            gpcrdb_target = []

            for a in prot_info['name']:
                if a == ' ':
                    gpcrdb_target.append('_')
                elif a != '&' and a != ';':
                    gpcrdb_target.append(a)

            gpcrdb_target = sub("[<].*?[>]", "", ''.join(gpcrdb_target))

            if prot_info['species'] == 'Homo sapiens':
                gpcrdb_target += '_human'
            else:
                print('Attention: non-human proteins selected')

            return gpcrdb_target
        else:
            raise ValueError('Uniprot name not provided')

    def get_pdb_entries(self):
        """"
        Retrieve list of PDB codes of structures available for a protein (Uniprot name)
        """
        if self.uniprot_name:
            url = f'https://gpcrdb.org/services/structure/protein/{self.uniprot_name}/'
            pdb_codes = get(url).json()
            if type(pdb_codes) is not list:
                pdb_codes = [pdb_codes]

            return pdb_codes
        else:
            raise ValueError('Uniprot name not provided')

    def get_segment_dictionary(self):
        """"
        Retrieve dictionary of residues in GPCR segments for a protein (Uniprot name)
        """
        if self.uniprot_name:
            url = f'https://gpcrdb.org/services/residues/extended/{self.uniprot_name}/'
            extended = get(url).json()

            segment_dict = {}

            for residue in extended:
                sequence_n = residue['sequence_number']
                segment = residue['protein_segment']

                segment_dict[sequence_n] = segment

            return segment_dict
        else:
            raise ValueError('Uniprot name not provided')


    def get_pdb_interactions(self):
        """"
        Retrieve list of interacting residues in pdb structure with co-crystalized ligand
        """
        if self.pdb:
            url = f'https://gpcrdb.org/services/structure/{self.pdb}/interaction/'
            aa_binding_pocket = get(url).json()

            return aa_binding_pocket
        else:
            raise ValueError('PDB code not provided')


    def get_family_targets(self):
        """"
        Retrieve list of Uniprot names of all GPCRs in a family as defined by its slug
        """
        if self.family_slug:
            url = f'https://gpcrdb.org/services/proteinfamily/proteins/{self.family_slug}/'
            if self.species:
                url = f'https://gpcrdb.org/services/proteinfamily/proteins/{self.family_slug}/{self.species}/'

            target_list = []
            targets = get(url).json()
            for target in targets:
                target_list.append(target['entry_name'])

            return target_list

        else:
            raise ValueError('Protein family slug not provided')

    def get_MSA(self):
        """
        Compute multiple sequence alignment (MSA) for two or more GPCRs
        Input: Comma-separated list of proteins (Uniprot names) e.g. 'adrb2_human,5ht2a_human'
        :return: MSA dictionary
        """
        if self.target_list:
            url = f'https://gpcrdb.org/services/alignment/protein/{self.target_list}/'
            msa = get(url).json()

            return msa

        else:
            raise ValueError('List of Uniprot names for MSA not provided')

def map_msa_segment(target,segment_dict,msa_dict):
    """"
    Map to MSA a dictionary of residues in GPCR segments for a protein (Uniprot name)
    """
    dict = segment_dict
    msa = msa_dict[target]

    msa_segment = []
    j = 1
    for i, pos in enumerate(msa):
        if pos == '-':
            pos1 = '-'
        else:
            pos1 = dict[j]
            j += 1
        msa_segment.append(pos1)

    return msa_segment

######################################################################################################################
## ChEMBL
######################################################################################################################
# def get_approval_status(chembl_id_list):
#     """"
#     Retrieve from ChEMBL API approval status (i.e. max_phase) for molecules in list defined by chembl_id
#     """
#     chembl_approval_dict = {}
#
#     # Keep first ID when an entry has multiple (common in Papyrus)
#     chembl_id_list_clean = [chembl_id.split(';')[0] for chembl_id in chembl_id_list]
#     molecule = new_client.molecule
#
#     # Execute in chunks to avoid max memory size error
#     chunksize = 1000
#     chembl_id_list_chunks = [chembl_id_list[i * chunksize:(i + 1) * chunksize] for i in range((len(chembl_id_list)
#                                                                                                + chunksize - 1) // chunksize)]
#     chembl_id_list_clean_chunks = [chembl_id_list_clean[i * chunksize:(i + 1) * chunksize] for i in
#                                    range((len(chembl_id_list_clean) + chunksize - 1) // chunksize)]
#
#     for list,clean_list in zip(chembl_id_list_chunks,chembl_id_list_clean_chunks):
#         approval_list = molecule.filter(molecule_chembl_id__in=clean_list).only(['molecule_chembl_id', 'max_phase'])
#         approval_dict = {}
#
#         for i,mol in enumerate(list):
#             try:
#                 approval_dict[mol] = approval_list[i]['max_phase']
#             except TypeError:
#                 approval_dict[mol] = np.NaN
#
#         # Add to final dictionary
#         chembl_approval_dict.update(approval_dict)
#
#     return chembl_approval_dict

def get_approved_drugs():
    """"
    Retrieve list of approved drugs (i.e. max_phase=4) as chembl_id
    """
    # Check if list is already available
    approved_file = './PCM_modelling/ChEMBL30_approved_drugs.txt'
    if not os.path.isfile(approved_file):
        molecule = new_client.molecule
        approved_drugs = molecule.filter(max_phase=4).only('molecule_chembl_id', 'max_phase')
        approved_drugs.set_format(frmt='json')

        approved_drugs_chembl_id = []
        for mol in approved_drugs:
            approved_drugs_chembl_id.append(mol['molecule_chembl_id'])

        # Write list to file
        with open(approved_file, 'w') as f:
            f.write(approved_drugs_chembl_id)

    else:
        approved_drugs = pd.read_csv(approved_file, sep='\t')
        approved_drugs_chembl_id = approved_drugs.molecule_chembl_id.to_list()
        # with open(approved_file, 'r') as f:
        #     approved_drugs_chembl_id = f.read()

    return approved_drugs_chembl_id

###################################################################################################################
## PLOTTING
###################################################################################################################
def dendrogram(
        X,
        plot_signals,
        hierarchy_metric='euclidean',
        linkage_method='single',
        title=None,
        colors=False,
        colors_dict=None,
        xmax=None,
        ticks_dict=None,
        cmap_name='hsv',
        save_fig=False,
        save_path=f'dendrogram_{datetime.datetime.today().strftime("%d-%m-%Y")}.png',
        **kwargs
):
    """Displays a dendrogram of hierarchical clustering between descriptors.

    :param X: array_like, An m by n array of m original observations in
     an n-dimensional space.
    :param plot_signals: array_like, signals to be plotted in dendrogram
    :param hierarchy_metric: str or callable, The distance metric to use.
     Additional info on scipy.spacial.distance.pdist
    :param linkage_method: str, The following are methods for calculating
     the distance between the newly formed clusters. Additional info:
     scipy.cluster.hierarchy.linkage
    :param title: str (optional), Title for the plot
    :param colors: bool (optional), Add color to signal plots.
     If n_transformations > 1, a separate color is picked for
     1 + n_transformations signals
    :param xmax: int or float (optional), Maximum distance value to be plotted
    :param ticks_dict: dict (optional), dictionary of names for each signal
    :param cmap_name: str (optional), color map to be used for signal colors
    :param save_fig: bool, saves the output plot to save_path
    :param save_path: str, path to saved figure
    :param kwargs: additional arguments for hierarchy_metric
    :return:
    """

    dist = pdist(X, metric=hierarchy_metric, **kwargs)
    sim_matrix = squareform(dist)

    truncate_mode = None
    distance_sort = False
    count_sort = False
    orientation = 'right'
    Z = hierarchy.linkage(dist, method=linkage_method)

    # plot dendrogram to obtain index number order of each signal
    fig, ax = plt.subplots()
    hierarchy.dendrogram(
        Z,
        ax=ax,
        orientation=orientation,
        distance_sort=distance_sort,
        truncate_mode=truncate_mode,
        count_sort=count_sort
    )

    positions = ax.get_yticklabels()
    ordered_labels = [int(pos.get_text()) for pos in positions]
    ordered_labels.reverse()
    plt.close()

    # plot dendrogram with original signals
    n_signals = len(sim_matrix)
    fig, ax = plt.subplots(n_signals, 1, figsize=(10, n_signals*0.5))
    fig.set_facecolor('w')

    if colors:
        # define the number of colors to use
        if ticks_dict is not None:
            # use one color per target as defined in dictionary
            targets_list = [f"{entry.split('_')[1]}_human" for entry in ticks_dict.values()]
            c = [colors_dict[t] for t in targets_list]
        else:
            n_colors = n_signals
            cmap = plt.cm.get_cmap(cmap_name, n_colors)
            c = [cmap(i) for i in range(n_colors)]

        color_dict = {i: color for i, color in enumerate(c)}

    # plot every signal individually in stacked manner
    for i, signal_idx in enumerate(ordered_labels):

        if colors:
            color = color_dict[signal_idx]
            ax[i].plot(plot_signals[signal_idx], color=color)

        else:
            ax[i].plot(plot_signals[signal_idx])

        ax[i].set_ylabel(f'{signal_idx}')
        ax[i].axis('off')

    # set additional axis measures
    if ticks_dict is not None:
        left = 1.15
    else:
        left = 0.99
    left, bottom, width, height = [left, 0, 0.5, 1]

    # add dendrogram next to signals
    add_ax = fig.add_axes([left, bottom, width, height])
    hierarchy.dendrogram(
        Z,
        ax=add_ax,
        orientation=orientation,
        distance_sort=distance_sort,
        truncate_mode=truncate_mode,
        count_sort=count_sort,
        # color_threshold=50
        # above_threshold_color='grey'
    )
    add_ax.set_xlabel('Distance')
    if xmax is not None:
        add_ax.set_xlim(0, xmax)

    # add custom ticks
    if ticks_dict is not None:
        ordered_labels.reverse()
        # ticks = [ticks_dict[label] for label in ordered_labels]
        ticks = [" ".join(ticks_dict[label].split("_")[1:3]) for label in ordered_labels]
        add_ax.set_yticklabels(ticks, rotation=0, fontsize=12)

    plt.title(title, fontsize=15)

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_3ddpd_dendrogram(entry_list: list, desc_dir: str, desc_name: str, save:bool):
    """
    :param entry_list: list of entry names to include in dendrogram
    :param desc_dir: directory where 3DDPD files are located
    :param desc_name: name of the descriptor to plot
    """
    # Read 3DDPD descriptor
    df = pd.read_csv(os.path.join(desc_dir, f'{desc_name}.txt'), sep='\t', index_col=0)
    # Keep entries from list
    df = df[df.index.isin(entry_list)]
    # keep only features that are common for all targets
    df_nozero = df.loc[:, (df != 0).all(axis=0)]
    # Extract entry names for plotting
    entry_dict = {i: n for i, n in enumerate(df.index.tolist())}
    # Plot descriptors
    # dendrogram(df_nozero.values, df.values, colors=True, colors_dict=get_colors(),ticks_dict=entry_dict, save_fig=save,
    #            save_path=os.path.join(desc_dir,f'dendrogram_{desc_name}.svg'))
    dendrogram(df.values, df.values, colors=True, colors_dict=get_colors(), ticks_dict=entry_dict, save_fig=save,
               save_path=os.path.join(desc_dir, f'dendrogram_{desc_name}.svg'))

def plot_rmsf_dendrogram(entry_list: list, md_analysis_dir: str, save:bool, alias:str, **md_options:dict):
    """
    :param entry_list: list of entry names to include in dendrogram
    :param md_dir: directory where MD files are located
    :param msa_file: path to MSA file to use for aligning the RMSF results
    """
    # Define file to read based on options
    rmsf_output_file = 'RMSF.txt'
    if ('normalize' in md_options) and (md_options['normalize'] == True):
        rmsf_output_file = rmsf_output_file.replace('.', f'_N.')
    if ('plot_align' in md_options) and (md_options['plot_align'] == True):
        rmsf_output_file = rmsf_output_file.replace('.', f'_A.')

    # Read RMSF dictionary
    with open(os.path.join(md_analysis_dir, rmsf_output_file)) as json_file:
        rmsf_dict = json.load(json_file)

    rmsf = pd.DataFrame.from_dict(rmsf_dict,orient='index')

    # Keep only entries in list
    rmsf = rmsf[rmsf.index.isin(entry_list)]
    # keep only features that are common for all targets
    rmsf_nozero = rmsf.loc[:, (rmsf != 0).all(axis=0)]
    # Extract entry names for plotting
    rmsf_entry_dict = {i: n for i, n in enumerate(rmsf.index.tolist())}
    # Plot RMSF
    # dendrogram(rmsf_nozero.values, rmsf.values, colors=True, colors_dict=get_colors(),ticks_dict=rmsf_entry_dict)
    dendrogram(rmsf.values, rmsf.values, colors=True, colors_dict=get_colors(),ticks_dict=rmsf_entry_dict, save_fig=save,
               save_path=os.path.join(md_analysis_dir,f'dendrogram_{rmsf_output_file.replace(".txt","")}_{alias}.svg'))