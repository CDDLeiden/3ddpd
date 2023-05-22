# Load packages
from .definitions import *
from .utils import read_trajectory
from .StructBindingPocket import BindingPocket
import os
import json
from math import floor,ceil
import pandas as pd
import numpy as np
import itertools
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Import definitions
computed_gasteiger_charges = computed_gasteiger_charges()
other_descriptors = {'MSWHIM': MSWHIM(),
                     'STscales': STscales(),
                     'Zscales5': Zscales5(),
                     'Zscales3': Zscales3(Zscales5())}

# Define functions used to compute ps3DDPD and rs3DDPD descriptors
def descriptor_name(desc_type,input_alias,sel_atoms,sel_residues,frame_split,user_flex,pca_option,other_desc,**bp_kwargs):
    """
    Generate a file name to write the descriptor generation output
    """
    # Trajectory atom selection options
    if sel_atoms == 'all':
        atom_tag = 'aa' # All heavy atoms
    elif sel_atoms == 'nonC':
        atom_tag = 'rc' # All heavy atoms minus carbons

    # Trajectory selection options
    if not sel_residues:
        res_tag = 'fs' # Full sequence
    else:
        # Check BP selection options
        if 'BP_input_file' in bp_kwargs:
            hierarchy_tag = os.path.basename(bp_kwargs.get('BP_input_file')).split('_')[-3]
        elif 'hierarchy' in bp_kwargs:
            hierarchy_tag = bp_kwargs.get('hierarchy')
        else:
            raise ValueError('Residue selection is not full sequence but no hierarchy is defined. '
                             'Check binding pocket kwargs.')

        if hierarchy_tag in ['NoHierarchy', 'None']:
            raise ValueError('Residue selection is not full sequence but no hierarchy is defined. '
                             'Check binding pocket kwargs.')
        elif hierarchy_tag == 'gpcrdbA':
            res_tag = 'ga'
        elif hierarchy_tag == 'family':
            res_tag = 'f'
        elif hierarchy_tag == 'subfamily':
            res_tag = 'sf'
        elif hierarchy_tag == 'target':
            res_tag = 'i'

    # Build up name tag for descriptor
    if desc_type == 'RS':
        name_tag = f'{input_alias}_3DDPD_RS_{user_flex}_f{frame_split}_pc{pca_option}_{res_tag}_{atom_tag}' # pca_option == numberpc
        # Additional descriptor options
        if other_desc is not None:
            name_tag += f'_{"_".join(other_desc)}'
    elif desc_type == 'PS':
        name_tag = f'{input_alias}_3DDPD_PS_{user_flex}_f{frame_split}_pc{pca_option}_{res_tag}_{atom_tag}' # pca_option == pca_explain

    return name_tag

def parse_binding_pocket_selection(**bp_kwargs):
    """
    Read or calculate binding pocket residue selection and MSA
    :param bp_kwargs: dictionary with options for BPselect.BindingPocket class
    :return:
    """
    # Read or compute binding pocket MSA and residue selection
    if ('MSA_input_file' in bp_kwargs) and ('BP_input_file' in bp_kwargs):
        print('Reading MSA and binding pocket selection from json files...')
        try:
            with open(bp_kwargs.get('MSA_input_file'), 'r') as MSA_file:
                MSA_inputs = json.load(MSA_file)
        except:
            print('Valid MSA input file needed')
        try:
            with open(bp_kwargs.get('BP_input_file'), 'r') as BP_file:
                residue_inputs = json.load(BP_file)
        except:
            print('No binding pocket selection read')
    else:
        print('Calculating MSA and binding pocket selection...')
        bp = BindingPocket(**bp_kwargs)
        MSA_inputs, residue_inputs = bp.get_output()
    print('Done')
    return MSA_inputs,residue_inputs

def atom_selection(traj,sel_atoms):
    """
    Slice a trajectory by selecting only atom types of interest
    :param traj:
    :param sel_atoms: atom selection option
    :return:
    """
    atom_list_residue = []

    for atom_traj in traj._topology._atoms:
        # Use all heavy atoms
        if sel_atoms == 'all':
            if atom_traj.name in ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'OG', 'OG1', 'SG', 'CG2', 'CD', 'CD1', 'OD1',
                                  'ND1', 'SD', 'CD2', 'OD2', 'ND2', 'CE', 'CE1', 'OE1', 'NE', 'NE1', 'CE2', 'OE2',
                                  'NE2', 'CE3', 'CZ', 'CZ2', 'NZ', 'CZ3', 'CH2', 'OH', 'NH1', 'NH2']:
                atom_list_residue.append(atom_traj)
        # Use only non-Carbon heavy atoms
        else:
            if atom_traj.name in ['N', 'O', 'OG', 'OG1', 'SG', 'OD1', 'ND1', 'SD', 'OD2', 'ND2', 'OE1', 'NE', 'NE1',
                                  'OE2', 'NE2', 'NZ', 'OH', 'NH1', 'NH2']:
                atom_list_residue.append(atom_traj)

    print(
        f'Atom select MD size:    <mdtraj.Trajectory with {traj.n_frames} frames, {len(atom_list_residue)} atoms, {traj.n_residues} residues, and unitcells>')

    return atom_list_residue

def residue_selection(traj,target,residue_inputs):
    """
    Slice a trajectory by selecting only residue numbers of interest
    :param traj:
    :param target: uniprot name of target e.g. 'aa1r_human'
    :param residue_inputs:
    :return:
    """
    # Extract residue selection from dictionary
    residue_select = list(
        traj.topology.select(str(residue_inputs[target])))
    # Slice trajectory
    traj = traj.atom_slice(residue_select)

    return traj

def atom_data_framesplit(atom, frame_split, trajectory, gasteiger_list, user_flex):
    """
    Calculates atomic data properties given a framesplit
    """
    # Initialize
    start = 0
    end = frame_split
    coordinate_data = []
    atom_properties = []

    # Add Gasteriger partial charges per atom
    for i in gasteiger_list:
        if (atom.residue.name + '_' + atom.name) == i[1]:
            atom_properties.append(i[0])

    # Read XYZ coordinates for defined framesplit
    x_full = trajectory._xyz[:, atom.index, :][:, 0]
    y_full = trajectory._xyz[:, atom.index, :][:, 1]
    z_full = trajectory._xyz[:, atom.index, :][:, 2]
    while end <= trajectory.n_frames:
        x = x_full[start:end]
        y = y_full[start:end]
        z = z_full[start:end]
        if user_flex == 'std':
            # Calculate SD of XYZ coordinates
            coordinate_data.append([x.std(), y.std(), z.std()])
        else:
            # Calculate Mean, Median, and SD
            if frame_split == 1:
                coordinate_data.append([x, y, z]) # If frame split is 1 no statistics possible
            else:
                coordinate_data.append(
                    [x.mean(), pd.Series(x).median(), x.std(), y.mean(), pd.Series(y).median(), y.std(), z.mean(),
                     pd.Series(z).median(), z.std()])
        start = 0 + end
        end = start + frame_split

    # Return list of properties per atom
    for sublist in coordinate_data:
        for item in sublist:
            atom_properties.append(item)
    return atom_properties

def atom_pca(data_atoms,pca_explain):
    """
    Calculate PCA from atomic data grouped by one or all targets and return PCs explaining the defined variance
    :param data_atoms:
    :param pca_explain:
    :return:
    """
    # Scale atom data
    if isinstance(data_atoms, pd.DataFrame):
        data_atoms = StandardScaler().fit_transform(data_atoms)
    else:
        data_atoms = StandardScaler().fit_transform(DataFrame(data_atoms))

    # Compute PCA to explain variability defined
    pca_atoms = PCA(pca_explain)
    # Select number of PCs that satisfy the variability explanation
    pc_atoms = pca_atoms.fit_transform(data_atoms)

    print(f'Number of atom PCs:             {len(pca_atoms.explained_variance_ratio_)}')
    print(f'Explained variance per atom PC: {pca_atoms.explained_variance_ratio_}')
    print(f'Explained variance atom total:  {sum(pca_atoms.explained_variance_ratio_)}')

    return pca_atoms,pc_atoms


def add_descriptors(data_residues, traj, descriptor_tag_list):
    """
    Add other one-hot encoded descriptors to residue data prior to residue PCA
    :param data_residues:
    :param descriptor_list:
    :return:
    """
    # Extract AA names for residues to join classical descriptors
    res_codes = [residue.code for residue in traj._topology._residues]

    # Add AA names to atom PCA results aggregated per residue
    data_residues = DataFrame(data_residues).merge(DataFrame(res_codes), left_index=True, right_index=True)
    data_residues = data_residues.values.tolist()

    data_residues_extra_descriptors = []
    # Add extra descriptors' features to atom PCA data for each residue in the trajectory
    for data in data_residues:
        # Keep existing atom PCA data
        residue_features = data[:-1]
        # Read extra descriptors' features
        for descriptor_tag in descriptor_tag_list:
            descriptor = other_descriptors[descriptor_tag]
            for AA in descriptor:
                if data[-1] == AA[0]: # Check if AA name matches
                    residue_features.extend(AA[1:]) # Append the extra descriptors' features
        data_residues_extra_descriptors.append(residue_features)

    return data_residues_extra_descriptors

def residue_pca(data_residues, traj, numberpc):
    """
    Calculate PCA from residue data and return defined number of PCs
    :param residue_data:
    :param residue_pc_number:
    :return:
    """
    # Scale residue data
    data_residues = StandardScaler().fit_transform(
        data_residues)
    # Compute PCA and extract fixed number of PCs
    pca_residues = PCA(n_components=numberpc)
    pc_residues = pca_residues.fit_transform(data_residues)
    # Add residue labels to PCs extracted
    res_id = []
    for residue in traj._topology._residues:
        res_id.append((residue.code + str(residue.resSeq)))
    pc_residues = DataFrame(res_id).merge(DataFrame(pc_residues), left_index=True, right_index=True)

    print(f'Number of residue PCs:             {len(pca_residues.explained_variance_ratio_)}')
    print(f'Explained variance per residue PC: {pca_residues.explained_variance_ratio_}')
    print(f'Explained variance residue total:  {sum(pca_residues.explained_variance_ratio_)}')

    return pca_residues,pc_residues

def report_descriptor_stats(desc_type,desc_dir,name_tag,input_entries,user_flex,frame_split,sel_atoms,sel_residues,pca_options_dict,other_desc):
    """
    Write report with descriptor statistics
    :param desc_type:
    :param desc_dir:
    :param name_tag:
    :param input_entries:
    :param user_flex:
    :param frame_split:
    :param sel_atoms:
    :param sel_residues:
    :param pca_options_dict:
    :param other_desc:
    :return:
    """
    with open(os.path.join(desc_dir, f'{name_tag}.log'), 'w') as report_file:
        report_file.write('\n#####################################################################\n')
        report_file.write(f'SUMMARY - {name_tag}:\n')
        report_file.write(f'Entries/targets:           {input_entries}\n')
        if user_flex == 'all':
            report_file.write(' - Atom data:      Average/Median/SD x/y/z-coordinates + Gasteiger Charge.\n')
        elif user_flex == 'std':
            report_file.write(' - Atom data:      SD x/y/z-coordinates + Gasteiger Charge.\n')
        report_file.write(f' - Framesplit:     {frame_split} frames.\n')
        if desc_type == 'RS':
            report_file.write(f' - Atom PCs:       {pca_options_dict["pca_explain"] * 100}% variance explained\n')
            atompc_avg = DataFrame(pca_options_dict["atom_pc_len"]).mean(axis=0)
            atompc_avg = atompc_avg.round(0)
            atompc_std = DataFrame(pca_options_dict["atom_pc_len"]).std(axis=0)
            atompc_std = atompc_std.round(0)
            report_file.write(f'      Average # Atom PCs:                       {atompc_avg.values.tolist()}\n')
            report_file.write(f'      SD # Atom PCs:                            {atompc_std.values.tolist()}\n')
            report_file.write(f' - Residue PCs:    {pca_options_dict["numberpc"]} PCs.\n')
            variance_per_pc = DataFrame(pca_options_dict["variance_per_pc"]).mean(axis=0) * 100
            variance_per_pc = variance_per_pc.round(1)
            report_file.write(f'     Average explained variance per residue PC: {variance_per_pc.values.tolist()}\n')
            variance_avg = DataFrame(pca_options_dict["variance_total"]).mean(axis=0) * 100
            variance_avg = variance_avg.round(1)
            variance_std = DataFrame(pca_options_dict["variance_total"]).std(axis=0) * 100
            variance_std = variance_std.round(1)
            report_file.write(f'     Average explained variance total:          {variance_avg.values.tolist()}\n')
            report_file.write(f'     SD explained variance total:               {variance_std.values.tolist()}\n')

        elif desc_type == 'PS':
            report_file.write(f' - Atom PCs:       {len(pca_options_dict["pca_atoms"].explained_variance_)} PCs.\n')
            variance_per_pc = DataFrame(pca_options_dict["pca_atoms"].explained_variance_ratio_) * 100
            variance_per_pc = variance_per_pc.round(1)
            report_file.write(f'     Explained variance per atom PC: {variance_per_pc.values.tolist()}\n')
            variance_total = sum(pca_options_dict["pca_atoms"].explained_variance_ratio_) * 100
            variance_total = variance_total.round(1)
            report_file.write(f'     Explained variance total:       {variance_total}\n')

        if sel_atoms == 'all':
            report_file.write(' - Atom select:    All atoms.\n')
        elif sel_atoms == 'nonC':
            report_file.write(' - Atom select:    All non-Carbon atoms.\n')
        if not sel_residues:
            report_file.write(' - Residue select: All residues.\n')
        else:
            report_file.write(' - Residue select: Custom selection.\n')

        if desc_type == 'RS':
            if other_desc == None:
                report_file.write(' - Other descr.:   None.\n')
            else:
                report_file.write(f' - Other descr.:   {", ".join(other_desc)}\n')
        report_file.write('\n#####################################################################')


# Generate RS-3DDPDs
def rs3ddpd_generation(md_dir: str,
                       desc_dir: str,
                       input_entries: str,
                       input_alias: str,
                       sel_atoms: str,
                       sel_residues: bool,
                       frame_split: int,
                       user_flex: str,
                       pca_explain: float,
                       numberpc: int,
                       other_desc: str,
                       **bp_kwargs: dict):
    """
        Generates residue-specific (rs)3DDPDs

        :param md_dir: pathway to folder containing the MD files [str]
        :param desc_dir: pathway to folder to write 3DDPDs
        :param input_entries: comma-separated (no spaces) string of filenames to compute descriptors for [str].
                            Entry contains: RefID_target_mutation_replicate (e.g. 87_5ht1b_wt_1,87_5ht1b_W247K_1)
        :param input_alias: alias to identify in the output file name the set of input entries
        :param sel_atoms: choice of atoms to generate descriptors on ['all'/'nonC']
                        'all': All atoms
                        'nonC': All non-Carbon atoms
        :param sel_residues: choice of residues to generate descriptors on [True/False]
                        False: All residues
                        True: Selection of residues (eg: General GPCR class A binding pocket)
        :param frame_split: number of frames to be included in the frame_split [int]
        :param user_flex: choice of atom data option ['all'/'std']
                         'all': Average/Median/SD x/y/z-coordinates + Gasteiger Charge
                         'std' SD x/y/z-coordinates + Gasteiger Charge
        :param pca_explain: % of variance covered by atom PCA [float (0.0-1.0)]
        :param numberpc: number of principal components (PC) in residue PCA [int]
        :param other_desc: list with choice of additional descriptors to add prior to residue PCA
                        One or more of: ['MSWHIM','Stscales','Zscales5','Zscales3'] or None
        :kwargs options for BPselect BindingPocket.get_output()
        """
    # Parse binding pocket selection
    MSA_inputs,residue_inputs = parse_binding_pocket_selection(**bp_kwargs)

    # Output file initialization
    name_tag = descriptor_name('RS',input_alias,sel_atoms,sel_residues,frame_split,user_flex,numberpc,other_desc,**bp_kwargs)
    # Compute descriptor if output file does not exist
    descriptor_file = os.path.join(desc_dir, f'{name_tag}.txt')
    if not os.path.exists(descriptor_file):
        descriptor_dict = {}

        # Reporters initialization
        variance_per_pc = []
        variance_total = []
        atom_pc_len = []

        entries = input_entries.split(',')
        targets = [f"{entry.split('_')[1]}_human" for entry in entries]
        mutants = ['_'.join(entry.split('_')[1:3]) for entry in entries]

        # Initialize and generate one rs3DDPD per independent trajectory
        for num_entry,(entry,target,mutant) in enumerate(zip(entries,targets,mutants)):
        # Load MD trajectory and keep only protein
            print(f'\nNow processing: ({entry})')
            traj_full = read_trajectory(md_dir, entry)
            print(f'Total MD size:          {traj_full}')
            traj = traj_full.restrict_atoms(traj_full.topology.select("protein"))
            print(f'Protein MD size:        {traj}')

        # Select only residues of interest
            if sel_residues:
                traj = residue_selection(traj,target,residue_inputs)
            print(f'Residue select MD size: {traj}')

        # Select only atoms of interest
            atom_list_residue = atom_selection(traj,sel_atoms)

        # Atom data calculation for selection of interest
            data_atoms = []
            res_numbers = []

            for atom_res in atom_list_residue:
                data_atoms.append(atom_data_framesplit(atom_res, frame_split, traj, computed_gasteiger_charges, user_flex))
                # Specify residues for which atom data is calculated for later grouping
                res_numbers.append(atom_res.residue.resSeq)

        # Atom data PCA
            pca_atoms, pc_atoms = atom_pca(data_atoms,pca_explain)
            # Add data to reporter variable
            atom_pc_len.append(len(pca_atoms.explained_variance_ratio_))

        # Group atom PCA results per residue to prepare for residue PCA
            # Add residue labels to selected atom PCs
            pc_atoms = DataFrame(pc_atoms).merge(DataFrame(res_numbers), left_index=True, right_index=True)
            pc_atoms = pc_atoms.values.tolist()
            # Initialize data collection for residue PCA
            data_residues = []

            for residue in traj.topology._residues: # Updated to account for lisozyme residue numbers (in the thousands)
                # Extract atom PCA results corresponding to this residue number
                atoms_res = [atom_pc[:-1] for atom_pc in pc_atoms if atom_pc[-1] == residue.resSeq]

                if atoms_res != []:
                    # Aggregate atom PCA results per residue and calculate statistics to keep
                    atoms_res = DataFrame(atoms_res)
                    data_res_stats = [atoms_res.mean(axis=0).values.tolist(), atoms_res.median(axis=0).values.tolist(), atoms_res.std(axis=0).values.tolist()]
                    # Flatten aggregated list to satisfy input conditions for residue PCA
                    data_res_stats_flat = [i for j in data_res_stats for i in j]
                    data_residues.append(data_res_stats_flat)

        # Add other descriptors to residue data prior to residue PCA
            if other_desc is not None:
                data_residues = add_descriptors(data_residues, traj, other_desc)
        # Residue data PCA
            pca_residues, pc_residues = residue_pca(data_residues, traj, numberpc)
            # Add data to reporters
            variance_per_pc.append(pca_residues.explained_variance_ratio_)
            variance_total.append(sum(pca_residues.explained_variance_ratio_))

        # Align residue PCs to MSA
            MSA_input = MSA_inputs[target]
            pc_residues = pc_residues.values.tolist()

            descriptor_msa = []
            for msa_pos in MSA_input:
                res_match = False
                for res in pc_residues:
                    if msa_pos == res[0]: # First item is residue label
                        descriptor_msa.append(res[1:]) # Next items are PCs selected for that residue
                        res_match = True # Signal that this residue has already been included in MSA
                if not res_match: # If the MSA position did not have residue PCs calculated
                    descriptor_msa.append(msa_pos)

        # Populate descriptor dictionary
            # If there is no match with MSA, input as many zeroes as numberpc (number of PC selected per residue)
            descriptor_dict[entry] = list(itertools.chain(*[desc_pos if isinstance(desc_pos, list) else [0]*numberpc for desc_pos in descriptor_msa]))

        # Define feature labels (same for all entries, as it only depends of the length of MSA and number of PCs per residue)
        msa_pc_combinations = list(itertools.product(list(range(1,len(MSA_input)+1)), list(range(1,numberpc +1))))
        feature_labels = [f'AA{msa_pos}_PC{PC_num}' for msa_pos,PC_num in msa_pc_combinations]

        # Write descriptor dataframe for all entries
        descriptor_df = pd.DataFrame.from_dict(descriptor_dict, orient='index', columns=feature_labels)
        descriptor_df = descriptor_df.rename_axis('entry').reset_index()

        # Write descriptor to file for all entries if it does not exist
        descriptor_df.to_csv(descriptor_file, sep='\t', index=False)

        # Report descriptor statistics
        report_descriptor_stats('RS', desc_dir, name_tag, input_entries, user_flex, frame_split, sel_atoms,
                                sel_residues,
                                {'pca_explain': pca_explain, 'numberpc': numberpc, 'atom_pc_len': atom_pc_len,
                                 'variance_per_pc': variance_per_pc, 'variance_total': variance_total}, other_desc)

        with open(os.path.join(desc_dir, f'{name_tag}.log'), 'r') as report_file:
            print(report_file.read())
    else:
        print(f'Descriptor file {name_tag}.txt exists in output directory.')
        print('Summary report of previous calculation was:')
        with open(os.path.join(desc_dir, f'{name_tag}.log'), 'r') as report_file:
            report_lines = report_file.read().splitlines()
            for line in report_lines:
                print(line)
            # Check that the same alias was used for the same list of entries !
            report_entries = report_lines[3][27:]
            if input_entries != report_entries:
                raise ValueError('Input alias is the same but input entries do not match. CHECK! ')

# Generate PS-3DDPDs
def ps3ddpd_generation(md_dir: str,
                       desc_dir: str,
                       input_entries: str,
                       input_alias: str,
                       sel_atoms: str,
                       sel_residues: bool,
                       frame_split: int,
                       user_flex: str,
                       pca_explain: float,
                       **bp_kwargs: dict):
    """
        Generates PS-3DDPDs

        :param md_dir: pathway to folder containing the MD files [str]
        :param desc_dir: pathway to folder to write 3DDPDs
        :param input_entries: comma-separated (no spaces) string of filenames to compute descriptors for [str].
                            Entry contains: RefID_target_mutation_replicate (e.g. 87_5ht1b_wt_1,87_5ht1b_W247K_1)
        :param input_alias: alias to identify in the output file name the set of input entries
        :param sel_atoms: choice of atoms to generate descriptors on ['all'/'nonC']
                        'all': All atoms
                        'nonC': All non-Carbon atoms
        :param sel_residues: choice of residues to generate descriptors on [True/False]
                        False: All residues
                        True: Selection of residues (eg: General GPCR class A binding pocket)
        :param frame_split: number of frames to be included in the frame_split [int]
        :param user_flex: choice of atom data option ['all'/'std']
                         'all': Average/Median/SD x/y/z-coordinates + Gasteiger Charge
                         'std' SD x/y/z-coordinates + Gasteiger Charge
        :param pca_explain: % of variance covered by atom PCA [float (0.0-1.0)]
        :param user_residue_inputs: residue selection if sel_residues = 'b'; generated from StructBindingPocket.py
        :bp_kwargs BP_input_file: json file with MDTraj binding pocket selection for targets of interest
    """

    # Parse binding pocket selection
    MSA_inputs,residue_inputs = parse_binding_pocket_selection(**bp_kwargs)

    # Output file initialization
    name_tag = descriptor_name('PS', input_alias, sel_atoms, sel_residues, frame_split, user_flex, pca_explain, None,
                               **bp_kwargs)
    # Compute descriptor if output file does not exist
    descriptor_file = os.path.join(desc_dir, f'{name_tag}.txt')
    if not os.path.exists(descriptor_file):
        descriptor_dict = {}

        # Reporters initialization
        data_atoms_entries = []
        data_atoms_entries_ref = []

        entries = input_entries.split(',')
        targets = [f"{entry.split('_')[1]}_human" for entry in entries]
        mutants = ['_'.join(entry.split('_')[1:3]) for entry in entries]

        # Initialize ps3DDPD calculation per independent trajectory
        for entry,target,mutant in zip(entries,targets,mutants):
            # Load MD trajectory and keep only protein
            print(f'\nNow processing: ({entry})')
            traj_full = read_trajectory(md_dir, entry)
            print(f'Total MD size:          {traj_full}')
            traj = traj_full.restrict_atoms(traj_full.topology.select("protein"))
            print(f'Protein MD size:        {traj}')

            # Select only residues of interest
            if sel_residues:
                traj = residue_selection(traj, target, residue_inputs)
            print(f'Residue select MD size: {traj}')

            # Select only atoms of interest
            atom_list_residue = atom_selection(traj, sel_atoms)

            # Atom data calculation for selection of interest
            for atom_res in atom_list_residue:
                data_atoms_entries.append(atom_data_framesplit(atom_res, frame_split, traj, computed_gasteiger_charges, user_flex))
                # Specify entries for which atom data is calculated for later grouping
                data_atoms_entries_ref.append(entry)

        # Generate ps3DDPDs for all entry trajectories simultaneously
        # If all trajectories are not of the same length, keep only the atom properties for the common frames
        data_atoms = DataFrame(data_atoms_entries).loc[:, ~DataFrame(data_atoms_entries).isnull().any()]

        # Atom data PCA
        pca_atoms, pc_atoms = atom_pca(data_atoms, pca_explain)

        # Group atom PCA results per entry to finalize descriptor
        # Add entry labels to selected atom PCs
        pc_atoms = DataFrame(data_atoms_entries_ref).merge(DataFrame(pc_atoms), left_index=True, right_index=True)
        pc_atoms = pc_atoms.values.tolist()

        for entry, target, mutant in zip(entries, targets, mutants):
            # Extract atom PCA results corresponding to this entry
            atoms_entry = [pc_atom[1:] for pc_atom in pc_atoms if entry == pc_atom[0]]
            # Aggregate atom PCA results per entry and calculate statistics to keep
            atoms_entry = DataFrame(atoms_entry)
            data_entry_stats = [atoms_entry.mean(axis=0).values.tolist(), atoms_entry.median(axis=0).values.tolist(), atoms_entry.std(axis=0).values.tolist()]
            # Flatten aggregated list, which is already the descriptor
            descriptor_dict[entry] = [i for j in data_entry_stats for i in j]

        # Define feature labels (dependent on number of atom PCs)
        pc_stat_combinations = list(itertools.product(list(range(1,len(pc_atoms[0]))), ['Avg', 'Med', 'StD']))
        feature_labels = [f'PC{PC_num}_{stat}' for PC_num,stat in pc_stat_combinations]

        # Write descriptor dataframe for all entries
        descriptor_df = pd.DataFrame.from_dict(descriptor_dict, orient='index', columns=feature_labels)
        descriptor_df = descriptor_df.rename_axis('entry').reset_index()
        print(descriptor_df)

        # Write descriptor to file for all entries if it does not exist
        descriptor_df.to_csv(descriptor_file, sep='\t', index=False)

        # Report descriptor statistics
        report_descriptor_stats('PS', desc_dir, name_tag, input_entries, user_flex, frame_split, sel_atoms,
                                sel_residues, {'pca_atoms': pca_atoms}, None)
        with open(os.path.join(desc_dir, f'{name_tag}.log'), 'r') as report_file:
                print(report_file.read())
    else:
        print(f'Descriptor file {name_tag}.txt exists in output directory.')
        print('Summary report of previous calculation was:')
        with open(os.path.join(desc_dir, f'{name_tag}.log'), 'r') as report_file:
            report_lines = report_file.read().splitlines()
            for line in report_lines:
                print(line)
            # Check that the same alias was used for the same list of entries !
            report_entries = report_lines[3][27:]
            if input_entries != report_entries:
                raise ValueError('Input alias is the same but input entries do not match. CHECK! ')


class Descriptor:
    def __init__(self,desc_dir,desc_name,out_dir):
        # Reading options
        self.descriptor_name = desc_name
        descriptor_file = os.path.join(desc_dir, f'{desc_name}.txt')
        df = pd.read_csv(descriptor_file, sep='\t', engine='python')
        df.set_index('entry', inplace=True)
        self.descriptor_df = df

        # Writing options
        self.output_file = os.path.join(out_dir,f'{desc_name}.svg')


    def plot_individual_entry(self, save):
        """"
        Plot descriptor values (Y) per feature (X) for each entry (target/mutant/replicate)
        """
        entry_list = self.descriptor_df.index

        for entry in entry_list:
            colors = get_colors()

            fig, axs = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
            fig.set_facecolor('w')

            y = self.descriptor_df.loc[entry].to_list()

            target = f"{entry.split('_')[1]}_human"

            axs.plot(y, color=colors[target], linewidth=1)
            axs.set_title(entry, fontsize=10)

            axs.set(xlabel='Feature #n', ylabel='Feature value')
            axs.label_outer()

            plt.ylim(-12,22)

            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.1,
                                hspace=0.4)

            plt.tight_layout()

            if save:
                plt.savefig(self.output_file.replace('.', f'_{entry}.'))

    def plot_all_entries(self, subset, title_var, save):
        """
        Plot descriptor values (Y) per feature (X) for all or a subset of entries (target/mutant/replicate)
        """
        # Use subset to filter df
        if subset is not None:
            df = self.descriptor_df.loc[subset]
        else:
            df = self.descriptor_df

        entry_list = df.index
        target_list = [f"{entry.split('_')[1]}_human" for entry in entry_list]
        entry_target_list = [(entry,target) for entry,target in zip(entry_list,target_list)]

        colors = get_colors()

        target_order = [target for target in colors.keys() if target in target_list]
        entry_target_list_sorted = [tuple for x in target_order for tuple in entry_target_list if tuple[1]==x]

        min_y = floor(min(df.min().tolist()))
        max_y = ceil(max(df.max().tolist()))

        coords = []

        rows = ceil(len(target_list) / 2)
        for row in np.arange(rows):
            coords.append((row, 0))
            coords.append((row, 1))

        fig, axs = plt.subplots(ceil(int(df.shape[0]/2)), 2, figsize=(10, 1 * rows), dpi=300)
        fig.set_facecolor('w')

        for i, (entry, target) in enumerate(entry_target_list_sorted):
            y = df.loc[entry].to_list()

            axs[coords[i][0], coords[i][1]].plot(y, color=colors[target], linewidth=1)
            if title_var == 'entry':
                axs[coords[i][0], coords[i][1]].set_title(entry, fontsize=10)
            elif title_var == 'target':
                axs[coords[i][0], coords[i][1]].set_title(target, fontsize=10)

        for ax in axs.flat:
            ax.set(xlabel='Feature #n', ylabel='Feature\nvalue')
            ax.label_outer()
            ax.set_ylim(min_y, max_y)
            ax.set_yticks([min_y, 0, max_y])

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.4)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_file.replace('.', f'_2col.'))

    def plot_all_entries_one_col(self, subset, title_var, save):
        """
        Plot descriptor values (Y) per feature (X) for all or a subset of entries (target/mutant/replicate)
        """
        # Use subset to filter df
        if subset is not None:
            df = self.descriptor_df.loc[subset]
        else:
            df = self.descriptor_df

        entry_list = df.index.tolist()
        # print(entry_list)
        # import re
        # entry_list.sort(key=lambda s: int(re.search(r'\d+', s.split('_')[2]).group()))
        # print(entry_list)
        target_list = [f"{entry.split('_')[1]}_human" for entry in entry_list]
        entry_target_list = [(entry,target) for entry,target in zip(entry_list,target_list)]

        colors = get_colors()

        target_order = [target for target in colors.keys() if target in target_list]
        entry_target_list_sorted = [tuple for x in target_order for tuple in entry_target_list if tuple[1]==x]

        min_y = floor(min(df.min().tolist()))
        max_y = ceil(max(df.max().tolist()))

        coords = []

        rows = len(target_list)
        # for row in np.arange(rows):
        #     coords.append((row, 0))
        #     coords.append((row, 1))

        fig, axs = plt.subplots(rows, 1, figsize=(8, 2 * rows), dpi=300)
        fig.set_facecolor('w')

        for i, (entry, target) in enumerate(entry_target_list_sorted):
            y = df.loc[entry].to_list()

            axs[i].plot(y, color=colors[target], linewidth=1)
            if title_var == 'entry':
                axs[i].set_title(entry)
            elif title_var == 'target':
                axs[i].set_title(target)

        for ax in axs.flat:
            ax.set(xlabel='Feature #n', ylabel='Feature value')
            # ax.label_outer()
            ax.set_ylim(min_y, max_y)
            ax.set_yticks([min_y, 0, max_y])

        # plt.subplots_adjust(left=0.1,
        #                     bottom=0.1,
        #                     right=0.9,
        #                     top=0.9,
        #                     wspace=0.1,
        #                     hspace=0.4)

        fig.tight_layout(pad=1.0)

        if save:
            plt.savefig(self.output_file)

