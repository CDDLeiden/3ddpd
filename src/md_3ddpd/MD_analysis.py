### Load packages
from .utils import read_trajectory,GPCRdb,map_msa_segment
from .definitions import get_colors
import os
import mdtraj as md
import glob
import json
from math import ceil
from matplotlib import pyplot as plt
import numpy as np

target_list_3ddpd = '5ht1b_human,5ht2b_human,aa1r_human,aa2ar_human,acm1_human,acm2_human,acm4_human,adrb2_human,agtr1_human,ccr5_human,cnr1_human,cxcr4_human,drd3_human,ednrb_human,ffar1_human,hrh1_human,lpar1_human,oprd_human,oprk_human,oprx_human,ox1r_human,ox2r_human,p2ry1_human,p2y12_human,par1_human,s1pr1_human'

class MDTrajectory:
    def __init__(self,trajectory_path,entry,**kwargs):
        # Read trajectory/trajectories
        if len(entry.split(',')) == 1:
            self.entry = entry
            self.unique_entry = True
            print(f'Processing one MD entry: {entry}')
            self.target = f"{entry.split('_')[1]}_human"
            self.trajectory = read_trajectory(trajectory_path, entry)
            print(f'MD trajectory read for entry {self.entry}')
        else:
            self.unique_entry = False
            self.entry_list = entry.split(',')
            print(f'Processing a list of {len(self.entry_list)} MD entries')
            trajectory_dict = {}
            for t in self.entry_list:
                trajectory_dict[t] = {}
                trajectory_dict[t]['target'] = f"{t.split('_')[1]}_human"
                trajectory_dict[t]['trajectory'] = read_trajectory(trajectory_path, t)
                print(f'MD trajectory read for entry {t}')
            self.trajectory_dict = trajectory_dict

        # Analysis options
        if 'MSA_file' in kwargs:
            msa_file = kwargs['MSA_file']
            with open(msa_file, 'r') as MSA_file:
               self.msa = json.load(MSA_file)
        if 'average_replicates' in kwargs:
            self.average_replicates = kwargs['average_replicates']
            if self.average_replicates:
                if self.unique_entry:
                    print('No replicates defined for averaging.')
                else:
                    self.replicates_dict = {}
                    unique_systems = list(set(['_'.join(entry.split('_')[:-1]) for entry in self.entry_list]))
                    for system in unique_systems:
                        replicates = [entry for entry in self.entry_list if system in entry]
                        self.replicates_dict[system] = {}
                        for replicate in replicates:
                            self.replicates_dict[system][replicate] = self.trajectory_dict[replicate]
        else:
            self.average_replicates = False


        if 'normalize' in kwargs:
            self.normalize = kwargs['normalize']
            def fetch_baseline_entry(entry):
                # Define normalized entry based on kwarg 'normalize_entry' keywords
                fetch_normalize_entry = f'{entry.split("_")[1]}_{kwargs["normalize_entry"]}'
                fetched_normalize_entry = glob.glob(f'{trajectory_path}\\{entry.split("_")[1]}_human\\*_{fetch_normalize_entry}*')

                # Fetch potential entries
                if len(fetched_normalize_entry) == 1:
                    normalize_entry = fetched_normalize_entry[0]
                else:
                    raise TypeError('More restrictive normalizing entry definition needed')
                # Define replicate to use
                if len(kwargs["normalize_entry"].split('_')) == 1:  # No replicate defined (same as entry)
                    selected_normalize_entry = normalize_entry.split("\\")[-1] + '_' + str(entry.split("_")[-1])
                elif len(kwargs["normalize_entry"].split('_')) == 2:
                    selected_normalize_entry = normalize_entry
                # Keep only entry name without extension or path
                selected_normalize_entry = os.path.basename(selected_normalize_entry).split('.')[0]

                return selected_normalize_entry

            if self.normalize:
                if self.unique_entry:
                    self.normalize_entry = fetch_baseline_entry(self.entry)
                    # Read normalization entry trajectory
                    self.normalize_trajectory = read_trajectory(trajectory_path, self.normalize_entry)
                    print(f'Baseline MD trajectory read for entry {self.normalize_entry}')
                else:
                    for t in self.entry_list:
                        self.trajectory_dict[t]['normalize_entry'] = fetch_baseline_entry(t)
                        # Read normalization entry trajectory
                        self.trajectory_dict[t]['normalize_trajectory'] = read_trajectory(trajectory_path,self.trajectory_dict[t]['normalize_entry'])
                        print(f'Baseline MD trajectory read for entry {self.trajectory_dict[t]["normalize_entry"]}')

        else:
            self.normalize = False

        # Plotting options
        if 'plot_align' in kwargs:
            self.plot_align = kwargs['plot_align']
        else:
            self.plot_align = False
        if 'plot_segments' in kwargs:
            self.plot_segments = kwargs['plot_segments']
        else:
            self.plot_segments = False
        if 'plot_normalize' in kwargs:
            self.plot_normalize = kwargs['plot_normalize']
        else:
            self.plot_normalize = False
        if 'plot_mutation' in kwargs:
            self.plot_mutation = kwargs['plot_mutation']
        else:
            self.plot_mutation = False

        # Output options
        if 'output_dir' in kwargs:
            self.output_dir = kwargs['output_dir']
            self.plot_output_file = f'RMSF.svg'
            self.rmsf_output_file = f'RMSF.txt'

            if self.normalize:
                self.plot_output_file = self.plot_output_file.replace('.', f'_N.')
                self.rmsf_output_file = self.rmsf_output_file.replace('.', f'_N.')
            if self.plot_align:
                self.plot_output_file = self.plot_output_file.replace('.', f'_A.')
                self.rmsf_output_file = self.rmsf_output_file.replace('.', f'_A.')
            if self.plot_segments:
                self.plot_output_file = self.plot_output_file.replace('.', f'_S.')
            if self.plot_mutation:
                self.plot_output_file = self.plot_output_file.replace('.', f'_M.')


    def rmsf(self,normalize,entry,trajectory):
        """"
        Compute RMSF of trajectory backbone
        """
        # Choose trajectory
        if normalize:
            if self.unique_entry:
                trajectory = self.normalize_trajectory
                entry = self.normalize_entry
            else:
                trajectory = self.trajectory_dict[entry]['normalize_trajectory']
                entry = self.trajectory_dict[entry]['normalize_entry']

        # Remove lisozyme or other stabilizing artifacts
        gpcr = trajectory.topology.select('protein and (residue <= 1000)')
        traj_gpcr = trajectory.atom_slice(gpcr)

        # Slice backbone trajectory
        backbone = traj_gpcr.topology.select('protein and name CA')
        traj_bb = traj_gpcr.atom_slice(backbone)

        # Save backbone residues for plotting
        bb_residues = []
        bb_aas = []
        for residue in traj_bb.topology.residues:
            bb_aas.append(residue)
            residue_number = int(str(residue)[3:])  # Residue number is residue without the aa name
            bb_residues.append(residue_number)

        # Calculate RMSF for protein backbone
        rmsf_bb = md.rmsf(traj_bb, traj_bb, 0) * 10.0  # Calculate RMSF in Å (default in nm)
        print(f'RMSF calculated for entry {entry}')

        return rmsf_bb,bb_residues,bb_aas


    def align_rmsf(self,normalize,entry,trajectory,target):
        """"
        Map backbone RMSF to MSA used in rs3DDPD descriptor calculation for direct comparison
        """
        if not self.msa:
            raise TypeError("must provide MSA json file")
        else:
            # Compute RMSF
            if not normalize:
                # rmsf_bb, bb_residues, bb_aas = self.rmsf(normalize=False)
                rmsf_bb, bb_residues, bb_aas = self.rmsf(normalize=False,entry=entry,trajectory=trajectory)
            else:
                # rmsf_bb, bb_residues, bb_aas = self.rmsf(normalize=True)
                rmsf_bb, bb_residues, bb_aas = self.rmsf(normalize=True,entry=entry,trajectory=trajectory)

            # Read GPCRdb MSA
            # msa = self.msa[self.target]
            msa = self.msa[target]

            # Insert zeroes in RMSF output corresponding to gaps in the MSA and residues not present in the trajectory topology
            rmsf_bb_msa = []
            bb_residues_msa = []
            j = 0
            k = 0
            for i, pos in enumerate(msa):
                if pos == '-':
                    pos1 = 0.0
                    res1 = '-'
                else:
                    if (j + 1) in bb_residues:
                        pos1 = rmsf_bb[k]
                        res1 = bb_residues[k]
                        k += 1
                    else:
                        pos1 = 0.0
                        res1 = '-'
                    j += 1

                rmsf_bb_msa.append(pos1)
                bb_residues_msa.append(res1)

            return rmsf_bb_msa,bb_residues_msa

    def average_rmsf(self,normalize,align_rmsf,trajectory_dict):
        """
        Compute the average RMSF between replicates for one system
        """
        replicates_rmsf = []
        for entry in trajectory_dict:
            trajectory = trajectory_dict[entry]['trajectory']
            target = trajectory_dict[entry]['target']
            if not align_rmsf:
                rmsf_bb, bb_residues, bb_aas = self.rmsf(normalize,entry=entry,trajectory=trajectory)
            else:
                rmsf_bb, bb_residues_msa = self.align_rmsf(normalize, entry=entry, trajectory=trajectory, target=target)
            replicates_rmsf.append(rmsf_bb)
        average_rmsf = np.mean(np.array(replicates_rmsf), axis=0)
        std_rmsf = np.std(np.array(replicates_rmsf), axis=0)

        if not align_rmsf:
            return average_rmsf, std_rmsf, bb_residues, bb_aas
        else:
            return average_rmsf, std_rmsf, bb_residues_msa

    def plot_individual_rmsf(self, entry, target, trajectory, fig, position, n_subplots):
        """
        Generate a RMSF plot with the specified options for the entry of interest
        """
        # Initialize plot
        colors = get_colors()
        ax = fig.add_subplot(n_subplots, 1, position)
        # Calculate RMSF to plot
        if not self.average_replicates:
            if not self.plot_align:
                y, bb_residues, bb_aas = self.rmsf(normalize=False, entry=entry, trajectory=trajectory)
            else:
                y, bb_residues_msa = self.align_rmsf(normalize=False, entry=entry, trajectory=trajectory, target=target)
        else:
            if not self.plot_align:
                y, error, bb_residues, bb_aas = self.average_rmsf(normalize=False, align_rmsf=self.plot_align, trajectory_dict=trajectory)
            else:
                y, error, bb_residues_msa = self.average_rmsf(normalize=False, align_rmsf=self.plot_align, trajectory_dict=trajectory)

        # Normalize RMSF values to baseline (e.g. wt)
        if self.plot_normalize:
            if not self.average_replicates:
                if not self.plot_align:
                    y_wt = self.rmsf(normalize=True, entry=entry, trajectory=trajectory)[0]
                else:
                    y_wt = self.align_rmsf(normalize=True, entry=entry, trajectory=trajectory, target=target)[0]
            else:
                y_wt = self.average_rmsf(normalize=True, align_rmsf=self.plot_align, trajectory_dict=trajectory)

            y_norm = np.array(y) - np.array(y_wt)
            y_write = y_norm
            # Plot RMSF (normalized)
            ax.plot(y_norm, color=colors[target], linewidth=1.5)

            # Plot line at zero to simbolize baseline (i.e. wt)
            plt.axhline(y=0.0, color='black', linestyle=':', linewidth=1.5)

        else:
            y_write = y
            # Plot RMSF (not normalized)
            plt.plot(y, color=colors[target], linewidth=1.5)

        # Write file with RMSF values plotted
        if os.path.exists(os.path.join(self.output_dir, self.rmsf_output_file)):
            with open(os.path.join(self.output_dir, self.rmsf_output_file)) as rmsf_file:
                rmsf_dict = json.load(rmsf_file)
                if entry not in rmsf_dict.keys():
                    rmsf_dict[entry] = [float(y_item) for y_item in y_write]
        else:
            rmsf_dict = {entry: [float(y_item) for y_item in y_write]}
        with open(os.path.join(self.output_dir, self.rmsf_output_file), 'w') as rmsf_file:
            json.dump(rmsf_dict, rmsf_file)

        # Add shades representing standard deviation if average RMSF is plotted
        if self.average_replicates:
            plt.fill_between(range(0,len(y)),y-error, y+error, color=colors[target], alpha=0.4)

        # Add shades representing the location of the 7 TM domains
        if self.plot_segments:
            # Calculate segment dictionary and MSA used to generate 3ddpds
            GPCRdb_target = GPCRdb(uniprot_name=target)
            GPCRdb_3ddpdset = GPCRdb(target_list=target_list_3ddpd)
            segment_dict = GPCRdb_target.get_segment_dictionary()
            msa_3ddpdset = GPCRdb_3ddpdset.get_MSA()

            if self.plot_align:
                # Align segment dict to MSA and plot
                msa_segment = map_msa_segment(target, segment_dict, msa_3ddpdset)
            else:
                # Assign segment to available residues
                msa_segment = []
                for res in bb_residues:
                    try:
                        msa_segment.append(segment_dict[res])
                    except:
                        msa_segment.append('-')

            reversed_msa_segment = msa_segment[::-1]
            segments = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
            for segment in segments:
                first = msa_segment.index(segment)
                last = len(msa_segment) - 1 - reversed_msa_segment.index(segment)
                ax.axvspan(first, last, color=colors[target], alpha=0.1)

        # Add red line at the location of the mutation
        if self.plot_mutation:
            mutation = entry.split('_')[2]
            if mutation != 'wt':
                if not self.plot_align:
                    for i, res in enumerate(bb_residues):
                        if res == int(mutation[1:-1]):
                            mutation_x_location = i
                else:
                    for i, res in enumerate(bb_residues_msa):
                        if res == int(mutation[1:-1]):
                            mutation_x_location = i
                plt.axvline(x=mutation_x_location, color='red')

        # If not aligned, plot real sequence numbers as X ticks (every 20)
        if not self.plot_align:
            xticks = []
            xticklabels = []
            for i, x in enumerate(bb_residues):
                if (x == 1) or ((x > 0) and (i % 25 == 0)):
                    xticks.append(i)
                    xticklabels.append(x)

            ax.set(xticks=xticks, xticklabels=xticklabels)

        # Include titles to plot
        plt.title(f'RMSF {entry}')
        if not self.plot_align:
            plt.xlabel('Residue')
        else:
            plt.xlabel('MSA position')
        plt.ylabel('RMSF ($\AA$)')

        # Add y axis limit so all plots look the same
        if self.plot_normalize:
            plt.ylim(-3.5, 3.5)
        else:
            plt.ylim(0, 10.5)

        return fig

    def plot_rmsf(self, alias):
        """"
        Plot RMSF
        """
        if self.unique_entry:
            fig = plt.figure(1,figsize=(8,4), dpi=300)
            fig = self.plot_individual_rmsf(self.entry, self.target, self.trajectory, fig, 1, 1)
        else:
            if self.average_replicates:
                n_subplots = len(self.replicates_dict.keys())
                fig = plt.figure(1, figsize=(8, 2 * n_subplots), dpi=300)
                for i,system in enumerate(self.replicates_dict.keys()):
                    entry = f'{system}_avg'
                    target = self.replicates_dict[system][list(self.replicates_dict[system].keys())[0]]['target']
                    trajectory_dict = self.replicates_dict[system]
                    fig = self.plot_individual_rmsf(entry, target, trajectory_dict, fig, i + 1, n_subplots)

            else:
                n_subplots = len(self.entry_list)
                fig = plt.figure(1, figsize=(8, 2 * n_subplots), dpi=300)
                for i,entry in enumerate(self.entry_list):
                    target = self.trajectory_dict[entry]['target']
                    trajectory = self.trajectory_dict[entry]['trajectory']
                    fig = self.plot_individual_rmsf(entry, target, trajectory, fig, i+1, n_subplots)

        fig.tight_layout(pad=1.0)

        if self.plot_output_file:
            if self.unique_entry:
                plt.savefig(os.path.join(self.output_dir, self.plot_output_file.replace('.', f'_{self.entry}.')))
            elif not self.unique_entry and alias is not None:
                plt.savefig(os.path.join(self.output_dir, self.plot_output_file.replace('.', f'_{alias}.')))

    def plot_rmsf_2col(self, alias):
        """
        Plot RMSF in 2 columns in the same style as 3DDPDs
        """
        if self.unique_entry:
            print('Multiple entries needed to generate this plot')
        else:
            if not self.average_replicates:
                n_subplots = len(self.entry_list)
                coords = []

                rows = ceil(len(self.entry_list) / 2)
                for row in np.arange(rows):
                    coords.append((row, 0))
                    coords.append((row, 1))

                fig, axs = plt.subplots(rows, 2, figsize=(10, 1 * rows), dpi=300)
                fig.set_facecolor('w')

                entry_list = self.entry_list
                import re
                entry_list.sort(key=lambda s: int(re.search(r'\d+', s.split('_')[2]).group()))
                target_list = [f"{entry.split('_')[1]}_human" for entry in entry_list]
                entry_target_list = [(entry, target) for entry, target in zip(entry_list, target_list)]
                colors = get_colors()
                target_order = [target for target in colors.keys() if target in target_list]
                entry_target_list_sorted = [tuple for x in target_order for tuple in entry_target_list if tuple[1] == x]

                for i, (entry, target) in enumerate(entry_target_list_sorted):
                    # Plot RMSF
                    trajectory = self.trajectory_dict[entry]['trajectory']
                    y, bb_residues_msa = self.align_rmsf(normalize=False, entry=entry, trajectory=trajectory, target=target)
                    if not self.plot_normalize:
                        print('plotting raw')
                        axs[coords[i][0], coords[i][1]].plot(y, color=colors[target], linewidth=1.5)
                    else:
                        print('plotting normlaized')
                        y_wt = self.align_rmsf(normalize=True, entry=entry, trajectory=trajectory, target=target)[0]
                        y_norm = np.array(y) - np.array(y_wt)
                        axs[coords[i][0], coords[i][1]].plot(y_norm, color=colors[target], linewidth=1.5)
                    # axs[coords[i][0], coords[i][1]].set_title(target, fontsize=10)
                    axs[coords[i][0], coords[i][1]].set_title(' '.join(entry.split('_')[1:3]), fontsize=10)

                    # Plot TM segments
                    GPCRdb_target = GPCRdb(uniprot_name=target)
                    GPCRdb_3ddpdset = GPCRdb(target_list=target_list_3ddpd)
                    segment_dict = GPCRdb_target.get_segment_dictionary()
                    msa_3ddpdset = GPCRdb_3ddpdset.get_MSA()

                    msa_segment = map_msa_segment(target, segment_dict, msa_3ddpdset)
                    reversed_msa_segment = msa_segment[::-1]
                    segments = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
                    for segment in segments:
                        first = msa_segment.index(segment)
                        last = len(msa_segment) - 1 - reversed_msa_segment.index(segment)
                        axs[coords[i][0], coords[i][1]].axvspan(first, last, color=colors[target], alpha=0.1)

                    # Plot mutation
                    if self.plot_mutation:
                        mutation = entry.split('_')[2]
                        if mutation != 'wt':
                            for j, res in enumerate(bb_residues_msa):
                                if res == int(mutation[1:-1]):
                                    mutation_x_location = j
                            axs[coords[i][0], coords[i][1]].axvline(x=mutation_x_location, color='red')

                for ax in fig.axes:
                    ax.set(xlabel='MSA position', ylabel='RMSF\n($\AA$)')
                    if self.plot_normalize:
                        ax.set_ylim(-5.5, 5.5)
                    else:
                        ax.set_ylim(0, 10.5)
                    # ax.set_ylim(0, 10)
                    # ax.set_yticks([0, 5, 10])
                    ax.label_outer()

                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.1,
                                    hspace=0.4)

                plt.tight_layout()

                if self.plot_output_file:
                    plt.savefig(os.path.join(self.output_dir, self.plot_output_file.replace('.', f'_{alias}_2col.')))


