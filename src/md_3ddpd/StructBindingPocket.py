### Load packages
from .utils import GPCRdb
import os.path
from csv import DictReader
import json
import itertools
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / 'data'

class BindingPocket:
    """"
    Makes a binding pocket selection based on MSA of the targets of interest

    :param target_input: Comma-separated (no spaces) string of uniprot_names of the desired targets (GPCRs)
                        If '3ddpd_set', then pre-defined 3DDPD target set is selected
    :param hierarchy: Hierarchy level to make subsets of targets prior to generating MSA ['None'/'gpcrdbA'/'family'/'subfamily'/'target']
    :param species_input: Latin species name (e.g. 'Homo sapiens')
                        If 'all_species', all species available are included
    :param output_type: List of output types to return ['a'/'b'/'c'/'d'/'e']
                        a) subset binding pocket: AAs per target (N32 A35 ... A379)
                        b) subset binding pocket: Chimera AA selection per target (select :32,35, ... ,379)
                        c) subset binding pocket: MDTraj AA selection per target  ("residue 32 or residue 35 or ... or residue 379")
                        d) subset MSA full sequence with AA_ID    (['-', '-', 'M1', 'P2' ... , '-'])
                        e) subset MSA binding pocket with AA_ID   (['N32', 'A35' ... '-', 'A379' ...] )
    :param precision: MSA alignment precision (%) [0-100]
    :param allosteric: Overlap between the defined binding pockets and the GPCRdb binding pocket (%) [0-100]
    :kwarg target_input_alias: Target input alias for output file name
    :kwarg output_dir: Path to json file to write out the selected MSA and binding pocket selection outputs
    """
    def __init__(self, target_input, hierarchy, species_input, output_type, precision=90, allosteric=0, **kwargs):
        self.target_input = target_input
        self.hierarchy = hierarchy
        self.species_input = species_input
        self.output = output_type
        self.user_precision = precision
        self.user_gpcrdb_overlap = allosteric

        if self.target_input == '3ddpd_set':
            self.target_input_tag = '3ddpd_set'
            self.target_list = ['5ht1b_human','5ht2b_human','aa1r_human','aa2ar_human','acm1_human','acm2_human',
                                'acm4_human','adrb2_human','agtr1_human','ccr5_human','cnr1_human','cxcr4_human',
                                'drd3_human','ednrb_human','ffar1_human','hrh1_human','lpar1_human','oprd_human',
                                'oprk_human','oprx_human','ox1r_human','ox2r_human','p2ry1_human','p2y12_human',
                                'par1_human','s1pr1_human']
        else:
            if not 'target_input_alias' in kwargs:
                raise TypeError("Must specify target input alias in kwargs")
            else:
                self.target_input_tag = kwargs['target_input_alias']
                self.target_list = self.target_input.split(',')

        if self.hierarchy == 'None':
            self.compute_subsets = False
            self.hierarchy_tag = 'NoHierarchy'
        else:
            self.compute_subsets = True
            self.hierarchy_tag = self.hierarchy

        self.species_input_tag = self.species_input

        if 'output_dir' in kwargs:
            self.output_dir = kwargs['output_dir']

    def get_target_subsets(self):
        """"
        Initialize target subsets to compute MSA and binding pocket selection based on input
        """
        # Make hierarchy groups
        subset_dict = {}
        if self.hierarchy == 'None':
            for target in self.target_list:
                subset_dict[target] = self.target_list

        else:
            for target in self.target_list:
                GPCRdb_target = GPCRdb(uniprot_name=target)
                family_slug = GPCRdb_target.get_family_slug()

                for i,hierarchy_level in enumerate(['gpcrdbA','family','subfamily','target']):
                    if self.hierarchy == hierarchy_level:
                        hierarchy_slug = '_'.join(family_slug.split('_')[0:i+1])

                        if self.species_input == 'all_species':
                            GPCRdb_family = GPCRdb(family_slug=hierarchy_slug)
                        else:
                            GPCRdb_family = GPCRdb(family_slug=hierarchy_slug,species=self.species_input)
                        hierarchy_targets = GPCRdb_family.get_family_targets()

                        subset_dict[target] = hierarchy_targets

        return subset_dict

    def get_GPCRdb_orth_BP(self, keep_gaps=False):
        """"
        Read definition of GPCRdb class A orthosteric binding pocket
        Input file needed in repo
        """
        gpcrdb_orth_pocket = []

        with open(os.path.join(data_path, 'GPCRA_orthosteric_binding_pocket_GPCRdb'), 'r') as gpcrdb_binding_pocket:
            reader = DictReader(gpcrdb_binding_pocket, delimiter='\t', fieldnames=['gpcrdb_target', 'seq_num'])
            for line in reader:
                if line['gpcrdb_target'] != 'GPCRdb_target':
                    line['seq_num'] = line['seq_num'].split(',')
                    if not keep_gaps:
                        line['seq_num'] = sorted([int(elem) for elem in line['seq_num'] if elem != '-'])

                gpcrdb_orth_pocket.append(line)

        return gpcrdb_orth_pocket

    def select_target_binding_pocket(self, uniprot_name):
        """
        Make a structure-based binding pocket selection for one target
        """
        print(f'\n{uniprot_name} - analyzing and preparing ')
        GPCRdb_entry = GPCRdb(uniprot_name=uniprot_name)

        # Initialize binding pocket dictionary
        binding_pocket_target = {}

        # Retrieve PDB structures per target
        pdb_entries = GPCRdb_entry.get_pdb_entries()
        pdb_codes = [x['pdb_code'] for x in pdb_entries]
        pdb_ligands_list = [x['ligands'] for x in pdb_entries]
        print(f'Structures analyzed: ({len(pdb_entries)})')
        # Initialize curation list
        pdb_codes_to_remove = []
        removal_reason = []

        # Extract and curate binding pocket per PDB structure
        for pdb_entry,pdb_code,pdb_ligands in zip(pdb_entries,pdb_codes,pdb_ligands_list):
            GPCRdb_pdb = GPCRdb(pdb=pdb_code)
            aa_binding_pocket = GPCRdb_pdb.get_pdb_interactions()

            # Remove structure if binding pocket not defined
            if aa_binding_pocket == []:
                print(f"   {pdb_code} removed | Non-defined binding-pocket.")

            else:
                # Remove structure if main ligand is peptide or protein
                for ligand_type in ['peptide', 'protein']:
                    if pdb_ligands[0]['type'] == ligand_type:
                        pdb_codes_to_remove.append(pdb_code)
                        removal_reason.append(f"   {pdb_code} removed | Non-small molecule ligand ({ligand_type}).")

                # Remove structure if sequence numbering is not standard
                interacting_aas = [interacting_aa['sequence_number'] for interacting_aa in aa_binding_pocket]
                if max(interacting_aas) >= 1000:
                    if pdb_code not in pdb_codes_to_remove:
                        pdb_codes_to_remove.append(pdb_code)
                        removal_reason.append(f"   {pdb_code} removed | Non-standardized AA sequence numbering.")

                # Iterate over interacting amino acids
                binding_pocket_pdb = []
                for interacting_aa in aa_binding_pocket:
                    # Extract interacting amino acids for structural binding pocket definition
                    binding_pocket_pdb.append([interacting_aa['sequence_number'], interacting_aa['amino_acid'], interacting_aa['pdb_code']])

                # Remove duplicates in nested list
                binding_pocket_pdb.sort()
                binding_pocket_pdb = list(l for l, _ in itertools.groupby(binding_pocket_pdb))

                binding_pocket_target[pdb_code] = binding_pocket_pdb

        # Remove PDB codes with reasons defined above
        for pdb_code,reason in zip(pdb_codes_to_remove,removal_reason):
            del binding_pocket_target[pdb_code]
            print(reason)

        return binding_pocket_target

    def correct_allostery(self, uniprot_name, binding_pocket_target):
        """
        Calculate overlap of defined binding pocket with GPCRdb class A orthosteric binding pocket and remove structures
        from binding pocket target dictionary if overlap is below the defined limit, suggesting allosterism
        """
        # Uniprot_name to GPCRdb_target (preparation allosteric pocket check)
        GPCRdb_entry = GPCRdb(uniprot_name=uniprot_name)
        gpcrdb_target = GPCRdb_entry.get_gpcrdb_target()

        # Read reference orthosteric GPCR class A binding pocket
        gpcrdb_orth_pocket = self.get_GPCRdb_orth_BP()
        gpcrdb_orth_pocket_target = [x['seq_num'] for x in gpcrdb_orth_pocket if x['gpcrdb_target'].lower() == gpcrdb_target.lower()][0]

        # Iterate over all PDB structures
        pdb_codes_to_remove = []
        for pdb_code in binding_pocket_target.keys():
            # Calculate overlap
            binding_pocket_pdb = binding_pocket_target[pdb_code]
            interacting_aas = [interacting_aa[0] for interacting_aa in binding_pocket_pdb]
            overlapping_aas = [aa for aa in interacting_aas if aa in gpcrdb_orth_pocket_target]

            overlap = (len(overlapping_aas) / len(interacting_aas)) * 100

            if overlap < self.user_gpcrdb_overlap:
                pdb_codes_to_remove.append(pdb_code)

        # Remove PDB codes with allostery above threshold
        for pdb_code in pdb_codes_to_remove:
            del binding_pocket_target[pdb_code]
            print(
                f"   {pdb_code} removed | Suspected allosteric ligand binding. "
                f"(GPCRdb binding pocket overlap = {round(overlap, 1)}% | goal = {self.user_gpcrdb_overlap}%)")

        return binding_pocket_target


    def correct_alignment(self, uniprot_name, binding_pocket_target, msa_target):
        """
        Correct binding pocket amino acids based on their sequence number correspondence to the MSA sequence number
        """
        msa_sequence = msa_target.replace('-','')

        def calculate_alignment_precision(binding_pocket_target):
            correct_scores_target = {}
            correct_aas_target = []
            interacting_aas_target = []

            for pdb_code, interacting_aas in binding_pocket_target.items():
                correct_aas_pdb = []
                for interacting_aa in interacting_aas:
                    seq_number = interacting_aa[0]
                    aa = interacting_aa[1]
                    # Check alignment on MSA sequence (seq numbers starts from 1, not zero)
                    try:
                        if msa_sequence[seq_number-1] == aa:
                            correct_aas_pdb.append(seq_number)
                    except IndexError: # If sequence number does not match, then do not consider correct
                        continue

                correct_aas_pdb = sorted(list(set(correct_aas_pdb)))
                correct_aas_target.extend(correct_aas_pdb)
                interacting_aas_pdb = sorted(list(set([x[0] for x in interacting_aas])))
                interacting_aas_target.extend(interacting_aas_pdb)

                # Report quality of binding pocket selection alignment per PDB structure
                correct_score = len(correct_aas_pdb) / len(interacting_aas_pdb) * 100
                correct_scores_target[pdb_code] = correct_score

            correct_aas_target = sorted(list(set(correct_aas_target)))
            interacting_aas_target = sorted(list(set(interacting_aas_target)))

            # Report quality of binding pocket selection alignment per target
            try:
                correct_score_target = len(correct_aas_target) / len(interacting_aas_target) * 100
            except ZeroDivisionError:
                correct_score_target = 0

            return correct_aas_target, interacting_aas_target, correct_score_target, correct_scores_target

        correct_aas_target, interacting_aas_target, correct_score_target, correct_scores_target = calculate_alignment_precision(binding_pocket_target)
        print(f'{len(correct_aas_target)} AAs correctly aligned out of {len(interacting_aas_target)}.'
              f'({round(correct_score_target,1)}%)')


        # Remove most incorrect structures until desired alignment precision is reached
        while correct_score_target < self.user_precision:
            if correct_score_target != 0:
                pdb_worse_score = min(correct_scores_target, key=correct_scores_target.get)

                del binding_pocket_target[pdb_worse_score]
                print(
                    f"   {pdb_worse_score} removed | High alignment missmatch ({correct_scores_target[pdb_worse_score]}%). ")

                # Re-calculate target alignment precision
                correct_aas_target, interacting_aas_target, correct_score_target, correct_scores_target = calculate_alignment_precision(
                    binding_pocket_target)
                print(f'{len(correct_aas_target)} AAs correctly aligned out of {len(interacting_aas_target)}.'
                      f'({round(correct_score_target, 1)}%)')
            else:
                binding_pocket_target = {}
                break

        return binding_pocket_target

    def unique_binding_residues(self, binding_pocket_target):
        """
        Extract unique binding residues from different PDB codes for one target
        """
        interacting_aas_target = []

        for pdb_code, interacting_aas in binding_pocket_target.items():
            for interacting_aa in interacting_aas:
                seq_number = interacting_aa[0]
                aa = interacting_aa[1]
                # Append only sequence number and AA, not PDB code
                interacting_aas_target.append([seq_number, aa])

        # Remove duplicates in nested list
        interacting_aas_target.sort()
        interacting_aas_target_unique = list(l for l, _ in itertools.groupby(interacting_aas_target))

        return interacting_aas_target_unique

    def get_subset_MSA(self):
        """
        Get from GPCRdb an alignment for the targets in a subset
        """
        # Read predefined GPCRdb Class A MSA
        if self.hierarchy == 'gpcrdbA':
            with open(os.path.join(data_path, 'GPCRdb_classA_MSA.json')) as json_file:
                classA_msa_dict = json.load(json_file) # MSA is list of AAs in format e.g. "Y37"
                classA_msa_dict_str = {k:''.join([aa[0] for aa in v]) for k,v in classA_msa_dict.items()} # Keep only AA without sequence number
                subset_msa_dict = {k: classA_msa_dict_str for k in self.target_list}

        # Calculate MSA for targets in target_input variable
        elif self.hierarchy == 'None':
            target_input_list = self.target_list

            GPCRdb_targets = GPCRdb(target_list=','.join(self.target_list)) # We do not use self.target_input to account for 3ddpd_set
            set_msa = GPCRdb_targets.get_MSA()

            subset_msa_dict = {}
            # Format in the same way as other subsets
            for uniprot_name in target_input_list:
                subset_msa_dict[uniprot_name] = set_msa

        # Calculate MSA for hierarchy-defined subset
        else:
            subset_dict = self.get_target_subsets()

            subset_msa_dict = {}
            for uniprot_name,subset_targets in subset_dict.items():
                GPCRdb_targets = GPCRdb(target_list=','.join(subset_targets))
                subset_msa_dict[uniprot_name] = GPCRdb_targets.get_MSA()

        return subset_msa_dict

    def MSA_sequence_to_AA_list(self, msa_sequence):
        """
        Convert MSA in sequence string format '---MPPS--EVL-----VPLVI--' to list format with sequence numbers:
        ['-','-','-','M1','P2','P3','S3','-','-','E5'...]
        """
        msa_list = []
        seq_number = 0
        for aa in msa_sequence:
            if aa != '-':
                seq_number += 1
                msa_list.append(f'{aa}{seq_number}')
            else:
                msa_list.append(f'-')
        return msa_list

    def align_binding_pocket_target(self, msa_target, binding_pocket_target):
        """
        Align extracted interacting AAs to the corresponding target subset MSA.
        Return the MSA positions where an interacting AA is found.
        """
        # Extract unique curated binding residues for this target
        interacting_aas_target = self.unique_binding_residues(binding_pocket_target)

        # Extract position in subset MSA of interacting AAs
        msa_binding_pos_target = []
        seq_number = 0
        for pos, aa in enumerate(msa_target):
            if aa != '-':
                seq_number += 1
                for interacting_aa in interacting_aas_target:
                    if seq_number == interacting_aa[0]:
                        msa_binding_pos_target.append(pos)

        return msa_binding_pos_target


    def get_subset_binding_pocket(self):
        """
        For the input targets of interest, extract the binding pocket of the length of combined binding pocket of
        all available targets in the hierarchy subset.
        E.g. self.hierarchy = 'subfamily'; self.target_input = 'aa1r_human,adrb2_human'

        """

        subset_msa_dict = self.get_subset_MSA()

        subset_binding_pocket = {}
        subset_binding_pocket_msa = {}

        for target_name in self.get_target_subsets().keys():
            # If hierarchy is GPCRdbA, select list of GPCRdb defined orthosteric class A binding pocket
            if self.hierarchy == 'gpcrdbA':
                # Uniprot_name to GPCRdb_target
                GPCRdb_entry = GPCRdb(uniprot_name=target_name)
                gpcrdb_target = GPCRdb_entry.get_gpcrdb_target()

                # Extract GPCRdb class A pocket for target
                bp = self.get_GPCRdb_orth_BP(keep_gaps=True)
                bp_seq = \
                [x['seq_num'] for x in bp if x['gpcrdb_target'].lower() == gpcrdb_target.lower()][0]

                # Map binding pocket positions to AAs
                target_sequence = GPCRdb_entry.get_sequence()
                bp_aa = [target_sequence[int(pos)-1] if pos != '-' else '-' for pos in bp_seq]

                subset_binding_pocket_target = [[seq_number, aa] for seq_number,aa in zip(bp_seq,bp_aa) if seq_number != '-']

                subset_binding_pocket[f'{target_name}'] = subset_binding_pocket_target

                # For target of interest, create subset binding pocket-only MSA
                subset_binding_pocket_msa[f'{target_name}'] = [f'{aa}{seq_number}' for seq_number,aa in zip(bp_seq,bp_aa)]


            # For all other hierarchy types, calculate structure-based subset binding pocket
            else:
                # Initialize MSA position list
                msa_binding_pos_subset = []

                # Extract binding pocket for all the members of the target subset
                for uniprot_name in self.get_target_subsets()[target_name]:
                    # Initialize output_dictionary
                    binding_pocket_aligned_target = {}

                    # Compute binding pocket
                    binding_pocket_target = self.select_target_binding_pocket(uniprot_name)

                    if binding_pocket_target:
                        # Correct binding pocket for allostery
                        binding_pocket_target = self.correct_allostery(uniprot_name,binding_pocket_target)

                        # Correct binding pocket for wrong alignment
                        msa_target = subset_msa_dict[target_name][uniprot_name]
                        binding_pocket_target = self.correct_alignment(uniprot_name,binding_pocket_target,msa_target)

                        print(f'Total structures used for binding pocket:    ({len(binding_pocket_target.keys())})')

                        # Define in which MSA positions this target has binding residues
                        msa_binding_pos_target = self.align_binding_pocket_target(msa_target, binding_pocket_target)
                        msa_binding_pos_subset.extend(msa_binding_pos_target)
                    else:
                        continue

                # Keep unique interacting positions in MSA
                msa_binding_pos_subset.sort()
                msa_binding_pos_subset = list(l for l, _ in itertools.groupby(msa_binding_pos_subset))

                # For target of interest, extract BP residues according to subset BP
                msa_target = subset_msa_dict[target_name][target_name]
                subset_binding_pocket_msa_target = []

                for binding_pos in msa_binding_pos_subset:
                    seq_number = 0
                    for pos, aa in enumerate(msa_target):
                        if aa != '-':
                            seq_number += 1

                        if pos == binding_pos:
                            if aa != '-':
                                subset_binding_pocket_msa_target.append(f'{aa}{seq_number}')
                            else:
                                subset_binding_pocket_msa_target.append(f'-')

                subset_binding_pocket_msa[f'{target_name}'] = subset_binding_pocket_msa_target

                # For target of interest, create subset binding pocket-only MSA
                subset_binding_pocket[f'{target_name}'] = [[x[1:],x[0]] for x in subset_binding_pocket_msa[f'{target_name}'] if x != '-']

            # Print summary of subset binding pocket retrieval
            print('\n#####################################################################\n')
            print(f'Target input: {target_name} | Hierarchy level: {self.hierarchy_tag}')
            if self.hierarchy == 'gpcrdbA':
                print(f'Length of binding pocket MSA: {len(list(subset_binding_pocket_msa.values())[0])}')
            else:
                print(f'Number of proteins in {self.hierarchy_tag} subset: {len(subset_msa_dict[target_name].keys())}')
                print(f'Length of subset MSA: {len(msa_target)}')
            print(f'Length of subset binding pocket MSA: {len(subset_binding_pocket_msa[f"{target_name}"])}')
            print(f'Number of residues in subset binding pocket: {len(subset_binding_pocket[f"{target_name}"])}')
            print('\n#####################################################################\n')

        return subset_binding_pocket,subset_binding_pocket_msa


    def get_output(self):
        """
        Get MSA or binding pocket selection output.
        Returns a dictionary and optionally writes it out as a json file (if output_dir is defined)
        """
        # Initialize output json file names
        file_name_tag = f'{self.target_input_tag}_{self.species_input_tag}_{self.hierarchy_tag}_' \
                        f'precision{self.user_precision}_allosteric{self.user_gpcrdb_overlap}'

        # Calculate (subset) binding pocket residues and MSA
        subset_binding_pocket, subset_binding_pocket_msa = self.get_subset_binding_pocket()

        # Define functions to format binding pocket selection output
        def aa_list_format(bp):
            return ' '.join([f'{aa[1]}{aa[0]}' for aa in bp])

        def chimera_format(bp):
            expr_list = [f'{binding_aa[0]}' for binding_aa in bp]
            expr = f"select: {','.join(expr_list)}"
            return expr

        def mdtraj_format(bp):
            expr_list = [f'{binding_aa[0]}' for binding_aa in bp]
            expr = f"residue {' '.join(expr_list)}"
            return expr

        def msa_full_format(msa):
            return self.MSA_sequence_to_AA_list(msa)

        # Generate outputs
        output_type_dict = {'a':['BP_AA', aa_list_format],
                            'b':['BP_Chimera', chimera_format],
                            'c':['BP_MDtraj', mdtraj_format],
                            'd':['MSA_full',msa_full_format],
                            'e':['MSA_bp',]}

        # Iterate over output types to generate as many as defined
        for output_type,output_options in output_type_dict.items():
            if output_type in self.output:
                if 'BP' in output_options[0]:
                    # Format binding pocket output
                    output = {k: output_options[1](v) for k, v in subset_binding_pocket.items()}

                if 'MSA_full' in output_options[0]:
                    # Return full sequence subset MSA for input targets
                    output = {k: output_options[1](v[k]) for k, v in self.get_subset_MSA().items()}

                if 'MSA_bp' in output_options[0]:
                    # Return binding pocket MSA
                    output = subset_binding_pocket_msa

                # Print output
                print(f'=================== Binding pocket / MSA selection output: {output_options[0]} ===================')
                if 'BP' in output_options[0]:
                    print(json.dumps(output, indent=4))
                else:
                    print(output)

                # Write output if directory is defined
                try:
                    with open(os.path.join(self.output_dir, f'{output_options[0]}_{file_name_tag}.json'), 'w') as file:
                        json.dump(output, file)
                except AttributeError:
                    print('Output directory not defined. Skipping writting output...')

