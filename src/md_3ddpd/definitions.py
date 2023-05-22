import mdtraj as md
from mdtraj import load
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import ComputeGasteigerCharges
from pandas import Series
import os

### Definitions
def gasteiger_charges(Res_3letter, SMILES):
    residue_charges = []
    mol = MolFromSmiles(SMILES)
    ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    residue_charges += [contribs[0], (Res_3letter + '_O')], [contribs[1], (Res_3letter + '_C')], [contribs[7],
                                                                                                  (Res_3letter + '_CA')]
    if Res_3letter == 'GLY':
        residue_charges += [[contribs[8], (Res_3letter + '_N')]]
    if Res_3letter == 'ALA':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_N')]
    if Res_3letter == 'SER':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_OG')], [contribs[10], (
                    Res_3letter + '_N')]
    if Res_3letter == 'THR':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_OG1')], [contribs[10], (
                    Res_3letter + '_CG2')], [contribs[11], (Res_3letter + '_N')]
    if Res_3letter == 'CYS':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_SG')], [contribs[10], (
                    Res_3letter + '_N')]
    if Res_3letter == 'VAL':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG1')], [contribs[10], (
                    Res_3letter + '_CG2')], [contribs[11], (Res_3letter + '_N')]
    if Res_3letter == 'LEU':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD1')], [contribs[11], (Res_3letter + '_CD2')], [contribs[12], (Res_3letter + '_N')]
    if Res_3letter == 'ILE':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG2')], [contribs[10], (
                    Res_3letter + '_CG1')], [contribs[11], (Res_3letter + '_CD1')], [contribs[12], (Res_3letter + '_N')]
    if Res_3letter == 'MET':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_SD')], [contribs[11], (Res_3letter + '_CE')], [contribs[12], (Res_3letter + '_N')]
    if Res_3letter == 'PRO':
        residue_charges += [contribs[8], (Res_3letter + '_N')], [contribs[13], (Res_3letter + '_CD')], [contribs[14], (
                    Res_3letter + '_CG')], [contribs[15], (Res_3letter + '_CB')]
    if Res_3letter == 'PHE':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD1')], [contribs[11], (Res_3letter + '_CE1')], [contribs[12],
                                                                                     (Res_3letter + '_CZ')], [
                               contribs[13], (Res_3letter + '_CE2')], [contribs[14], (Res_3letter + '_CD2')], [
                               contribs[15], (Res_3letter + '_N')]
    if Res_3letter == 'TYR':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD1')], [contribs[11], (Res_3letter + '_CE1')], [contribs[12],
                                                                                     (Res_3letter + '_CZ')], [
                               contribs[13], (Res_3letter + '_OH')], [contribs[14], (Res_3letter + '_CE2')], [
                               contribs[15], (Res_3letter + '_CD2')], [contribs[16], (Res_3letter + '_N')]
    if Res_3letter == 'TRP':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD1')], [contribs[11], (Res_3letter + '_NE1')], [contribs[12],
                                                                                     (Res_3letter + '_CE2')], [
                               contribs[13], (Res_3letter + '_CD2')], [contribs[14], (Res_3letter + '_CE3')], [
                               contribs[15], (Res_3letter + '_CZ3')], [contribs[16], (Res_3letter + '_CH2')], [
                               contribs[17], (Res_3letter + '_CZ2')], [contribs[18], (Res_3letter + '_N')]
    if Res_3letter == 'ASP':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_OD1')], [contribs[11], (Res_3letter + '_OD2')], [contribs[12],
                                                                                     (Res_3letter + '_N')],
    if Res_3letter == 'GLU':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD')], [contribs[11], (Res_3letter + '_OE1')], [contribs[12],
                                                                                    (Res_3letter + '_OE2')], [
                               contribs[13], (Res_3letter + '_N')]
    if Res_3letter == 'ASN':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_ND2')], [contribs[11], (Res_3letter + '_OD1')], [contribs[12], (Res_3letter + '_N')]
    if Res_3letter == 'GLN':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD')], [contribs[11], (Res_3letter + '_NE2')], [contribs[12],
                                                                                    (Res_3letter + '_OE1')], [
                               contribs[13], (Res_3letter + '_N')]
    if Res_3letter == 'HIS':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD2')], [contribs[11], (Res_3letter + '_NE2')], [contribs[12],
                                                                                     (Res_3letter + '_CE1')], [
                               contribs[13], (Res_3letter + '_ND1')], [contribs[14], (Res_3letter + '_N')]
    if Res_3letter == 'LYS':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD')], [contribs[11], (Res_3letter + '_CE')], [contribs[12],
                                                                                   (Res_3letter + '_NZ')], [
                               contribs[13], (Res_3letter + '_N')]
    if Res_3letter == 'ARG':
        residue_charges += [contribs[8], (Res_3letter + '_CB')], [contribs[9], (Res_3letter + '_CG')], [contribs[10], (
                    Res_3letter + '_CD')], [contribs[11], (Res_3letter + '_NE')], [contribs[12],
                                                                                   (Res_3letter + '_CZ')], [
                               contribs[13], (Res_3letter + '_NH2')], [contribs[14], (Res_3letter + '_NH1')], [
                               contribs[15], (Res_3letter + '_N')]
    return residue_charges

### Manually added data
def computed_gasteiger_charges():
    c_gasteiger_charges = gasteiger_charges('GLY', 'O=C(NCC(O)=O)CNC(CN)=O') + gasteiger_charges('ALA',
                                                                                           'O=C(NCC(O)=O)C(C)NC(CN)=O') + gasteiger_charges(
    'SER', 'O=C(NCC(O)=O)C(CO)NC(CN)=O') + gasteiger_charges('THR',
                                                             'O=C(NCC(O)=O)C(C(O)C)NC(CN)=O') + gasteiger_charges('CYS',
                                                                                                                  'O=C(NCC(O)=O)C(CS)NC(CN)=O') + gasteiger_charges(
    'VAL', 'O=C(NCC(O)=O)C(C(C)C)NC(CN)=O') + gasteiger_charges('LEU',
                                                                'O=C(NCC(O)=O)C(CC(C)C)NC(CN)=O') + gasteiger_charges(
    'ILE', 'O=C(NCC(O)=O)C(C(C)CC)NC(CN)=O') + gasteiger_charges('MET',
                                                                 'O=C(NCC(O)=O)C(CCSC)NC(CN)=O') + gasteiger_charges(
    'PRO', 'O=C(NCC(O)=O)C1N(C(CN)=O)CCC1') + gasteiger_charges('PHE',
                                                                'O=C(NCC(O)=O)C(CC1=CC=CC=C1)NC(CN)=O') + gasteiger_charges(
    'TYR', 'O=C(NCC(O)=O)C(CC1=CC=C(O)C=C1)NC(CN)=O') + gasteiger_charges('TRP',
                                                                          'O=C(NCC(O)=O)C(CC1=CNC2=C1C=CC=C2)NC(CN)=O') + gasteiger_charges(
    'ASP', 'O=C(NCC(O)=O)C(CC([O-])=O)NC(CN)=O') + gasteiger_charges('GLU',
                                                                     'O=C(NCC(O)=O)C(CCC([O-])=O)NC(CN)=O') + gasteiger_charges(
    'ASN', 'O=C(NCC(O)=O)C(CC(N)=O)NC(CN)=O') + gasteiger_charges('GLN',
                                                                  'O=C(NCC(O)=O)C(CCC(N)=O)NC(CN)=O') + gasteiger_charges(
    'HIS', 'O=C(NCC(O)=O)C(CC1=CN=CN1)NC(CN)=O') + gasteiger_charges('LYS',
                                                                     'O=C(NCC(O)=O)C(CCCC[NH3+])NC(CN)=O') + gasteiger_charges(
    'ARG', 'O=C(NCC(O)=O)C(CCCNC(N)=[NH2+])NC(CN)=O')

    return c_gasteiger_charges

def MSWHIM():
    MSWHIM = [['A', -0.73, 0.20, -0.62], ['M', -0.70, 1.00, -0.32], ['C', -0.66, 0.26, -0.27], ['N', 0.14, 0.20, -0.66],
              ['D', 0.11, -1.00, -0.96], ['P', -0.43, 0.73, -0.60], ['E', 0.24, -0.39, -0.04], ['Q', 0.30, 1.00, -0.30],
              ['F', 0.76, 0.85, -0.34], ['R', 0.22, 0.27, 1.00], ['G', -0.31, -0.28, -0.75], ['S', -0.80, 0.61, -1.00],
              ['H', 0.84, 0.67, -0.78], ['T', -0.58, 0.85, -0.89], ['I', -0.91, 0.83, -0.25], ['V', -1.00, 0.79, -0.58],
              ['K', -0.51, 0.08, 0.60], ['W', 1.00, 0.98, -0.47], ['L', -0.74, 0.72, -0.16], ['Y', 0.97, 0.66, -0.16]]

    return MSWHIM

def STscales():
    STscales = [['A', -1.552, -0.791, -0.627, 0.237, -0.461, -2.229, 0.283, 1.221],
                ['M', -0.693, 0.498, 0.658, 0.457, -0.231, 1.064, 0.248, -0.778],
                ['C', -1.276, -0.401, 0.134, 0.859, -0.196, -0.72, 0.639, -0.857],
                ['N', -0.888, -0.057, -0.651, -0.214, 0.917, 0.164, -0.14, -0.166],
                ['D', -0.907, -0.054, -0.781, -0.248, 1.12, 0.101, -0.245, -0.075],
                ['P', -1.049, -0.407, -0.067, -0.066, -0.813, -0.89, 0.021, -0.894],
                ['E', -0.629, 0.39, -0.38, -0.366, 0.635, 0.514, 0.175, 0.367],
                ['Q', -0.622, 0.228, -0.193, -0.105, 0.418, 0.474, 0.172, 0.408],
                ['F', -0.019, 0.024, 1.08, -0.22, -0.937, 0.57, -0.357, 0.278],
                ['R', -0.059, 0.731, -0.013, -0.096, -0.253, 0.3, 1.256, 0.854],
                ['G', -1.844, -0.018, -0.184, 0.573, -0.728, -3.317, 0.166, 2.522],
                ['S', -1.343, -0.311, -0.917, -0.049, 0.549, -1.533, 0.166, 0.28],
                ['H', -0.225, 0.361, 0.079, -1.037, 0.568, 0.273, 1.208, -0.001],
                ['T', -1.061, -0.928, -0.911, -0.063, 0.538, -0.775, -0.147, -0.717],
                ['I', -0.785, -1.01, -0.349, -0.097, -0.402, 1.091, -0.139, -0.764],
                ['V', -1.133, -0.893, -0.325, 0.303, -0.561, -0.175, -0.02, -0.311],
                ['K', -0.504, 0.245, 0.297, -0.065, -0.387, 1.011, 0.525, 0.553],
                ['W', 0.853, 0.039, 0.26, -1.163, 0.16, -0.202, 1.01, 0.195],
                ['L', -0.826, -0.379, 0.038, -0.059, -0.625, 1.025, -0.229, -0.129],
                ['Y', 0.308, 0.569, 1.1, -0.464, -0.144, -0.354, -1.099, 0.162]]
    return STscales

def Zscales5():
    Zscales5 = [['A', 0.24, -2.32, 0.60, -0.14, 1.30], ['M', -2.85, -0.22, 0.47, 1.94, -0.98],
                ['C', 0.84, -1.67, 3.71, 0.18, -2.65], ['N', 3.05, 1.62, 1.04, -1.15, 1.61],
                ['D', 3.98, 0.93, 1.93, -2.46, 0.75], ['P', -1.66, 0.27, 1.84, 0.70, 2.00],
                ['E', 3.11, 0.26, -0.11, -3.04, -0.25], ['Q', 1.75, 0.50, -1.44, -1.34, 0.66],
                ['F', -4.22, 1.94, 1.06, 0.54, -0.62], ['R', 3.52, 2.50, -3.50, 1.99, -0.17],
                ['G', 2.05, -4.06, 0.36, -0.82, -0.38], ['S', 2.39, -1.07, 1.15, -1.39, 0.67],
                ['H', 2.47, 1.95, 0.26, 3.90, 0.09], ['T', 0.75, -2.18, -1.12, -1.46, -0.40],
                ['I', -3.89, -1.73, -1.71, -0.84, 0.26], ['V', -2.59, -2.64, -1.54, -0.85, -0.02],
                ['K', 2.29, 0.89, -2.49, 1.49, 0.31], ['W', -4.36, 3.94, 0.59, 3.44, -1.59],
                ['L', -4.28, -1.30, -1.49, -0.72, 0.84], ['Y', -2.54, 2.44, 0.43, 0.04, -1.47]]

    return Zscales5

def Zscales3(Zscales5):
    Zscales3 = []
    for AA in Zscales5:
        Zscales3.append(AA[:4])

    return Zscales3

def get_colors():
    colors = {'5ht1b_human': '#ebc19b',  # orange
              '5ht2b_human': '#eb9f59',  # orange
              'acm1_human': '#d19258',  # orange-brown
              'acm2_human': '#bd6d22',  # orange-brown
              'acm4_human': '#91531a',  # orange-brown
              'adrb2_human': '#757575',  # grey-black
              'drd3_human': '#424242',  # grey-black
              'hrh1_human': '#030303',  # grey-black
              'agtr1_human': '#e3908a',
              'ednrb_human': '#e3685f',
              'oprd_human': '#eb4034',
              'oprk_human': '#e31a0b',
              'oprx_human': '#b3180c',
              'ox1r_human': '#ebc19b',
              'ox2r_human': '#eb9f59',
              'par1_human': '#e07c1d',  # red
              'ccr5_human': '#ebe4ab',  # yellow
              'cxcr4_human': '#f5e878',  # yellow
              'cnr1_human': '#b86cf5',  # purple
              'ffar1_human': '#d66dca',  # pink
              'lpar1_human': '#f531de',  # pink
              's1pr1_human': '#b315a0',  # pink
              'aa1r_human': '#9bc5eb',  # blue
              'aa2ar_human': '#5ba0de',  # blue
              'p2ry1_human': '#c2ede8',  # turquoise
              'p2y12_human': '#7ef2e5'  # turquoise
              }
    return colors