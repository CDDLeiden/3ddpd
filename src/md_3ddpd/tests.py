import os
import pytest
from parameterized import parameterized
from unittest import TestCase
from .utils import *
from .DynDescriptor import *
from .StructBindingPocket import *


def test_read_trajectory():
    # read GPCRmd trajectory
    traj = read_trajectory(f'{os.path.dirname(__file__)}/test_files', '49_aa2ar_wt_1')
    traj_protein = traj.restrict_atoms(traj.topology.select("protein"))
    assert traj.n_frames > 0
    assert traj_protein.n_residues > 0

    # read mutant trajectory
    traj_mut = read_trajectory(f'{os.path.dirname(__file__)}/test_files', '49_aa2ar_L85A_1')
    traj_mut_protein = traj_mut.restrict_atoms(traj_mut.topology.select("protein"))
    assert traj_mut.n_frames > 0
    assert traj_mut_protein.n_residues > 0

    # check that wt amd mutant of the same system have the same number of protein residues and frames, but not atoms
    assert traj.n_frames == traj_mut.n_frames
    assert traj_protein.n_residues == traj_mut_protein.n_residues
    #assert traj_protein.n_atoms != traj_mut_protein.n_atoms

# @pytest.mark.parametrize(["hierarchy", "species_input", "output_type", "precision", "allosteric"],
#     make([['None','gpcrdbA','family','subfamily','target'],
#         ['Homo sapiens', 'all_species'],
#         ['a','b','c','d','e'],
#         [50,90],
#         [0,50]], length=3)
#                          )
#
# @parameterized.expand([
#     (hierarchy,species_input,output_type,precision,allosteric)
#     for ])

class TestStructBindingPocket(TestCase):
    def validate_binding_pocket(self, pocket):
        self.assertTrue(len(pocket) > 0)
def test_custom_binding_pocket():

def test_custom_msa():

def test_calculate_rs3ddpd():

def test_calculate_ps3ddpd():