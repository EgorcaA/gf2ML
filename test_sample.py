from triqs.gf import *
from triqs.operators import *
from triqs.gf import Fourier
from h5 import *
import triqs.utility.mpi as mpi

from triqs.operators import *
from triqs_cthyb import Solver

from triqs.plot.mpl_interface import oplot,plt
from triqs.gf import MeshImTime
from triqs.gf import GfImFreq, GfImTime, make_gf_from_fourier

from triqs.gf import  MeshImTime, MeshReFreq, Gf
from triqs.plot.mpl_interface import oplot, plt

from multiprocessing import Pool
from itertools import repeat
from functools import partial

import csv
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# import pandas as pd

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# from pymatgen.electronic_structure.plotter import BSDOSPlotter, BSPlotter, DosPlotter
# from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

import pickle 
from  tqdm import tqdm

import sys
sys.path.append("./src")
# import DataBase
# import ml_model
# import Sample 
from band_generation import Band_generator
from util import * 


class sample():
    def __init__(self, 
                 beta=5.0, 
                 U=2.0, 
                 num_kpoints=21, 
                 n_iw=500,
                 n_tau=7000, 
                 n_w=500,
                 e_window=4.0,
                 ):

        self.beta = beta
        self.num_kpoints = num_kpoints
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_w = n_w
        self.U = U
        self.e_window = e_window

        self.kpoints= np.linspace(-np.pi, np.pi, 10)
        # kx = np.linspace(-0.5, 0.5, 10)
        # ky = np.linspace(-0.5, 0.5, 10)
        # kx, ky = np.meshgrid(kx, ky)
        # self.kpoints = [kx, ky]

        self.niwmesh = np.linspace(-n_iw-1, n_iw, 2*n_iw)
        self.iwmesh = np.pi*(2*self.niwmesh+1)/self.beta
        self.taumesh = np.linspace(0, self.beta, self.n_tau)
        self.wmesh = np.linspace(-e_window, e_window, self.n_w)
        

        self.Gw = 0
        self.Giw = 0
        self.G_legendre = 0
        self.Gtau = 0
        self.Sigma_iw = 0
        self.get_BS()
        # self.get_Green()



    def get_BS(self):
        t1 = np.random.rand()*1.5
        t2 = np.random.rand()
        t3 = np.random.rand()
        kpts = self.kpoints
        BS = 2 * t1 * np.cos(kpts) + 2 * t2 * np.cos(2 * kpts) + 2 * t3 * np.cos(3 * kpts)
        self.BS = -BS - np.mean(-BS)
        # print(BS)
        # bg = Band_generator()
        # bg.load_checkpoint('./models/AUTOmodel_v3')
        # self.BS = bg.getBS().flatten()


    def get_Green(self, saveQ=False):
        ''' 
        important

        self.G0_iw
        self.G0w
        self.G0tau
        self.G0_legendre
        '''

        iw_mesh = MeshImFreq(beta=self.beta, S='Fermion', n_iw=self.n_iw)
        self.Gks = [Gf(mesh=iw_mesh, target_shape=[1, 1]) for kpt in range(len(self.kpoints))]
        for kpt in range(len(self.kpoints) ):
            self.Gks[kpt] << inverse(iOmega_n - self.BS[kpt] + 0.0001j)
        self.Gks = np.array(self.Gks)

        dSarea = 1.0/(self.num_kpoints)


        ### G0(iw_n)
        self.G0_iw =  dSarea*np.sum( self.Gks, axis=0 )  
        # print(self.G0_iw.data)
        # print(self.G0_iw)

    def solve_impurity(self, n_cycles=5000,
                             length_cycle=2000,
                             n_warmup_cycles=10000, saveQ=False):
        ''' 
        important
        
        self.Gw
        self.Giw
        self.G_legendre
        self.Gtau
        self.Sigma_iw
        
        '''

        U = 2.0        
        mesh = MeshImFreq(beta=self.beta, S='Fermion', n_iw=self.n_iw)
        Gks = [Gf(mesh=mesh, target_shape=[1, 1]) for kpt in range(len(self.BS))]
        for kpt_ind, e in enumerate(self.BS ):
            Gks[kpt_ind] << inverse(iOmega_n - e)
        Gks = np.array(Gks)
        g_local = np.sum( Gks, axis=0 ) /float(len(self.BS))


        S = Solver(beta = self.beta, 
                   n_iw=self.n_iw, 
                   n_tau=self.n_tau, 
                   gf_struct = [ ('up',1), ('down',1) ])
        
        for name, g0 in S.G0_iw:
            g0 <<  g_local

        S.solve(h_int = U * n('up',0) * n('down',0),    # Local Hamiltonian
                n_cycles = 5000,                        # Number of QMC cycles
                length_cycle = 1000,                     # Length of a cycle
                n_warmup_cycles = 1000)                 # How many warmup cycles
        

sample1 = sample(beta=20.0)
sample1.solve_impurity()