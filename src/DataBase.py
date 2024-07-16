from triqs.operators import *
from triqs_cthyb import Solver
from triqs.gf import *
from triqs.gf.descriptors import MatsubaraToLegendre

from h5 import *
from functools import partial
from multiprocessing import Pool, Manager
import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
import sys


from Sample import sample
from util import * 


import torch
import torch.nn as nn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from ml_model import MLSolver    


class db_DMFT():
    """
    Class for the database object
    """

    def __init__(self, beta, U, n_entries, filename, 
                 params=None, samples=None):
        """
        Initialise the database. 

        Parameters
        ----------

        beta: list of length 2 of reals
        (Inverse temparature) Range betwen beta_min and beta_max

        U: list of length 2 of reals
        Range betwen U_min and U_max

        n_entries: integer
        number of entries per core in the database

        filename: string 
        name of the h5 file for the database
        
        """
        
        self.n_entries = n_entries
        self.filename = filename

        self.data_entries = []

        if params is not None:
            self.params= params
            for smple in samples:
                smpl  = sample.fromdict(smple)
                self.data_entries.append(smpl)
        else:
            self.params = [{'U': np.random.random()*(U[1] - U[0])+U[0],
                    'beta':np.random.random()*(beta[1] - beta[0])+beta[0]}
                    for sample in range(n_entries)]
            
            self.fill_db()

            
    def fill_db(self):

        for params in self.params:
            smpl  = sample(beta=params['beta'], U=params['U'] )
            self.data_entries.append(smpl)
    

    @staticmethod    
    def start_solving(sample, 
                    n_cycles,
                    length_cycle,
                    n_warmup_cycles):
        sample.solve_impurity(n_cycles, length_cycle, n_warmup_cycles)
        # print(sample.Giw)
        sys.stdout.flush()
        return sample


    def solve_db(self,n_cycles=5000,
                      length_cycle=200,
                      n_warmup_cycles=10000, mp=True):
        
        if mp:
            n_workers = 8
            with Manager() as manager:
                # Convert data_entries to a list proxy object
                data_entries = manager.list(self.data_entries)
                with Pool(n_workers) as pool:
                    # ss = pool.map(partial(self.start_solving, n_cycles=n_cycles, 
                                                     # length_cycle=length_cycle,
                                                     # n_warmup_cycles=n_warmup_cycles), data_entries)
                    results = list(tqdm(pool.imap(partial(self.start_solving,
                                          n_cycles=n_cycles,
                                          length_cycle=length_cycle,
                                          n_warmup_cycles=n_warmup_cycles),
                                 data_entries),\
                       total=len(data_entries)))
                self.data_entries = results

        else:
            for sample in tqdm(self.data_entries):
                sample.solve_impurity(n_cycles,\
                                      length_cycle,\
                                      n_warmup_cycles)

    def save_db(self):
        self.smpls_list = []
        for smpl_i, sample in enumerate(self.data_entries):
            smpl_dict = sample.save2dict()
            self.smpls_list.append( smpl_dict)

        with HDFArchive(self.filename) as ar:
                ar['n_entries'] = self.n_entries
                ar['filename'] = self.filename
                ar["params"] = self.params
                ar["samples"] = self.smpls_list
    

    @classmethod
    def fromh5(cls, filename):
        "Initialize db from a h5 file"
        with HDFArchive(filename,'r') as datadict:
            return cls(**datadict)


    def init_ML(self, model_path='./model_v1'):
        self.MLmodel = MLSolver().to(device, torch.float32)
        self.MLmodel.load_checkpoint(checkpoint_path=model_path)


    def plot_ML_solutions(self, numV=1, numH=1, what='dens'):
        
        fig, axes = plt.subplots(numV, numH, figsize=(2.5*numH, 1.5*numV))

        sample_num = 0
        for x in range(numV):
            for y in range(numH):
                
                sample = self.data_entries[sample_num]

                gl = np.real(sample.G0_legendre.data).flatten()
                beta = self.params[sample_num]['beta']
                U = self.params[sample_num]['U']
                features = np.hstack([U, beta, gl])
                leg = self.MLmodel(torch.Tensor(features)).cpu().detach().numpy()
                
                meshQ = MeshLegendre(beta, 'Fermion', 30)
                Gpredicted = Gf(mesh=meshQ, target_shape=[1,1])
                Gpredicted.data[:] = leg[:, np.newaxis, np.newaxis]
                
                tau_mesh = MeshImTime(beta=beta, statistic='Fermion', n_tau=500)
                GtauML = Gf(mesh=tau_mesh, target_shape=[1,1])
                GtauML.set_from_legendre(Gpredicted)

                iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=sample.n_iw)
                G_iw_ML = Gf(mesh=iw_mesh, target_shape=[1,1])
                G_iw_ML.set_from_legendre(Gpredicted)
                
                
                if what == 'dens':
                    w_mesh = MeshReFreq(window=(-4,4), n_w=sample.n_w)
                    G0w = Gf(mesh=w_mesh, target_shape=[1,1])
                    Gw_ML = Gf(mesh=w_mesh, target_shape=[1,1])
                    GwExact = Gf(mesh=w_mesh, target_shape=[1,1])
                    
                    
                    n_points = findExtrema(np.imag(sample.G0_iw.data.flatten()))*3
                    G0w.set_from_pade(sample.G0_iw, n_points = n_points, freq_offset = 0.005)
                    
                    n_points = findExtrema(np.imag(G_iw_ML.data.flatten()))*3
                    GwExact.set_from_pade(sample.Giw, n_points = n_points, freq_offset = 0.05)
                    Gw_ML.set_from_pade(G_iw_ML, n_points = n_points, freq_offset = 0.05)
                    

                    axes[x, y].plot(sample.wmesh, -1/np.pi*np.imag(Gw_ML.data.flatten()),'-',color='r', label='Interact. ML')
                    axes[x, y].plot(sample.wmesh, -1/np.pi*np.imag(GwExact.data.flatten()),'-',color='g', label='Interact. Exact')
                    axes[x, y].plot(sample.wmesh, -1/np.pi*np.imag(G0w.data.flatten()),'--',color='b', label='Noninteracting GF')
                    axes[x, y].set_title(f'U={self.params[sample_num]['U']:.2f} beta={self.params[sample_num]['beta']:.2f}')
                
                elif what == 'tau':

                    ntau = 500
                    taumesh = np.linspace(0, self.data_entries[sample_num].beta, ntau)
                    G_imp_rebinned = self.data_entries[sample_num].Gtau.rebinning_tau(new_n_tau=ntau)

                    axes[x, y].plot(self.data_entries[sample_num].taumesh, np.real(self.data_entries[sample_num].G0tau.data.flatten()),
                                    '--',color='b', label='Noninteracting GF')
                    axes[x, y].plot(taumesh, np.real(G_imp_rebinned.data.flatten()),'-',color='b', label='Interact. Exact')
                    axes[x, y].plot(taumesh, np.real(GtauML.data.flatten()),'-',color='r', label='Interact. ML')


                    axes[x, y].set_title(f'U={self.params[sample_num]['U']:.2f} beta={self.params[sample_num]['beta']:.2f}')
                
                axes[x, y].xaxis.set_ticklabels([])
                axes[x, y].yaxis.set_ticklabels([])
                if sample_num==0:
                    axes[x, y].legend()
                # axes[x, y].set_ylabel('G')

                sample_num += 1

        fig.tight_layout()
        
        # width = 10
        # fig.set_figwidth(width)     #  ширина и
        # fig.set_figheight(width/1.6)    #  высота "Figure"
        # fig.set_label('Gw')
        # plt.legend()
        # plt.xlim(-10, 10)
        # plt.savefig('./Green_up.png', dpi=200, bbox_inches='tight')

        plt.show()

    
    def plot_exact_solutions(self, numV=1, numH=1, what='dens'):
        
        fig, axes = plt.subplots(numV, numH, figsize=(2.5*numH, 1.5*numV))
        

        sample_num = 0
        for x in range(numV):
            for y in range(numH):
                # print(np.real(self.G0_iw))

                
                if what == 'dens':
                    axes[x, y].plot(self.data_entries[sample_num].wmesh, -1/np.pi*np.imag(self.data_entries[sample_num].Gw.data.flatten()),'-',color='g', label='I')
                    axes[x, y].plot(self.data_entries[sample_num].wmesh, -1/np.pi*np.imag(self.data_entries[sample_num].G0w.data.flatten()),'--',color='b', label='NI')
                    axes[x, y].set_title(f'U={self.params[sample_num]['U']:.2f} beta={self.params[sample_num]['beta']:.2f}')
                elif what== 'Gw':
                    axes[x, y].plot(self.data_entries[sample_num].wmesh, np.real(self.data_entries[sample_num].Gw.data.flatten()),'-',color='b', label='G re')
                    axes[x, y].plot(self.data_entries[sample_num].wmesh, np.imag(self.data_entries[sample_num].Gw.data.flatten()),'-',color='g', label='G im')
                    axes[x, y].plot(self.data_entries[sample_num].wmesh, np.real(self.data_entries[sample_num].G0w.data.flatten()),'--',color='b', label='G0 re')
                    axes[x, y].plot(self.data_entries[sample_num].wmesh, np.imag(self.data_entries[sample_num].G0w.data.flatten()),'--',color='g', label='G0 im')
                    axes[x, y].set_title(f'U={self.params[sample_num]['U']:.2f} beta={self.params[sample_num]['beta']:.2f}')
                    # axes[x, y].set_xlabel('w, eV')

                elif what== 'Giw':
                    axes[x, y].plot(self.data_entries[sample_num].iwmesh, np.real(self.data_entries[sample_num].Giw.data.flatten()),'-',color='b', label='G iw re')
                    axes[x, y].plot(self.data_entries[sample_num].iwmesh, np.imag(self.data_entries[sample_num].Giw.data.flatten()),'-',color='g', label='G iw im')
                    axes[x, y].plot(self.data_entries[sample_num].iwmesh, np.real(self.data_entries[sample_num].G0_iw.data.flatten()),'--',color='b', label='G0 iw re')
                    axes[x, y].plot(self.data_entries[sample_num].iwmesh, np.imag(self.data_entries[sample_num].G0_iw.data.flatten()),'--',color='g', label='G0 iw im')
                    axes[x, y].set_title(f'U={self.params[sample_num]['U']:.2f} beta={self.params[sample_num]['beta']:.2f}')
                    axes[x, y].set_xlim((-10, 10))

                elif what == 'tau':
                    # axes[x, y].plot(self.data_entries[sample_num].taumesh, np.real(self.data_entries[sample_num].Gtau.data.flatten()),'-',color='g', label='I')
                    # ntau = len(self.data_entries[sample_num].Gtau.data.flatten())
                    ntau = 500
                    taumesh = np.linspace(0, self.data_entries[sample_num].beta, ntau)
                    G_imp_rebinned = self.data_entries[sample_num].Gtau.rebinning_tau(new_n_tau=ntau)

                    axes[x, y].plot(self.data_entries[sample_num].taumesh, np.real(self.data_entries[sample_num].G0tau.data.flatten()),'--',color='b', label='NI')
                    axes[x, y].plot(taumesh, np.real(G_imp_rebinned.data.flatten()),'-',color='b', label='I')
                    axes[x, y].set_title(f'U={self.params[sample_num]['U']:.2f} beta={self.params[sample_num]['beta']:.2f}')

                elif what== 'BS':
                    axes[x, y].plot(self.data_entries[sample_num].kpoints, self.data_entries[sample_num].BS,'-',color='b')
                axes[x, y].xaxis.set_ticklabels([])
                axes[x, y].yaxis.set_ticklabels([])
                if sample_num==0:
                    axes[x, y].legend()
                # axes[x, y].set_ylabel('G')

                sample_num += 1

        fig.tight_layout()
        
        # width = 10
        # fig.set_figwidth(width)     #  ширина и
        # fig.set_figheight(width/1.6)    #  высота "Figure"
        # fig.set_label('Gw')
        # plt.legend()
        # plt.xlim(-10, 10)
        # plt.savefig('./Green_up.png', dpi=200, bbox_inches='tight')

        plt.show()

