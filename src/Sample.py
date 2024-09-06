from triqs.operators import *
from triqs_cthyb import Solver

# from triqs.gf import  MeshImTime, MeshReFreq, iOmega_n, inverse, GfLegendre, MeshImFreq, Gf, GfImFreq, GfImTime, Fourier
from triqs.gf import *
from triqs.gf.descriptors import MatsubaraToLegendre
# from triqs.plot.mpl_interface import plt

from h5 import *
from functools import partial
from multiprocessing import Pool, Manager
import numpy as np

import matplotlib.pyplot as plt
from  tqdm import tqdm
import sys
import time

from util import * 
from triqs_maxent import *


class sample():
    def __init__(self, 
                 beta=5.0, 
                 U=2.0, 
                 n_iw=1025,
                 n_tau=2200, 
                 n_w=500,
                 e_window=4.0,
                 BS=None,
                 G0_iw=None,
                 G0_tau=None,
                 G0_l=None,
                 G_l=None,
                 G_tau=None,
                 time2solve=None,
                 averageorder=None
                 ):

        self.beta = beta
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.U = U
        self.e_window = e_window
        self.n_w = n_w
        self.BS = BS
        assert BS is not None, "BS not defined!"

        self.niwmesh = np.linspace(-n_iw-1, n_iw, 2*n_iw)
        self.iwmesh = np.pi*(2*self.niwmesh+1)/self.beta
        self.taumesh = np.linspace(0, self.beta, self.n_tau)
        self.wmesh = np.linspace(-e_window, e_window, self.n_w)
        

        if G0_iw is not None:
            self.G0_iw = G0_iw
            self.G0_tau = G0_tau
            self.G0_l = G0_l
            self.G_l = G_l
            self.G_tau = G_tau
            self.time2solve = time2solve
            self.averageorder = averageorder
        else:
            self.G_l = 0
            self.G_tau = 0
            self.get_Green()

        
    @classmethod
    def fromdict(cls, datadict):
        "Initialize sample from a dict's items"
        return cls(**datadict)

    def save2dict(self):
        dict2save = {}
        dict2save['beta'] = self.beta 
        dict2save['U'] = self.U
        dict2save['BS'] = self.BS 
        dict2save['n_iw'] = self.n_iw
        dict2save['n_tau'] = self.n_tau 
        dict2save['n_w'] = self.n_w
        dict2save['e_window'] = self.e_window
        dict2save['G0_iw'] = self.G0_iw
        dict2save['G0_tau'] = self.G0_tau
        dict2save['G0_l'] = self.G0_l
        dict2save['G_tau'] = self.G_tau
        dict2save['G_l'] = self.G_l
        dict2save['time2solve'] = self.time2solve
        dict2save['averageorder'] = self.averageorder
        return dict2save


    def get_Green(self, saveQ=False):
        ''' 
        important

        self.G0_iw
        self.G0_tau
        self.G0_l
        '''
        ### G0(iw_n)
        iw_mesh = MeshImFreq(beta=self.beta, S='Fermion', n_iw=self.n_iw)
        Gks = [Gf(mesh=iw_mesh, target_shape=[1, 1]) for kpt in range(len(self.BS))]
        for kpt_ind, e in enumerate(self.BS ):
            Gks[kpt_ind] << inverse(iOmega_n - e )
        self.G0_iw = np.sum( np.array(Gks), axis=0 ) /float(len(self.BS))


        ### G0(tau)
        tau_mesh = MeshImTime(beta=self.beta, statistic='Fermion', n_tau=self.n_tau)
        #Mesh-points are evenly distributed in the interval [0,beta] including points at both edges.
        self.G0_tau = Gf(mesh=tau_mesh, target_shape=[1,1])
        self.G0_tau << Fourier(self.G0_iw)

        ### G0(legendre)
        self.G0_l = GfLegendre(indices = [1], beta = self.beta, n_points = 40)
        self.G0_l << MatsubaraToLegendre(self.G0_iw)

        if saveQ:
            with HDFArchive("features"+str(1) +".h5") as ar:
                ar["G0_tau-%s"%1] = self.G0_tau
                ar["G0_iw-%s"%1] = self.G0_iw
                ar["G0_l-%s"%1] = self.G0_l
            

    def solve_impurity(self, n_cycles=500,
                             length_cycle=2000,
                             n_warmup_cycles=10000, saveQ=False):
        ''' 
        important
        
        self.G_tau
        '''
        # Create a solver instance
        gf_struct = [ ('up', 1), ('down', 1) ]
        S = Solver(beta = self.beta, gf_struct = gf_struct)
        h_int = self.U * n('up',0) * n('down',0)

        for name, g0 in S.G0_iw:
            g0 << self.G0_iw

        start = time.time()
        S.solve(h_int = h_int,
                n_cycles  = n_cycles,
                length_cycle = length_cycle,
                n_warmup_cycles = n_warmup_cycles,
                measure_G_l=True)
        end = time.time()
        self.time2solve = end - start
        self.averageorder = S.average_order
        g = 0.5 * (S.G_iw['up'] + S.G_iw['down'])
                
        self.G_tau = 0.5 * (S.G_tau['up'] + S.G_tau['down'])#.rebinning_tau(new_n_tau=self.n_tau)
        self.G_l = 0.5 * (S.G_l['up'] + S.G_l['down'])

        if saveQ:
            with HDFArchive("targets"+str(1) +".h5") as ar:
                ar["G_tau-%s"%1] =self.G_tau
                ar["G_l-%s"%1] =  0.5 * (S.G_l['up'] + S.G_l['down'])
        


    # def plotBS(self):

    #     fig = plt.figure(figsize=(4,4))
    #     ax = fig.add_subplot(111, projection='3d')

    #     kx = np.linspace(-0.5, 0.5, 10)
    #     ky = np.linspace(-0.5, 0.5, 10)
    #     kx, ky = np.meshgrid(kx, ky)

    #     ax.plot_surface(kx, ky, np.reshape(self.BS, (10, 10)), cmap='viridis')

    #     # Add labels
    #     ax.set_xlabel('kx')
    #     ax.set_ylabel('ky')
    #     ax.set_zlabel('Energy')
    #     # ax.set_zlim((-4, 2))
    #     # Show the plot
    #     plt.show()


    # def plot_DOS(self, sigma=0.2):
    #     energy_min = -3  # Minimum energy value
    #     energy_max = 3  # Maximum energy value
    #     n_points = 100   # Number of points in the energy grid
    #     energy_grid = np.linspace(energy_min, energy_max, n_points)

    #     # sigma = 0.5

    #     # Step 4: Initialize the DOS array
    #     dos = np.zeros_like(energy_grid)

    #     # Step 5: Sum the Gaussians centered at each discrete energy level
    #     for energy in self.BS.flatten():
    #         dos += gaussian(energy_grid, energy, sigma)
    #     dos_integral = np.trapz(dos, energy_grid)  # Compute the integral of the DOS
    #     dos /= dos_integral  # Normalize the DOS
    #     # Step 6: Plot the DOS

    #     A0ens, A0 =  get_Gw(self.G0_tau)
    #     plt.plot(A0ens, A0, label='from G0')

    #     # plt.plot(self.wmesh, -np.imag(self.G0w.data.flatten())/dos_integral2, label='from G0')
    #     plt.plot(energy_grid, dos, label=f'Gaussian Broadened DOS (sigma={sigma})')
    #     if self.G_tau != 0:
    #         Aens, A =  get_Gw(self.G_tau)
    #         plt.plot(Aens, A, label=f'from G')

    #     plt.xlim((-3, 3))
    #     plt.xlabel('Energy')
    #     plt.ylabel('Density of States (DOS)')
    #     plt.title('Gaussian Broadened Density of States')
    #     plt.legend()
    #     plt.show()


    # def plot_G0(self, which='iw'):
    #     if which == 'iw':
    #         fig, dd = plt.subplots(figsize=(6,3))
    #         # print(np.real(self.G0_iw))
    #         dd.plot(self.niwmesh, np.real(self.G0_iw.data.flatten()), label='re')
    #         dd.plot(self.niwmesh, np.imag(self.G0_iw.data.flatten()), label='im')
    #         # width = 10
    #         # fig.set_figwidth(width)     #  ширина и
    #         # fig.set_figheight(width/1.6)    #  высота "Figure"
    #         fig.set_label('G0_iw')
    #         plt.legend()
    #         # plt.xlim(-10, 10)

    #         plt.ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.

    #         # plt.xlabel("X Label", fontsize=10)
    #         # plt.ylabel("Y Label", fontsize=10)
    #         # plt.colorbar()
    #         # plt.savefig('./Green_up.png', dpi=200, bbox_inches='tight')

    #         plt.show()

    #     elif which == 'w':
    #         fig, dd = plt.subplots(figsize=(6,3))
    #         # print(np.real(self.G0_iw))
    #         dd.plot(self.wmesh, np.real(self.G0w.data.flatten()), label='re')
    #         dd.plot(self.wmesh, np.imag(self.G0w.data.flatten()), label='im')
    #         # width = 10
    #         # fig.set_figwidth(width)     #  ширина и
    #         # fig.set_figheight(width/1.6)    #  высота "Figure"
    #         fig.set_label('G0w')
    #         plt.legend()
    #         # plt.xlim(-10, 10)

    #         plt.ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.

    #         # plt.xlabel("X Label", fontsize=10)
    #         # plt.ylabel("Y Label", fontsize=10)
    #         # plt.colorbar()
    #         # plt.savefig('./Green_up.png', dpi=200, bbox_inches='tight')

    #         plt.show()
    #     elif which == 'tau':
    #         fig, dd = plt.subplots(figsize=(6,3))
    #         # print(np.real(self.G0_iw))
    #         dd.plot(self.taumesh, np.real(self.G0_tau.data.flatten()), label='re')
    #         dd.plot(self.taumesh, np.imag(self.G0_tau.data.flatten()), label='im')
    #         # width = 10
    #         # fig.set_figwidth(width)     #  ширина и
    #         # fig.set_figheight(width/1.6)    #  высота "Figure"
    #         fig.set_label('G0_tau')
    #         plt.legend()
    #         # plt.xlim(-10, 10)

    #         plt.ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.

    #         # plt.xlabel("X Label", fontsize=10)
    #         # plt.ylabel("Y Label", fontsize=10)
    #         # plt.colorbar()
    #         # plt.savefig('./Green_up.png', dpi=200, bbox_inches='tight')

    #         plt.show()

    #     elif which == 'legendre':
    #         fig, dd = plt.subplots(figsize=(6,3))
    #         # print(np.real(self.G0_iw))
    #         dd.scatter(np.arange(0, 40, 1), np.real(self.G0_l.data.flatten()), label='re')
    #         dd.scatter(np.arange(0, 40, 1), np.imag(self.G0_l.data.flatten()), label='im')
    #         # width = 10
    #         # fig.set_figwidth(width)     #  ширина и
    #         # fig.set_figheight(width/1.6)    #  высота "Figure"
    #         fig.set_label('G0leg')
    #         plt.legend()
    #         # plt.xlim(-10, 10)

    #         plt.ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.

    #         # plt.xlabel("X Label", fontsize=10)
    #         # plt.ylabel("Y Label", fontsize=10)
    #         # plt.colorbar()
    #         # plt.savefig('./Green_up.png', dpi=200, bbox_inches='tight')

    #         plt.show()

