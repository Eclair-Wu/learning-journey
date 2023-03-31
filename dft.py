import numpy as np
from pyscf import dft
import scipy as sp
def tran_f(H,X,):
    H_new=X@H@X.T
    dia=sp.linalg.eigh(H_new)
    C_trans=dia[1]
    C=X@C_trans
    return C
def calc_noxc(mol,P):
    hcore=mol.intor('int1e_kin')+mol.intor('int1e_nuc')
    h2e=mol.intor('int2e')
    H=hcore+np.einsum("la, lauv -> uv",P,h2e)
    return H
def dft_scf(mol,grids,inital_guess,xc_code,maxcycle=30):
    P=inital_guess
    X=sp.linalg.fractional_matrix_power(mol.intor('int1e_ovlp'),-0.5)
    h_core=mol.intor('int1e_kin')+mol.intor('int1e_nuc')
    iter=0
    while iter<maxcycle:
        v_xc=dft.numint.nr_vxc(mol,grids,xc_code,P)[2]
        e_xc=dft.numint.nr_vxc(mol,grids,xc_code,P)[1]
        H=calc_noxc(mol,P)+v_xc
        C=tran_f(H,X)
        P_new=2*C[:,:mol.nelectron//2] @ C[:,:mol.nelectron//2].T
        if np.allclose(P,P_new):
            E_0=1/2*np.sum(P*(H-v_xc+h_core))+mol.energy_nuc()+e_xc
            print('SCF done')
            print('DFT_energy_final:',E_0)
            break
        else:
            P=2 * C[:, :mol.nelectron//2] @ C[:, :mol.nelectron//2].T
            iter+=1
            E_0=1/2*np.sum(P*(H-v_xc+h_core))+mol.energy_nuc()+e_xc
            print('iteration:',iter)
            print('DFT_energy:',E_0)
