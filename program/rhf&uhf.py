import numpy as np
import scipy as sp


class HF:
    def __init__(self, e1, e2, s, nuc, n, k):
        self.ao1e = e1
        self.ao2e = e2
        self.ovlp = s
        self.enuc = nuc
        self.nele = n
        self.nbas = k

    def fock(self, P):
        F = self.ao1e + np.einsum('kl,ijkl -> ij', P, self.ao2e) - \
            np.einsum('kl,ikjl->ij', P, self.ao2e*1/2)
    return F

    def tran_f(F):
        X = np.linalg.fractional_matrix_power(self.ovlp, -0.5)
        F_new = X@F@X.T
        e, C_trans = sp.linalg.eigh(F_new)
        C = X@C_trans
    return C, e

    def conv(self, P, P_new, delta):
        judge = np.sqrt(np.sum(np.square(P_new-P)))/self.nbas
    if judge > delta:
        return False
    else: 
        return True

    def SCF_rhf(self, N, P, maxcycle=30, delta=1e-6):
        iter = 0
        while iter < maxcycle:
            F = self.fock(P)
            C, e = tran_f(F)
            P_new = 2*C[:, :N//2] @ C[:, :N//2].T
            if (conv(P, P_new, delta)):
                F_final = self.fock(P_new)
                E_0 = 1/2*np.sum(P_new*(self.ao1e+F_final))+self.enuc
                print('SCF done')
                print('HF_energy_final:', E_0)
                break
            else:
                P = 2 * C[:, :N//2] @ C[:, :N//2].T
                iter += 1
                E_0 = 1/2*np.sum(P*(self.ao1e+F))+self.enuc
                print('iteration:', iter)
                print('HF_energy:', E_0)
        if iter == maxcycle:
            print('not converge')
        else:
            output = {'energy': E_0, 'Orbital energy': e}
    return output

def calc_f_u(P,P_a,P_b,h2e,h1e,K):
    G_a= h2e.reshape(K**2, K**2) @ P.ravel()-h2e.transpose(0,3,2,1).reshape(K**2, K**2) @ P_a.ravel()
    F_a = G_a.reshape(K, K)+h1e
    G_b= h2e.reshape(K**2, K**2) @ P.ravel()-h2e.transpose(0,3,2,1).reshape(K**2, K**2) @ P_b.ravel()
    F_b = G_b.reshape(K, K)+h1e
    return F_a,F_b

def SCF_uhf(P,h1e,h2e,S,Nuc_replus,K,N_a,N_b,maxcycle=30):
    iter=0
    X=np.linalg.fractional_matrix_power(S,-0.5)
    P_a=np.zeros((K,K))
    P_b=P-P_a
    while iter<maxcycle:
        F_a,F_b=calc_f_u(P,P_a,P_b,h2e,h1e,K)
        C_a=tran_f(F_a,X)[0]
        C_b=tran_f(F_b,X)[0]
        P_anew=C_a[:, :N_a] @ C_a[:, :N_a].T
        P_bnew=C_b[:, :N_b] @ C_b[:, :N_b].T
        P_new=P_anew+P_bnew
        if(conv(P,P_new,K,1e-5)):
            F_a,F_b=calc_f_u(P_new,P_anew,P_bnew,h2e,h1e,K)
            E=1/2*np.sum((P_anew+P_bnew)*h1e+F_a*P_anew+F_b*P_bnew)+Nuc_replus
            print('SCF done')
            print('HF_energy_final:',E)
            break
        else:
            P_a=C_a[:,:N_a] @ C_a[:, :N_a].T
            P_b=C_b[:,:N_b] @ C_b[:, :N_b].T
            P=P_a+P_b
            iter+=1
            print(iter)
            print('iteration:',iter)
            E=1/2*np.sum((P_a+P_b)*h1e+F_a*P_a+F_b*P_b)+Nuc_replus
            print('HF_energy:',E)
    if iter==maxcycle:
        print('cant converge')
    return E