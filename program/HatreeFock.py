import numpy as np
import scipy as sp


class GHF(object):
    def __init__(self, e1, e2, s, nuc, na, nb, k):
        self.ao1e = sp.linalg.block_diag(e1, e1)
        self.ao2e = e2
        self.ovlp = sp.linalg.block_diag(s, s)
        self.enuc = nuc
        self.elea = na
        self.eleb = nb
        self.nao = k
        self.jk = np.zeros((2*k, 2*k), dtype=np.float64)
        self.fock = np.zeros((2*k, 2*k), dtype=np.float64)
        self.den = np.zeros((2*k, 2*k), dtype=np.float64)

    def getfock(self):
        coul = np.einsum(
            'ij,klij ->kl', self.den[:self.nao, :self.nao]+self.den[self.nao:, self.nao:], self.ao2e)
        self.jk[:self.nao, :self.nao] = coul-np.einsum(
            'ij,kjil ->kl', self.den[:self.nao, :self.nao], self.ao2e)
        self.jk[self.nao:, self.nao:] = coul-np.einsum(
            'ij,kjil ->kl', self.den[self.nao:, self.nao:], self.ao2e)
        self.jk[:self.nao, self.nao:] = -np.einsum(
            'ij,kjil ->kl', self.den[:self.nao, self.nao:], self.ao2e)

        self.jk[self.nao:, :self.nao] = self.jk[:self.nao, self.nao:]

        self.fock = self.jk+self.ao1e
        return self.fock

    def getene(self):
        ene = 1/2*np.trace((self.fock+self.ao1e)@self.den)+self.enuc
        return ene

    def getden(self, C):
        C_occ = C[:, :self.elea+self.eleb]
        self.den = np.einsum(
            'ik,jk ->ij', C_occ, C_occ)
        return self.den

    def scf(self, maxcycle=30, delta=1e-6):
        iter = 0
        while iter < maxcycle:
            E = 1/2*np.trace((self.fock+self.ao1e)@self.den)+self.enuc
            self.getfock()
            e, C = sp.linalg.eigh(self.fock, self.ovlp)
            self.getden(C)
            if (np.abs(E-self.getene()) < delta):
                print('SCF done')
                print('HF_energy_final:', self.getene())
                break
            else:
                iter += 1
                print('iteration:', iter)
                print('HF_energy:', self.getene())
        if iter == maxcycle:
            print('not converge')
        else:
            print('converge in ', iter, ' cycles')
        output = {'energy': self.getene(), 'Orbital energy': e}
        return output


class RHF(GHF):

    def calc_f_u(P, P_a, P_b, h2e, h1e, K):
        G_a = h2e.reshape(K**2, K**2) @ P.ravel()-h2e.transpose(0,
                                                                3, 2, 1).reshape(K**2, K**2) @ P_a.ravel()
        F_a = G_a.reshape(K, K)+h1e
        G_b = h2e.reshape(K**2, K**2) @ P.ravel()-h2e.transpose(0,
                                                                3, 2, 1).reshape(K**2, K**2) @ P_b.ravel()
        F_b = G_b.reshape(K, K)+h1e
        return F_a, F_b

    def SCF_uhf(P, h1e, h2e, S, Nuc_replus, K, N_a, N_b, maxcycle=30):
        iter = 0
        X = np.linalg.fractional_matrix_power(S, -0.5)
        P_a = np.zeros((K, K))
        P_b = P-P_a
        while iter < maxcycle:
            F_a, F_b = calc_f_u(P, P_a, P_b, h2e, h1e, K)
            C_a = tran_f(F_a, X)[1]
            C_b = tran_f(F_b, X)[2]
            P_anew = C_a[:, :N_a] @ C_a[:, :N_a].T
            P_bnew = C_b[:, :N_b] @ C_b[:, :N_b].T
            P_new = P_anew+P_bnew
        if (conv(P, P_new, K, 1e-5)):
            F_a, F_b = calc_f_u(P_new, P_anew, P_bnew, h2e, h1e, K)
            E = 1/2*np.sum((P_anew+P_bnew)*h1e+F_a *
                           P_anew+F_b*P_bnew)+Nuc_replus
            print('SCF done')
            print('HF_energy_final:', E)
           # break
        else:
            P_a = C_a[:, :N_a] @ C_a[:, :N_a].T
            P_b = C_b[:, :N_b] @ C_b[:, :N_b].T
            P = P_a+P_b
            iter += 1
            print(iter)
            print('iteration:', iter)
            E = 1/2*np.sum((P_a+P_b)*h1e+F_a*P_a+F_b*P_b)+Nuc_replus
            print('HF_energy:', E)
        if iter == maxcycle:
            print('cant converge')
        return E
