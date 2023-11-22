import numpy as np
import scipy as sp


class GHF(object):
    def __init__(self, e1, e2, s, nuc, n, k):
        self.ao1e = sp.linalg.block_diag(e1, e1)
        self.ao2e = e2
        self.ovlp = sp.linalg.block_diag(s, s)
        self.enuc = nuc
        self.ele = n
        self.nao = k
        self.jk = np.zeros_like(self.ao1e)
        self.fock = np.zeros_like(self.ao1e)
        self.den = np.zeros_like(self.ao1e)
        self.mo_ene = np.zeros(2*k)
        self.mocoeff = np.zeros_like(self.ao1e)

    def getfock(self):
        coul = np.einsum(
            'ij,klij ->kl', self.den[:self.nao, :self.nao]+self.den[self.nao:, self.nao:], self.ao2e)
        self.jk[:self.nao, :self.nao] = coul-np.einsum(
            'ij,kjil ->kl', self.den[:self.nao, :self.nao], self.ao2e)
        self.jk[self.nao:, self.nao:] = coul-np.einsum(
            'ij,kjil ->kl', self.den[self.nao:, self.nao:], self.ao2e)
        self.jk[:self.nao, self.nao:] = -np.einsum(
            'ij,kjil ->kl', self.den[:self.nao, self.nao:], self.ao2e)

        self.jk[self.nao:, :self.nao] = self.jk[:self.nao, self.nao:].conj()

        self.fock = self.jk+self.ao1e
        return self.fock

    def getene(self):
        ene = 1/2*np.trace((self.fock+self.ao1e)@self.den)+self.enuc
        return ene

    def getden(self):
        C_occ = self.mocoeff[:, :self.ele]
        self.den = np.einsum(
            'ik,jk ->ij', C_occ.conj(), C_occ)
        return self.den

    def scf(self, maxcycle=30, delta=1e-6, detail=False):
        iter = 0
        while iter < maxcycle:
            E = 1/2*np.trace((self.fock+self.ao1e)@self.den)+self.enuc
            self.getfock()
            self.mo_ene, self.mocoeff = sp.linalg.eigh(self.fock, self.ovlp)
            self.getden()
            if (np.abs(E-self.getene()) < delta):
                print('SCF done')
                print('HF_energy_final:', self.getene())
                break
            else:
                iter += 1
                if detail:
                    print('iteration:', iter)
                    print('HF_energy:', self.getene())
        if iter == maxcycle:
            print('not converge')
        else:
            print('converge in ', iter, ' cycles')
        output = {'energy': self.getene()}
        return output


class RHF(GHF):
    def __init__(self, e1, e2, s, nuc, n, k):
        self.ao1e = e1
        self.ao2e = e2
        self.ovlp = s
        self.enuc = nuc
        self.ele = n
        self.nao = k
        self.jk = np.zeros_like(self.ao1e)
        self.fock = np.zeros_like(self.ao1e)
        self.den = np.zeros_like(self.ao1e)
        self.mo_ene = np.zeros(k)
        self.mocoeff = np.zeros_like(self.ao1e)

    def getden(self):
        C_occ = self.mocoeff[:, :self.ele//2]
        self.den = 2*np.einsum(
            'ik,jk ->ij', C_occ.conj(), C_occ)
        return self.den

    def getfock(self):
        self.jk = np.einsum('ij,klij ->kl', self.den, self.ao2e) - \
            np.einsum('ij,kjil ->kl', self.den, 1/2*self.ao2e)
        self.fock = self.jk+self.ao1e
        return self.fock


class UHF(GHF):
    def __init__(self, e1, e2, s, nuc, n, k):
        self.ao1e = sp.linalg.block_diag(e1, e1)
        self.ao2e = e2
        self.ovlp = sp.linalg.block_diag(s, s)
        self.enuc = nuc
        self.ele = n
        self.nao = k
        self.jk = np.zeros_like(self.ao1e)
        self.fock = np.zeros_like(self.ao1e)
        self.den = np.zeros_like(self.ao1e)
        self.mo_ene = np.zeros(2*k)
        self.mocoeff = np.zeros_like(self.ao1e)

    def getfock(self):
        coul = np.einsum(
            'ij,klij ->kl', self.den[:self.nao, :self.nao]+self.den[self.nao:, self.nao:], self.ao2e)
        self.jk[:self.nao, :self.nao] = coul-np.einsum(
            'ij,kjil ->kl', self.den[:self.nao, :self.nao], self.ao2e)
        self.jk[self.nao:, self.nao:] = coul-np.einsum(
            'ij,kjil ->kl', self.den[self.nao:, self.nao:], self.ao2e)
        self.fock = self.jk+self.ao1e
        return self.fock

    def getden(self):
        C_occ = self.mocoeff[:, :self.ele]
        self.den = np.einsum(
            'ik,jk ->ij', C_occ.conj(), C_occ)
        return self.den
