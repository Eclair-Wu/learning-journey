import numpy as np
from itertools import combinations
import scipy as sp


class FCI(object):
    def __init__(self, e1, e2, n, k, nuc):
        self.mo1e = e1
        self.j = e2.transpose(0, 2, 1, 3)
        self.k = e2.transpose(0, 2, 3, 1)
        self.nbas = 2*k
        self.nele = n
        self.enuc = nuc

    def mtx_ele(self, config1, config2):  # evaluate matrix element
        diff = tuple(config1 ^ config2)
        config1 = list(config1)
        config2 = list(config2)

        if len(diff) == 0:
            mtx = self.slt_cd_1(config1)
        elif len(diff) == 2:
            mtx = self.slt_cd_2(config1, diff) * \
                self.det_sign(config1, config2, diff)
        elif len(diff) == 4:
            mtx = self.slt_cd_3(config1, config2, diff) * \
                self.det_sign(config1, config2, diff)
        else:
            mtx = 0
        return mtx

    def fci(self):
        config = []
        orb = np.linspace(0, self.nbas-1, self.nbas, dtype=np.int64)
        for a in combinations(orb, self.nele):
            config.append(set(a))
        lenth = len(config)
        mtx = np.zeros((lenth, lenth), dtype=np.float64)
        for i in range(lenth):
            for j in range(lenth):
                mtx[i, j] = self.mtx_ele(config[i], config[j])
        ene, ci_coeff = sp.linalg.eigh(mtx)
        print('Energy:', ene[0]+self.enuc)
        return ene+self.enuc

    def slt_cd_1(self, config):
        o1 = 0
        o2 = 0
        for i in config:
            o1 += (self.mo1e[i//2, i//2])
        for i in config:
            for j in config:
                if i % 2 == j % 2:
                    o2 += self.j[i//2, j//2, i//2, j//2] - \
                        self.k[i//2, j//2, i//2, j//2]
                else:
                    o2 += self.j[i//2, j//2, i//2, j//2]
        return o1+o2/2

    def slt_cd_2(self, config, diff):
        o1 = 0
        o2 = 0
        if diff[0] % 2 == diff[1] % 2:  # same spin
            o1 = self.mo1e[diff[0]//2, diff[1]//2]
            for i in config:
                if diff[0] % 2 == i % 2:
                    o2 += self.j[diff[0]//2, i//2, diff[1]//2, i//2] - \
                        self.k[diff[0]//2, i//2, diff[1]//2, i//2]
                else:
                    o2 += self.j[diff[0]//2, i//2, diff[1]//2, i//2]
        return o1+o2

    def slt_cd_3(self, config1, config2, diff):
        mo_1 = [i//2 for i in diff if i in config1]
        mo_2 = [i//2 for i in diff if i in config2]
        samespin = [i for i in diff if i % 2 == 0]
        if len(samespin) == 4 or len(samespin) == 0:
            o2 = self.j[mo_1[0], mo_1[1], mo_2[0], mo_2[1]] - \
                self.k[mo_1[0], mo_1[1], mo_2[0], mo_2[1]]
        elif len(samespin) == 2:
            mo1 = [i for i in diff if i in config1]
            mo2 = [i for i in diff if i in config2]
            if set(samespin) == {mo1[0], mo2[0]} or set(samespin) == {mo1[1], mo2[1]}:
                o2 = self.j[mo_1[0], mo_1[1], mo_2[0], mo_2[1]]
            elif set(samespin) == {mo1[0], mo2[1]} or set(samespin) == {mo1[1], mo2[0]}:
                o2 = -self.k[mo_1[0], mo_1[1], mo_2[0], mo_2[1]]
            else:
                o2 = 0
        else:
            o2 = 0
        return o2

    def det_sign(self, config1, config2, diff):
        pos1 = [config1.index(i) for i in diff if i in config1]
        pos2 = [config2.index(i) for i in diff if i in config2]
        pos = np.array(pos1+pos2)
        if (np.sum(pos)) % 2 == 0:
            c = 1
        else:
            c = -1
        return c
