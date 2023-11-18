import numpy as np
import pyscf
from itertools import combinations

def int_mo(h1e,h2e,c_final):#transform h1e h2e in mo
    h1e_mo=c_final.T@h1e@c_final
    h2e_mo=h2e
    for i in range(4):
        h2e_mo = np.tensordot(h2e_mo, c_final, axes=1).transpose(3, 0, 1, 2)
    return h1e_mo,h2e_mo

def mtx_ele(config1,config2,h1e_mo,h2e_mo,inactive=[]):#evaluate matrix element
    diff=tuple(config1^config2)
    config1=list(config1)
    config2=list(config2)
    if len(diff)==0:
        mtx=slt_cd_1(config1+inactive,h1e_mo,h2e_mo)
    elif len(diff)==2:
        mtx=slt_cd_2(config1+inactive,diff,h1e_mo,h2e_mo)*det_sign(config1,config2,diff)
    elif len(diff)==4:
        mtx=slt_cd_3(config1,config2,diff,h2e_mo)*det_sign(config1,config2,diff)
    else:
        mtx=0
    return mtx

def fci(K,N,h1e_mo,h2e_mo,nuc):
    config=[]
    orb=np.linspace(0,2*K-1,2*K,dtype=np.int64)
    for a in combinations(orb,N):
        config.append(set(a))
    lenth=len(config)
    mtx=np.zeros((lenth,lenth),np.float64)
    for i in range(lenth):
        for j in range(lenth):
            mtx[i,j]=mtx_ele(config[i],config[j],h1e_mo,h2e_mo,)
    energy=np.linalg.eigvalsh(mtx)[:1]+nuc
    return energy

def casci(N,act_mo,N_a,N_b,h1e_mo,h2e_mo):
    orb_tot=list(range(N))
    active_a=[]
    active_b=[]
    for a in act_mo:
        active_a.append(2*a)
        active_b.append(2*a+1)
    inactive=[orb for orb in orb_tot if orb not in active_a+active_b]
    config=[]
    for a in combinations(active_a,N_a):
        for b in combinations(active_b,N_b):
            config.append(set(a+b))
    lenth=len(config)
    mtx=np.zeros((lenth,lenth),np.float64)
    for i in range(lenth):
        for j in range(lenth):
            mtx[i,j]=mtx_ele(config[i],config[j],h1e_mo,h2e_mo,inactive)
    return mtx


def slt_cd_1(config,h1e_mo,h2e_mo):
    o1=0
    o2=0
    for orb1 in config:
        orb_1=int(orb1/2)
        o1+=h1e_mo[orb_1,orb_1]
        for orb2 in config:
            orb_2=int(orb2/2)
            if orb1%2==orb2%2:
                o2+=h2e_mo[orb_1,orb_1,orb_2,orb_2]-h2e_mo[orb_1,orb_2,orb_2,orb_1]
            else:
                o2+=h2e_mo[orb_1,orb_1,orb_2,orb_2]
    o2=o2/2
    return o1+o2

def slt_cd_2(config,diff,h1e_mo,h2e_mo):
    o1=0
    o2=0
    orb_a=int(diff[0]/2)
    orb_s=int(diff[1]/2)
    if diff[0]%2==diff[1]%2:#same spin
        o1=h1e_mo[orb_s,orb_a]
        for orb in config:
            orb_=int(orb/2)
            if diff[0]%2==orb%2:
                o2+=h2e_mo[orb_s,orb_a,orb_,orb_]-h2e_mo[orb_s,orb_,orb_,orb_a]
            else:
                o2+=h2e_mo[orb_s,orb_a,orb_,orb_]
    return o1+o2

def slt_cd_3(config1,config2,diff,h2e_mo):
    mo_1=[int(orb/2) for orb in diff if orb in config1]
    mo_2=[int(orb/2) for orb in diff if orb in config2]
    samespin=[orb for orb in diff if orb%2==0]
    if len(samespin)==4 or len(samespin)==0:
        o2=h2e_mo[mo_1[0],mo_2[0],mo_1[1],mo_2[1]]-h2e_mo[mo_1[0],mo_2[1],mo_1[1],mo_2[0]]
    elif len(samespin)==2:
        mo1=[orb for orb in diff if orb in config1]
        mo2=[orb for orb in diff if orb in config2]
        if set(samespin)=={mo1[0],mo2[0]} or set(samespin)=={mo1[1],mo2[1]}:
            o2=h2e_mo[mo_1[0],mo_2[0],mo_1[1],mo_2[1]]
        elif set(samespin)=={mo1[0],mo2[1]}or set(samespin)=={mo1[1],mo2[0]}:
            o2=-h2e_mo[mo_1[0],mo_2[1],mo_1[1],mo_2[0]]
        else:
            o2=0
    else:  
        o2=0
    return o2

def det_sign(config1,config2,diff):
    pos1=[config1.index(orb) for orb in diff if orb in config1]
    pos2=[config2.index(orb) for orb in diff if orb in config2]
    pos=np.array(pos1+pos2)
    if (np.sum(pos))%2==0:
        c=1
    else:
        c=-1
    return c