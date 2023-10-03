
from read import atom_trans, read_pos
import numpy as np
import json
from numba import njit




def min_dis(dis,mirror):
    for i in range(3):
        for j in range(10):
            if dis[i]>mirror[i]/2:
                dis[i]=dis[i]-mirror[i]
            elif dis[i]< -mirror[i]/2:
                dis[i]=dis[i]+mirror[i]
            else:
                break
    return dis


def verlet_list(input,r_cut,neighbor_r,natom,mirror,max_neighbor):
    r_list=r_cut+neighbor_r
    nlist=np.zeros(natom,dtype=np.int64)
    list_atom=np.zeros((natom,max_neighbor),dtype=np.int64)
    for id in range(natom):
        for i in range(id+1,natom):
            dis=input[id]-input[i]
            dis=min_dis(dis,mirror)
            if abs(dis[0])>r_list:
                continue
            elif abs(dis[1])>r_list:
                continue
            elif abs(dis[2])>r_list:
                continue
            elif(dis[0]**2+dis[1]**2+dis[2]**2)<= r_list**2:
                list_atom[id][nlist[id]]=i+1
                list_atom[i][nlist[i]]=id+1
                nlist[id]=nlist[id]+1
                nlist[i]=nlist[i]+1
    return nlist,list_atom




#warnings.filterwarnings('ignore')

if __name__ == '__main__':
    with open ('parameter.json','r') as p:
        para=json.load(p)
    input=read_pos(inputfile=para['geo_dir'],natom=para['natom'])
    input=input[...,1:4]
    mirror=[para['cell_para'],para['cell_para'],para['cell_para']]
    neighborlist=verlet_list(input,para['r_cut'],para['neighbor_r'],para['natom'],mirror,para['max_neighbor'])
    np.savetxt('verlet_list.txt',neighborlist[1][11],delimiter='',fmt='%.12g')








    
