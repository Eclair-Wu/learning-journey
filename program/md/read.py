from numba import jit
import json
import numpy as np
import pandas as pd



def atom_trans(x,y,array):
    for i in range(x,y):
        for k in range(1,4):
            array[i][k]=array[i][k]*1.88971616
    
    
def read_pos(inputfile,natom,begin=1):
    with open(inputfile,'r') as input:
        a=input.readlines()
        b=a.index('%ATOMIC_POSTION\n')
    pos=a[b+1:b+natom+1]
    output=np.zeros((natom,4),dtype=np.float64)
    for i in range(natom):
            pos[i]=pos[i].strip('\n')
            output[i][0]=begin
            begin=begin+1
            for j in range(1,4):
                output[i][j]=np.float64(pos[i].split('\t')[j])
    return output

def read_v(inputfile,natom,begin=1):
    with open(inputfile,'r') as input:
        a=input.readlines()
        c=a.index('%ATOMIC_VELOCITY\n')
        pos=a[c+1:c+natom+1]
        output=np.zeros((natom,4),dtype=np.float64)
    for i in range(natom):
            pos[i]=pos[i].strip('\n')
            output[i][0]=begin
            begin=begin+1
            for j in range(1,4):
                output[i][j]=np.float64(pos[i].split('\t')[j])
    return output



if __name__ == '__main__':
    with open ('parameter.json','r') as p:
        para=json.load(p)
    output=read_pos(inputfile=para['geo_dir'],natom=para['natom'])
    atom_trans(0,len(output),output)
    np.savetxt('output.in',output,delimiter='\t',fmt='%.12g')


