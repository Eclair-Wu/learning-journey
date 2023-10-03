from cmath import pi
import time
import pandas as pd
import read as rd
import numpy as np
import json
import neighbor_list as nl
from numba import njit

start=time.time()
Na=6.02214086e+23
k=1.38064852e-23
e=1.602176634e-19


def Ek(velocity,m):
    a=np.sum(np.square(velocity))
    ek=m*a*10/(Na*e*2)
    return ek

def Tem(velocity,m,natom):
    a=np.sum(np.square(velocity))
    tem=m*a*10/(Na*k*3*natom)
    return tem

def u(input,r_cut,neighborlist,a,b,natom):
    global mirror
    u_sum=0
    for id in range(natom):
        len=neighborlist[0][id]
        for i in range(len):
            index=neighborlist[1][id][i]-1
            dis=input[id]-input[index]
            dis=min_dis(dis,mirror)
            if(dis[0]**2+dis[1]**2+dis[2]**2)<=r_cut**2:
                mindis=np.sqrt(dis[0]**2+dis[1]**2+dis[2]**2)
                u1=4*b*((a/mindis)**12-(a/mindis)**6)
                u2=4*b*((a/r_cut)**12-(a/r_cut)**6)
                u_sum=u1-u2+u_sum
    u_sum=u_sum*(1/2)
    return u_sum

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

def force(input,r_cut,neighborlist,a,b,natom):
    sum_Force=np.zeros((natom,4),dtype=np.float64)
    global mirror
    for i in range(natom):
        sum_Force[i][0]=i+1
        len=neighborlist[0][i]
        Force=np.zeros((1,3),dtype=np.float64)
        for j in range(len):
            index=neighborlist[1][i][j]-1
            dis=input[i]-input[index]
            dis=min_dis(dis,mirror)
            if(dis[0]**2+dis[1]**2+dis[2]**2)<=r_cut**2:
                mindis=np.sqrt(dis[0]**2+dis[1]**2+dis[2]**2)
                Force1=4*b*((12*((a/mindis)**12))-6*((a/mindis)**6))*(dis/(mindis**2))
                Force=Force+Force1
        sum_Force[i][1]=Force[0][0]
        sum_Force[i][2]=Force[0][1]
        sum_Force[i][3]=Force[0][2]
    return sum_Force

def simulation_step(t,m):
    global pos,velocity,sum_force,neighborlist
    pos=(1/(2*m))*sum_force*(t**2)*0.1*Na*e+velocity*t+pos
    velocity=velocity+(1/(2*m))*sum_force*t*0.1*Na*e
    sum_force=force(pos,para['r_cut'],neighborlist,para['lj_sigma'],para['lj_epslion'],para['natom'])
    sum_force=sum_force[...,1:4]
    velocity=velocity+(1/(2*m))*sum_force*t*0.1*Na*e

def run_data(i):
    global velocity,neighborlist,pos
    ek=Ek(velocity,para['mol_mass'])
    eu=u(pos,para['r_cut'],neighborlist,para['lj_sigma'],para['lj_epslion'],para['natom'])
    t=Tem(velocity,para['mol_mass'],para['natom'])
    run={'Step':i,'Time':i*para['md_step'],'Kinetic energy':ek,'Potential energy':eu,'Total energy':eu+ek,'Temperature':t}
    return run

def x_v_f_data(step,a='a'):   
    global pos,velocity,sum_force
    pos_output=pd.DataFrame(pos,columns=['x','y','z'])
    pos_output.insert(0,'time',step*para['md_step'])
    pos_output.insert(0,'step',step)
    pos_output.to_csv('postion.txt',float_format='%.12g',sep='\t',mode=a)
    v_output=pd.DataFrame(velocity,columns=['x','y','z'])
    v_output.insert(0,'time',step*para['md_step'])
    v_output.insert(0,'step',step)
    v_output.to_csv('velocity.txt',float_format='%.12g',sep='\t',mode=a)
    force_output=pd.DataFrame(sum_force,columns=['x','y','z'])
    force_output.insert(0,'time',step*para['md_step'])
    force_output.insert(0,'step',step)
    force_output.to_csv('force.txt',float_format='%.12g',sep='\t',mode=a) 

def bernderson(tem0,t,tau):
    global velocity
    tem=Tem(velocity,para['mol_mass'],para['natom'])
    lamda=np.sqrt(1+(t/tau)*(tem0/tem-1))
    velocity=lamda*velocity

def inital(natom,m,t0,vread):
    if vread=true:
        v=rd.read_v(para['geo_dir'],natom)
        v=v[...,1,4]
    else:
        v=np.random.uniform(-0.5,0.5,(natom,3))
        for i in range(3):
            vc=np.full_like(v[:,i],(np.sum(v[:,i])/natom))
            v[:,i]=v[:,i]-vc
        a=np.sum(np.square(v))
        factor=t0*Na*k*3*natom/(10*a*m)
        v=np.sqrt(factor)*v
    return v

def rdf(dr,rdf_cut,natom,neighborlist,pos):
    global mirror
    ball=int(rdf_cut/dr)
    gr1=np.zeros(ball,np.float64)
    gr2=np.zeros(ball,np.float64)
    rou=natom/(para['cell_para']**3)
    for j in range(natom):
        len=neighborlist[0][j]
        for k in range(len):
            index=neighborlist[1][j][k]-1
            dis=pos[j]-pos[index]
            dis=min_dis(dis,mirror)
            mindis=np.sqrt(dis[0]**2+dis[1]**2+dis[2]**2)
            if mindis<=rdf_cut:
                a=int(mindis/dr)
                gr2[a]+=1.0
    for i in range(ball):
        gr1[i]=i*dr
        if i>0:
            b=(4*pi*((i*dr)**2))*dr*rou
            gr2[i]=gr2[i]/(natom*b)
    return gr2,gr1




if __name__ == '__main__':    
    with open('parameter.json','r') as p:
        para=json.load(p)
    mirror=[para['cell_para'],para['cell_para'],para['cell_para']]
    
    velocity=inital(para['natom'],para['mol_mass'],para['t_inital'],para['read_velocity'])
    pos=rd.read_pos(inputfile=para['geo_dir'],natom=para['natom'])
    pos=pos[...,1:4]
    neighborlist=nl.verlet_list(pos,para['r_cut'],para['neighbor_r'],para['natom'],mirror,para['max_neighbor'])
    sum_force=force(pos,para['r_cut'],neighborlist,para['lj_sigma'],para['lj_epslion'],para['natom'])
    sum_force=sum_force[...,1:4]
    
    x_v_f_data(0,'w') 
    sum_run=pd.DataFrame(columns=['Step','Time','Kinetic energy','Potential energy','Total energy','Temperature'])
    sum_run=sum_run.append(run_data(0),ignore_index=True)
   
    s_step=para['simulation_step']
    search=para['nstep_search']
    output=para['nstep_output']
    
    gr_sum=0
    gr1=rdf(para['rdf_dr'],para['rdf_rcut'],para['natom'],neighborlist,pos)[1]
    rdfinterval=para['rdf_interval']
    
    for i in range(1,s_step+1):
        print('step:',i)
        if i%search==0:
            neighborlist=nl.verlet_list(pos,para['r_cut'],para['neighbor_r'],para['natom'],mirror,para['max_neighbor'])
            if i%output==0:
                simulation_step(t=para['md_step'],m=para['mol_mass'])
                sum_run=sum_run.append(run_data(i),ignore_index=True)
                x_v_f_data(i)  
            else:
                simulation_step(t=para['md_step'],m=para['mol_mass'])

        else:
            if i%output==0:
                simulation_step(t=para['md_step'],m=para['mol_mass'])
                sum_run=sum_run.append(run_data(i),ignore_index=True)
                x_v_f_data(i)  
            else:
                simulation_step(t=para['md_step'],m=para['mol_mass'])
        bernderson(tem0=para['T0'],t=para['md_step'],tau=para['tau'])
        
        if(s_step>=para['rdf_start'] and s_step<=para['rdf_stop']):
            if(s_step-para['rdf_start'])%rdfinterval==0:
                g=rdf(para['rdf_dr'],para['rdf_rcut'],para['natom'],neighborlist,pos)[0]
                gr_sum=gr_sum+g
    gr_sum=gr_sum/((para['rdf_stop']-para['rdf_start']))
    gr=np.array([gr1,gr_sum]).T
    sum_run.to_csv('run.log',index=0,float_format='%.12g',sep='\t')
    np.savetxt('test',gr,fmt='%.6g')

end=time.time()
print(end-start)