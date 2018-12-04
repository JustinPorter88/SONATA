#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:49:51 2018

@author: gu32kij
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines

from matplotlib2tikz import save as tikz_save
import SONATA.Pymore.utl.read as read

def plot_histogram_2Ddata(data, ref = None, upper_tri = True, title='No Title'):
    '''
    data: array.shape = (sample, i, j)
    '''

    shape = data.shape[1:]
    fig, ax = plt.subplots(shape[0],shape[1], sharex=True)
    fig.suptitle(title, fontsize=14)
    
    #upper triangle 
    if upper_tri:
        ut = np.zeros((shape[0],shape[1]))            
        ut[np.triu_indices(shape[0])] = 1      
    else:
        ut = np.ones((shape[0],shape[1]))            
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            
            if ut[i,j] == 1 and ref[i,j] != 0:     
          
                if isinstance(ref, (np.ndarray)): 
                    if ref[i,j] != 0:
                        arr = ( data[:,i,j] - ref[i,j] ) / ref[i,j] * 100
                    
                    else:
                        mu = arr.mean()
                        arr = ( data[:,i,j] - mu ) / mu * 100

                    count, bins, ignored = ax[i][j].hist(arr, 30, density=True, alpha=0.5, label='histogram')
                    #print(count,bins,ignored)
                    sigma = arr.std()
                    mu = arr.mean()
                    if sigma > 0:
                        ax[i][j].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                                       np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                                 linewidth=2, color='r', linestyle='--', alpha=0.5, label='ML approx.')
                                   
                else:
                    arr = data[:,i,j]   
                    count, bins, ignored = ax[i][j].hist(arr, 30, density=True, alpha=0.5, label='histogram')

                
                string = '$\sigma$ = %.2f %%' % sigma
                ax[i][j].text(mu,0,string)
                #ax[i][j].set_xlim(-30,30)
                
            elif ut[i,j] == 0:
                fig.delaxes(ax[i][j])
                
            if i==j:
                ax[i][j].set_xlabel('% deviation from baseline value')
                if j==shape[1]-1:
                    ax[i][j].legend()
            
            if j==0:
                ax[i][j].set_ylabel('p(x)')
        
    return None


def plot_beam_properties(data, sigma, ref, x_offset = 0.8178698):
    '''
    Args: 
            Data: np.ndarray Massterms(6), Stiffness(21), damping(1), and coordinate(1) 
            stacked horizontally for the use in DYMORE/PYMORE/MARC  
            sigma: standart deviation of Data
            ref: np.ndarray of reference to 
            select = 'all', 'Massterms', 'Stiffness'
            x_offset = 0.8178698
    kwargs: 
            usetex
            
    '''
      
    #Inertia Properties:
    #ylabel_dct = {'m00': 'kg/m', 'mEta2':'None', 'mEta3':'None', 'm33':'None', 'm23':'None', 'm22':'None'}
    x = data[:,-1] + x_offset
    ref_x = ref[:,-1]  + x_offset
    fig1, ax1 = plt.subplots(2, 3, sharex=True)
    fig1.subplots_adjust(wspace=0.3, hspace=0.3)
    fig1.suptitle('Inertia Properties', fontsize=14)
    
    c = 0
    for i in range(2):
        for j in range(3):
            ax1[i][j].plot(ref_x,ref[:,c],'-.b')
            ax1[i][j].fill_between(x, data[:,c]-sigma[:,c], data[:,c]+sigma[:,c], alpha=0.25, edgecolor='r',  linestyle=':', facecolor='r', antialiased=True,)
            ax1[i][j].plot(x,data[:,c],'--k.')
            ax1[i][j].ticklabel_format(axis='y',style='sci')
            
            c += 1
    
    #6x6 Stiffness Properties:
    #stiffness_terms = {'k11':'N', 'k12':, k22, k13, k23, k33,... k16, k26, ...k66}
    ut = np.zeros((6,6))            
    ut[np.triu_indices(6)] = 1          
    
    fig2, ax2 = plt.subplots(6, 6)
    fig2.subplots_adjust(wspace=0.5, hspace=0.5)
    fig2.suptitle('6x6 Stiffness Properties', fontsize=14)
    
    c = 6
    for j in range(6):
        for i in range(6):
            if ut[i,j] == 1:
                ax2[i][j].plot(ref_x,ref[:,c],'-.b')
                ax2[i][j].fill_between(x, data[:,c]-sigma[:,c], data[:,c]+sigma[:,c], alpha=0.25, edgecolor='r',  linestyle=':', facecolor='r', antialiased=True,)
                ax2[i][j].plot(x,data[:,c],'--k.')
                ylabel = 'k%s%s' % (i+1,j+1)
                ax2[i][j].set_ylabel(ylabel)
                #ax2[i][j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                
                if i==j:
                    ax2[i][j].set_xlabel('radial station [m]')
                
                c += 1
            else:
                fig2.delaxes(ax2[i][j])
                
    plt.show()



def plot_fandiagram(res,  Omega, RPM_vec, sigma = None, ref_fname=None, ref_str = ['x1','x1','x1','x1','x1','x1','x1']):
    '''Plotting routine to generate a Fan-Diagram (rotor eigenfrequencies over
    rotor rotational speed). Re
    
    Args:
        res: np.array for 
        Omega: reference Rotational Speed to normalize frequencies
        RPM_Vec: np.ndarray, of 
        sigma: np.ndarray of the standart deviation corresponding to the fanplot.
        ref_fname: filename of the reference data.
        ref_str: list of strings to indentify the eigenmodes 
                (f1= first flap, l2= second lag, t1=first torsion)
                
    Returns:
        None
    
    '''
    res = np.real(res)
    
    #plt.close('all')
    plt.figure()
    plt.grid(True)
    legend_lines = []
    
    #plot rotor-harmonics 
    x =  np.linspace(0, 1.2, 20)
    y =  x*Omega/(2*np.pi)
    for i in range(1,9):
        #color = '#333333'
        plt.plot(x,i*y,'--',color='grey')
        string = r'$%i\Omega$' % (i)
        plt.text(x[-1]-.06, i*y[-1]+.5, string, color='grey')
    
    
    #read and plot reference data:
    if ref_fname != None:
        #fname = 'jobs/VariSpeed/uh60a_data_blade/fanplot_uh60a_bowen-davies-PhD.txt'
        ref_data = np.loadtxt(ref_fname,skiprows=1,delimiter=',')
        ref_str2 = open(ref_fname).readline().replace('\n','').split(',')
        x = ref_data[:,0]
        for i,d in enumerate(ref_data.T):
            s=ref_str2[i]
            if 'f' in s:
                colorhex = 'blue'
                plt.plot(x, d, ':',color=colorhex)
            elif 'l' in s:
                colorhex = 'red'
                plt.plot(x, d,':', color=colorhex)
            elif 't' in s:
                colorhex = 'green'
                plt.plot(x,d,':',color=colorhex)
        legend_lines.append(mlines.Line2D([], [], color='black', linestyle=':', label='Reference')) 
    
        
    #plot standart deviation and fill between -sigma and +sigma
    x = RPM_vec/Omega
    if isinstance(sigma, (np.ndarray)):
        for i,d in enumerate(res[:,:len(ref_str)].T):
            #print(d.shape, sigma[:,i].shape)
            plt.fill_between(x, d-sigma[:,i], d+sigma[:,i], alpha=0.25, edgecolor='r',  linestyle=':', facecolor='r', antialiased=True,)
        legend_lines.append(mlines.Line2D([], [], color='red', linestyle=':', label='Standard Deviation'))    
         
        
    #plot dymore frequencies:
    x = RPM_vec/Omega
    #ref_str = ['l1','f1','f2','f3','l2','t1','f4']
    D = {'1':'s','2':'^','3':'o','4':'d'}
    ms = 3
    for i,d in enumerate(res[:,:len(ref_str)].T):
        s=ref_str[i]
        #plt.plot(x,d,'b')
        m = D[s[-1]] 
        if 'f' in s:
            colorhex = 'blue'
            plt.plot(x, d, '-', color=colorhex, marker=m, markersize = ms)
            string = r'%s flap' % (s[-1])
            plt.text(x[-1]+.01, d[-1], string, color=colorhex)
        elif 'l' in s:
            colorhex = 'red'
            plt.plot(x, d, 'o-', color=colorhex, marker=m, markersize = ms)
            string = r'%s lead-lag' % (s[-1])
            plt.text(x[-1]+.01, d[-1], string, color=colorhex)
        elif 't' in s:
            colorhex = 'green'
            plt.plot(x, d, 'o-', color=colorhex, marker=m, markersize = ms)
            string = r'%s torsion' % (s[-1])
            plt.text(x[-1]+.01, d[-1], string, color=colorhex)
            
        else:
            colorhex = 'black'
            plt.plot(x, d, 'o-', color=colorhex, marker=m, markersize = ms)
            
    legend_lines.append(mlines.Line2D([], [], color='black', linestyle='-', marker='o', label='Eigenfrequencies'))


    plt.ylim((0,45))
    plt.xlim((0,1.2))
    plt.title('Fan Diagram')
    plt.xlabel(r'Rotor Rotational Speed, $\Omega / \Omega_{ref}$')
    plt.ylabel(r'Eigenfrequencies, $\omega$ [Hz]')
    plt.legend(handles=legend_lines)
    plt.show()

    return None


def plot_eigenmodes(eigv, string='BLADE_BP_CG01', **kwargs):
    
    '''
    TBD:
    For the general Illustration of the eigenmodes
    - MARC / DYMORE Interface:
     getter function for pos and IDs of the body to extract coordinates and 
     entry location in eigv matrix
        
    '''
        #==========================EIGEN-MODES==========================================
    blade_len = 7.36082856
    r_attachment = 0.81786984
    r_hinge = 0.378
    station = 8
    blade_with_att = blade_len + r_attachment - r_hinge
    R = blade_len + r_attachment
    
    
    plt.figure()
    plt.subplot(311)
    i = 1
    
    #            plt.plot(eigVec[0,70*i:70*(i+1)], 'k--')
    pos = (np.hstack((0.0,np.linspace(3.39,42.39,40.0)))*blade_with_att/42.39 + 0.378)
    IDs = np.hstack((70*(i+1), np.linspace(30+70*i,70*(i+1)-1,40,dtype=int)))
    
    print(pos, IDs)

    for x in range(eigv[:,IDs,:].shape[0]):
        for z in range(eigv[:,IDs,:].shape[2]):
            eigv[x,0,z] = np.interp(r_attachment, pos, eigv[x,IDs,z])
    
    pos = (np.hstack((0.0,np.linspace(2.39,42.39,41.0)))*blade_with_att/42.39 + 0.378)/R
    IDs = np.hstack((70*(i+1), 0, np.linspace(30+70*i,70*(i+1)-1,40,dtype=int)))
    
    
    print(pos)
    
    scale_vals = np.amax(eigv[:,IDs,:], axis = 1)
    scale_vals_min = np.amin(eigv[:,IDs,:], axis = 1)
    
    for index, x in np.ndenumerate(scale_vals):
        if np.absolute(scale_vals_min[index]) > x:
            scale_vals[index] = scale_vals_min[index]
    
    #scale_vals = np.ones(scale_vals.shape)
    
    plt.plot(pos,eigv[0,IDs,station]/scale_vals[0,station], color='red', marker='s', markersize=1.5, label='1. lead-lag')
    plt.plot(pos,eigv[1,IDs,station]/scale_vals[1,station], 'k-.^', label='2.mode: lead-lag')
    plt.plot(pos,eigv[2,IDs,station]/scale_vals[2,station], 'k:',   label='3.mode: lead-lag')
    plt.plot(pos,eigv[3,IDs,station]/scale_vals[3,station],  color='blue', marker='o', markersize=1.5, label='3. flap')
    plt.plot(pos,eigv[4,IDs,station]/scale_vals[4,station],  color='red', marker='^', markersize=1.5, label='2. lead-lag')
    plt.plot(pos,eigv[5,IDs,station]/scale_vals[5,station], 'k--+', label='6.mode: lead-lag')
    plt.plot(pos,eigv[6,IDs,station]/scale_vals[6,station], 'k-',   label='7.mode: lead-lag')
    
    plt.grid()
    #plt.legend()
    plt.ylim([-1.05,1.05])
    plt.ylabel('Normalized Lead-Lag')
    
    plt.subplot(312)
    i = 2
    #            plt.plot(eigVec[0,70*i:70*(i+1)], 'k--')
    #            plt.plot(eigVec[0,30+70*i:70*(i+1)], 'r--', label='1.mode: flap')
    pos = (np.hstack((0.0,np.linspace(3.39,42.39,40.0)))*blade_with_att/42.39 + 0.378)
    IDs = np.hstack((70*(i+1), np.linspace(30+70*i,70*(i+1)-1,40,dtype=int)))
    for x in range(eigv[:,IDs,:].shape[0]):
        for z in range(eigv[:,IDs,:].shape[2]):
            eigv[x,0,z] = np.interp(r_attachment, pos, eigv[x,IDs,z])
    
    pos = (np.hstack((0.0,np.linspace(2.39,42.39,41.0)))*blade_with_att/42.39 + 0.378)/R
    IDs = np.hstack((70*(i+1), 0, np.linspace(30+70*i,70*(i+1)-1,40,dtype=int)))
    
    scale_vals = np.amax(eigv[:,IDs,:], axis = 1)
    scale_vals_min = np.amin(eigv[:,IDs,:], axis = 1)
    
    for index, x in np.ndenumerate(scale_vals):
        if np.absolute(scale_vals_min[index]) > x:
            scale_vals[index] = scale_vals_min[index]
    
    #scale_vals = np.ones(scale_vals.shape)
    
    #plt.plot(pos,eigv[0,IDs,station]/scale_vals[0,station], 'k-.',  label='1.mode: flap')       
    plt.plot(pos,eigv[1,IDs,station]/scale_vals[1,station], color='blue', marker='s', markersize=1.5, label='1. flap')
    plt.plot(pos,eigv[2,IDs,station]/scale_vals[2,station], color='blue', marker='^', markersize=1.5, label='2. flap')
    plt.plot(pos,eigv[3,IDs,station]/scale_vals[3,station], color='blue', marker='o', markersize=1.5, label='3. flap')
    plt.plot(pos,eigv[4,IDs,station]/scale_vals[4,station], color='red', marker='^', markersize=1.5, label='2. lead-lag')
    #plt.plot(pos,eigv[5,IDs,station]/scale_vals[4,station], 'k--+', label='6.mode: flap')
    plt.plot(pos,eigv[6,IDs,station]/scale_vals[6,station], color='blue', marker='d', markersize=1.5, label='4. flap')
    plt.grid()
    #plt.legend(loc='lower center', ncol=5)
    plt.ylim([-1.05,1.05])
    plt.ylabel('Normalized Flap')
    
    plt.subplot(313)
    i = 3
    pos = (np.hstack((0.0,np.linspace(3.39,42.39,40.0)))*blade_with_att/42.39 + 0.378)
    IDs = np.hstack((70*(i+1), np.linspace(30+70*i,70*(i+1)-1,40,dtype=int)))
    for x in range(eigv[:,IDs,:].shape[0]):
        for z in range(eigv[:,IDs,:].shape[2]):
            eigv[x,0,z] = 0.0
    
    pos = (np.hstack((0.0,np.linspace(2.39,42.39,41.0)))*blade_with_att/42.39 + 0.378)/R
    IDs = np.hstack((70*(i+1), 0, np.linspace(30+70*i,70*(i+1)-1,40,dtype=int)))
    
    
    scale_vals = np.amax(eigv[:,IDs,:], axis = 1)
    scale_vals_min = np.amin(eigv[:,IDs,:], axis = 1)
    
    for index, x in np.ndenumerate(scale_vals):
        if np.absolute(scale_vals_min[index]) > x:
            scale_vals[index] = scale_vals_min[index]
    
    #scale_vals = np.ones(scale_vals.shape)
    
    #plt.plot(pos,eigv[0,IDs,station]/scale_vals[0,station], 'k-.',  label='1.mode: torsion')
    #plt.plot(pos,eigv[1,IDs,station]/scale_vals[1,station], 'k-.^', label='2.mode: torsion')
    #plt.plot(pos,eigv[2,IDs,station]/scale_vals[2,station], 'k:',   label='3.mode: torsion')
    #plt.plot(pos,eigv[3,IDs,station]/scale_vals[3,station], 'k--',  label='4.mode: torsion')
    #plt.plot(pos,eigv[4,IDs,station]/scale_vals[4,station], 'k--*', label='5.mode: torsion')
    plt.plot(pos,eigv[5,IDs,station]/scale_vals[5,station], color='green', marker='s', markersize=1.5)
    #plt.plot(pos,eigv[6,IDs,station]/scale_vals[6,station], 'k-',   label='7.mode: torsion')
    
    line1 = mlines.Line2D([], [], color='red', linestyle='-', marker='s', label='1 lead-lag')
    line2 = mlines.Line2D([], [], color='blue', linestyle='-',marker='s', label='1 flap')
    line3 = mlines.Line2D([], [], color='blue', linestyle='-',marker='^', label='2 flap')
    line4 = mlines.Line2D([], [], color='blue', linestyle='-',marker='o', label='3 flap')
    line5 = mlines.Line2D([], [], color='red', linestyle='-',marker='^',  label='2 lead-lag')
    line6 = mlines.Line2D([], [], color='green',linestyle='-',marker='s', label='1 torsion')
    line7 = mlines.Line2D([], [], color='blue', linestyle='-', marker='d', label='4 flap')
    
    plt.grid()
    #plt.legend()
    plt.subplots_adjust(bottom=0.3)
    plt.legend(handles=[line1,line2,line3,line4,line5,line6,line7],loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.7),)
    plt.ylim([-1.05,1.05])
    plt.xlabel('Radial Station, r/R')
    plt.ylabel('Normalized Torsion')
    #plt.subplots_adjust(wspace=0.3, left=0.1, right=0.9)
   


def sim_plot(fq_dym, RPM_vec, target_dir):

    #plt.figure(figsize=(12,8))
    plt.figure(figsize=(12,8))
    array_per = RPM_vec
    plt.plot(array_per, fq_dym[:,1],'k*')
    plt.plot(array_per, fq_dym[:,2],'k*')
    plt.plot(array_per, fq_dym[:,3],'k*')
    plt.plot(array_per, fq_dym[:,6],'k*')
    plt.plot(array_per, fq_dym[:,7],'k*')
    plt.plot(array_per, fq_dym[:,10],'k*')
    plt.plot(array_per, fq_dym[:,0],'r*')
    plt.plot(array_per, fq_dym[:,4],'r*')
    plt.plot(array_per, fq_dym[:,5],'g*')
    plt.plot(array_per, fq_dym[:,9],'k*')

    plt.xlabel(r'RPM / RPM_{ref}')
    plt.ylabel(r'Freq / RPM_{ref}')
    plt.legend()
    #tikz_save(target_dir+'F2.tikz',figurewidth='\\figurewidth', figureheight='\\figureheight')
    plt.savefig(target_dir+'F2.png')
    #plt.plot(np.array([0, 100]),np.array([0, 100]),'k-')