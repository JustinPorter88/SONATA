# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:54:37 2018

@author: TPflumm
"""
import numpy as np
import sys
import copy 
import math

from datetime import datetime
from openmdao.api import ExplicitComponent

from SONATA.cbm.fileIO.hiddenprints import HiddenPrints
from SONATA.cbm.sonata_cbm import CBM

class CBM_ExplComp(ExplicitComponent):
    
    """
    A simple CBM Component that computes the the composite beam properties.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ref_config = copy.deepcopy(config) 
        self.ref_dct = {}

    
    def setup(self):
        self.counter = 0
        self.startTime = datetime.now()

        self.set_input()
        self.set_output()

        # Finite difference all partials.
        #self.declare_partials('*', '*', method='fd')
    
    def set_input(self):
        self.add_input('s_w1', val=0.44)
        self.add_input('s_w2', val=0.56)
        self.add_input('t_erosion', val=0.82)
        self.add_input('t_overwrap', val=0.25)  
        self.add_input('t_spar1', val=3.00)
        self.add_input('rho_mat3', val=0.05)  
        self.add_input('t_sparcap1', val=2.05)
        self.add_input('t_sparcap2', val=1.85)
        self.add_input('t_sparcap3', val=1.85)
        self.add_input('t_sparcap4', val=0.50) 
        self.add_input('rho_mat11', val=0.05)  
        
      
    def set_output(self):
        self.add_output('obj', desc='objective_function')   
        #self.add_output('obj2', desc='objective function')  
        self.add_output('MpUS', desc='Mass per unit span (kg/m)')   
        self.add_output('Xm2', desc='x location of Center of Gravity')
        self.add_output('Xm3', desc='y location of Center of Gravity')
        self.add_output('Xs2', desc='x location of Shear Center')
        self.add_output('Xs3', desc='y location of Shear Center')
        self.add_output('EA', desc='Axial Stiffness')
        self.add_output('GJ', desc='Torsional Stiffness')
        self.add_output('EI2', desc='Flapping Bending Stiffness')
        self.add_output('EI3', desc='Lagging Bending Stiffness')
        
        # beam properties stored in the vector
        self.add_output('BeamPropSec', val=np.zeros((2,29)), desc='Massterms(6), Stiffness(21), damping(1) and coordinate(1)')   
        
        
    def compute(self, inputs, outputs):
#        elapsed_t = datetime.now() - self.startTime
#        m, s = divmod(elapsed_t.seconds, 60)
#        h, m = divmod(m, 60)
#        if self.counter == 0:
#            print('--time',  end=' ')  
#            for k in inputs:
#                print('--%s, ' %k, end=' ')  
#            print('')
#        print(('%2i' % self.counter), end=' ')
#        print('%02d:%02d:%02d [' % (h,m,s), end=' ')
#        print(('%2.3f' % inputs['s_w1'][0]), end=' ')
#        print(('%2.3f' % inputs['s_w2'][0]), end=' ')
#        print(('%2.3f' % inputs['t_sparcap1'][0]), end=' ')
#        print(('%2.3f' % inputs['t_sparcap2'][0]), end=' ')
#        print(('%2.3f' % inputs['t_sparcap3'][0]), end=' ')
#        print(('%2.3f' % inputs['t_sparcap4'][0]), end=' ')
#        print(('%2.3f' % inputs['rho_mat11'][0]), end=' ')
#        for k,v in inputs.items():
#             print('%.2f, ' %v, end=' ') 

        #SETUP A CBM JOB:
        self.job = None
        self.job = CBM(self.config)
        self.connect_input_to_config(inputs)

        try:
            with HiddenPrints():
                self.job.cbm_gen_topo()
                self.job.cbm_gen_mesh()
                self.job.cbm_run_vabs(rm_vabfiles=True)
            self.connect_output_from_job(outputs)
#            print('] ' + str(outputs['obj']))
            
        except KeyboardInterrupt:
            raise Exception
            
        except:
           outputs['obj'] = 1e3    
           self.job.cbm_post_2dmesh()
           print('] [Unexpected error:', sys.exc_info()[0], ']')
           
        self.counter += 1   
        

    def connect_input_to_config(self,inputs):
        #Architecture:
        self.job.config.webs[1]['Pos1'] = inputs['s_w1'][0]
        self.job.config.webs[1]['Pos2'] = 1-self.job.config.webs[1]['Pos1']
        
        self.job.config.webs[2]['Pos1'] = inputs['s_w2'][0]
        self.job.config.webs[2]['Pos2'] = 1-self.job.config.webs[2]['Pos1']
        
        #Segment 0 :
        self.job.config.segments[0]['Layup'][0][2] = inputs['t_erosion'][0]
        self.job.config.segments[0]['Layup'][1][2] = inputs['t_overwrap'][0]
        self.job.config.segments[0]['Layup'][2][2] = inputs['t_overwrap'][0]
        self.job.config.segments[0]['Layup'][3][2] = inputs['t_overwrap'][0]
        self.job.config.segments[0]['Layup'][4][2] = inputs['t_overwrap'][0]
        
        #Segment 1:
        self.job.config.segments[1]['Layup'][0][2] = inputs['t_spar1'][0]
        self.job.MaterialLst[3].rho = inputs['rho_mat3'][0]
        
        #Segment 2:
        self.job.config.segments[2]['Layup'][0][2] = inputs['t_sparcap1'][0]
        self.job.config.segments[2]['Layup'][1][2] = inputs['t_sparcap2'][0]
        self.job.config.segments[2]['Layup'][2][2] = inputs['t_sparcap3'][0]
        self.job.config.segments[2]['Layup'][3][2] = inputs['t_sparcap4'][0]
        
        #self.job.config.segments[2]['Layup'][3][4] = inputs['o_spar_cap1']
        
        #Segment 3:
        self.job.MaterialLst[11].rho = inputs['rho_mat11'][0]


    def connect_output_from_job(self, outputs):
        outputs['MpUS'] = self.job.BeamProperties.MpUS
        outputs['Xm2'] = self.job.BeamProperties.Xm2
        outputs['Xm3'] = self.job.BeamProperties.Xm3
        outputs['EA']   = self.job.BeamProperties.CS[0][0]
        outputs['GJ']   = self.job.BeamProperties.CS[1][1]
        outputs['EI2']  = self.job.BeamProperties.CS[2][2]
        outputs['EI3']  = self.job.BeamProperties.CS[3][3]
        outputs['EI3']  = self.job.BeamProperties.MpUS
        beamProp = np.repeat([self.job.cbm_set_DymoreMK()], 2, axis=0)
        beamProp[0,-1] = +0.000e+00
        beamProp[1,-1] = +7.361e+00
        outputs['BeamPropSec'] = beamProp
        #outputs['obj'] = self.compute_objective()
        
    def compute_objective(self):
        o1 = abs(self.job.BeamProperties.CS[2][2]*1e-6 - self.ref_dct['bending_stiffnesses'][0]) / self.ref_dct['bending_stiffnesses'][0]
        o2 = abs(self.job.BeamProperties.CS[3][3]*1e-6 - self.ref_dct['bending_stiffnesses'][1]) / self.ref_dct['bending_stiffnesses'][1]
        o3 = abs(self.job.BeamProperties.CS[1][1]*1e-6 - self.ref_dct['torsional_stiffness']) / self.ref_dct['torsional_stiffness']
        o4 = abs(self.job.BeamProperties.CS[0][0] - self.ref_dct['axial_stiffness']) / self.ref_dct['axial_stiffness']
        o5 = abs(self.job.BeamProperties.MpUS - self.ref_dct['mass_per_unit_span']) / self.ref_dct['mass_per_unit_span']
        #o6 = abs(self.job.BeamProperties.Xm2)
        
        self.residuum = np.mean([o1,o2,o3,o4,o5])
        self.rmse = math.sqrt(np.mean([o1**2,o2**2,o3**2,o4**2,o5**2]))
        return self.rmse
    
    def set_references(self,ref_dct):
        self.ref_dct = ref_dct


    def post_cbm(self):
        return self.job.cbm_post_2dmesh()