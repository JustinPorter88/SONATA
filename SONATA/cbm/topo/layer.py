import matplotlib.pyplot as plt

from OCC.gp import gp_Pnt2d

from SONATA.cbm.fileIO.CADinput import order_BSplineLst_Head2Tail

from SONATA.cbm.topo.BSplineLst_utils import get_BSplineLst_length, get_BSplineLst_Pnt2d, \
                                         trim_BSplineLst, copy_BSplineLst, \
                                         BSplineLst_from_dct, discretize_BSplineLst, \
                                         find_BSplineLst_pos, get_BSpline_length
from SONATA.cbm.topo.wire_utils import build_wire_from_BSplineLst
from SONATA.cbm.topo.cutoff import cutoff_layer
from SONATA.cbm.topo.offset import shp_parallel_offset
from SONATA.cbm.topo.para_Geom2d_BsplineCurve import ParaLst_from_BSplineLst, BSplineLst_from_ParaLst
from SONATA.cbm.topo.layer_utils import get_layer, get_web, get_segment       

from SONATA.cbm.mesh.mesh_byprojection import mesh_by_projecting_nodes_on_BSplineLst
from SONATA.cbm.mesh.mesh_utils import grab_nodes_of_cells_on_BSplineLst, equidistant_nodes_on_BSplineLst, sort_and_reassignID, find_cells_that_contain_node, \
                                 grab_nodes_on_BSplineLst, remove_duplicates_from_list_preserving_order, merge_nodes_if_too_close, \
                                 find_node_by_ID
                                 
from SONATA.cbm.mesh.mesh_improvements import modify_sharp_corners, modify_cornerstyle_one, second_stage_improvements, integrate_leftover_interior_nodes
class Layer(object):
    ''' 
    The layer object is constructed from multiple BSplineCurveSegments. It is the basis for all future operations. 
    The object can be constructed from either a discrete formulation of point tables or from an existing TopoDS_Wire.
    ''' 
   
    def __init__(self, ID, Boundary_BSplineLst, globalStart, globalEnd, thickness, Orientation = 0, MatID = 1, **kwargs):
        self.ID = ID                                     #First single Digit: SegmentNb, Last 3 Digits: LayerNb; e.g.: 1029, Segment1, Layer29
        self.Boundary_BSplineLst = Boundary_BSplineLst    #List of Geom2d_BSplineCurve, Geom_BSplineCurve 
        self.S1 = globalStart	                           #Starting Point in S coordinates
        self.S2 = globalEnd		                           #End Point in S coordinates
        self.thickness = thickness   	                    #in mm
        self.Orientation = Orientation                    #in deg
        self.MatID = MatID                                #type: int
        self.cells = []                                   #type: List
        self.ivLst = []                                   #type: List
        self.cumA_ivLst = []                               #type: List
        self.cumB_ivLst = []                               #type: List
        self.inverse_ivLst = []                           #type: List
        self.a_nodes = []                                 #type: List
        self.b_nodes = []                                 #type: List

        #KWARGS:
        if kwargs.get('name') == None:
             self.name = 'DEFAULT'                             
        else:
            self.name = kwargs.get('name')
        
        if (kwargs.get('cutoff_style') == None) or (type(kwargs.get('cutoff_style')) is not int):   #cutoff_style (step, linear, smooth_bezier)
             self.cutoff_style = 2                             
        else:
            self.cutoff_style = kwargs.get('cutoff_style')      
        
        
        if (kwargs.get('join_style') == None) or (type(kwargs.get('join_style')) is not int):      #offset algorithm join_style = 1#( 1:round,2:mitre,3:bevels)
             self.join_style = 1                             
        else:
            self.join_style = kwargs.get('join_style')          
        
    @property
    def StartPoint(self): #gp_Pnt2d
        return self.BSplineLst[0].StartPoint()
    
    
    @property
    def EndPoint(self): #gp_Pnt2d
        return self.BSplineLst[-1].EndPoint()
    
    
    @property
    def a_BSplineLst(self): #gp_Pnt2d
        return self.BSplineLst
    
    
    @property
    def b_BSplineLst(self): #gp_Pnt2d
        return self.Boundary_BSplineLst


    @property
    def IsClosed(self):
        return self.a_BSplineLst[0].StartPoint().IsEqual(self.a_BSplineLst[-1].EndPoint(),1e-5)   
        
    
    def __str__(self): 
        #we can tell Python how to prepresent an object of our class (when using a print statement) for general purposes use  __repr__(self): 
        return  str('LayerID: \tStart[-]: \tEnd[-]: \tthickness[-]: \tOrientation[deg]: \tMatID \tName:\t\n' \
                    '%s, \t%s, \t%s, \t%s, \t\t%s, \t\t%s, \t%s, ' % (self.ID, self.S1, self.S2, self.thickness, self.Orientation, self.MatID, self.name))
     
        
    def __getstate__(self):
        """Return state values to be pickled."""
        self.Para_BSplineLst = ParaLst_from_BSplineLst(self.BSplineLst)
        self.Para_Boundary_BSplineLst = ParaLst_from_BSplineLst(self.Boundary_BSplineLst)
        return (self.ID, self.S1, self.S2, self.thickness, self.Orientation, self.MatID, self.Para_Boundary_BSplineLst, self.Para_BSplineLst)   
    
    
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.ID, self.S1, self.S2, self.thickness, self.Orientation, self.MatID, self.Para_Boundary_BSplineLst, self.Para_BSplineLst = state
        self.Boundary_BSplineLst = BSplineLst_from_ParaLst(self.Para_Boundary_BSplineLst)
        self.BSplineLst = BSplineLst_from_ParaLst(self.Para_BSplineLst)
        self.build_wire()


    def copy(self):
        BSplineLstCopy =  copy_BSplineLst(self.BSplineLst)
        namecopy = self.name + '_Copy'
        LayerCopy = Layer(self.ID,BSplineLstCopy,self.globalStart,self.globalEnd,self.thickness, self.Orientation, self.MatID, namecopy)
        return LayerCopy
  
    
    def get_length(self): #Determine and return Legth of Layer self
         self.length = get_BSplineLst_length(self.BSplineLst)
         return self.length
         
     
    #def get_pnt2d(self,S,start,end): #Return, gp_Pnt2d of argument S of layer self  
    #    return get_BSplineLst_Pnt2d(self.BSplineLst,S,start,end)
    
    def get_Pnt2d(self,S,LayerLst,WebLst):
        #print self.ID
        #print 'cum_ivLst:',self.cumA_ivLst
        for iv in self.cumA_ivLst:
            if iv[0]<=S<iv[1]:
                #print 'Coordinate',S,'is on layer',int(iv[2])
                break
        
        print(iv[2])
        tmp_layer = next((x for x in LayerLst if x.ID == int(iv[2])), None)
        if not tmp_layer == None: 
            Pnt2d = get_BSplineLst_Pnt2d(tmp_layer.a_BSplineLst,S,tmp_layer.S1,tmp_layer.S2)
        else: #Web
            WebID = -int(iv[2])-1
            Pnt2d = get_BSplineLst_Pnt2d(WebLst[WebID].BSplineLst, S, start = WebLst[WebID].Pos2, end = WebLst[WebID].Pos1)

        return Pnt2d      
                
                
    def build_wire(self): #Builds TopoDS_Wire from connecting BSplineSegments and returns it  
        self.wire = build_wire_from_BSplineLst(self.BSplineLst)   
        
        
    def trim(self,S1,S2,start, end): #Trims layer between S1 and S2
        return trim_BSplineLst(self.BSplineLst, S1, S2,  start, end)
        
    
    def trim_to_coords(self, start, end):
        self.BSplineLst = trim_BSplineLst(self.BSplineLst, self.globalStart, self.globalEnd,  start, end)
        return self.BSplineLst
            
    
    def build_layer(self):
        deflection = 1e-3
        npArray = discretize_BSplineLst(self.Boundary_BSplineLst, deflection) 
        offlinepts = shp_parallel_offset(npArray,self.thickness,self.join_style)
        OffsetBSplineLst = BSplineLst_from_dct(offlinepts)
        OffsetBSplineLst = cutoff_layer(self.Boundary_BSplineLst,OffsetBSplineLst,self.S1,self.S2,self.cutoff_style)
        self.BSplineLst = OffsetBSplineLst
         
        
    def determine_a_nodes(self,SegmentLst,global_minLen,display=None):
        ''' '''
        unmeshed_ids = []
        for seg in SegmentLst:
            if seg.LayerLst:
                unmeshed_ids.append(int(seg.LayerLst[-1].ID+1))
            
        new_a_nodes=[]
        #print self.inverse_ivLst
        for iv_counter,iv in enumerate(self.inverse_ivLst):
            if int(iv[2]) in unmeshed_ids: #if 
                #print iv, "equidistand nodes on BsplineLst of LayerLst entry"
                eq_nodes = []
                BSplineLst = self.a_BSplineLst             
                iv_BSplineLst = trim_BSplineLst(BSplineLst,iv[0],iv[1],self.S1,self.S2  )
                if iv_counter==0 and len(self.inverse_ivLst)>1: #first but not last
                    IncStart=True
                    IncEnd=False
                
                elif iv_counter==0 and len(self.inverse_ivLst)==1: #first and last
                    IncStart=True
                    IncEnd=True

                elif iv_counter==len(self.inverse_ivLst)-1 and len(self.inverse_ivLst)>1: #last but not first
                    #print iv, 'Last but not first'
                    if  iv_counter==len(self.inverse_ivLst)-1 and iv[1]==1 and self.inverse_ivLst[0][0]==0:
                        IncStart=False
                        IncEnd=False
                    else:
                        IncStart=False
                        IncEnd=True
                                        
                else:
                    IncStart=False
                    IncEnd=False
                
                eq_nodes = equidistant_nodes_on_BSplineLst(iv_BSplineLst, True, IncStart, IncEnd, minLen = global_minLen, LayerID = self.ID)
                new_a_nodes.extend(eq_nodes)
                
                
            else:
                #only use once for each layer!
                #print iv, "use nodes b_nodes of layer", int(iv[2])
                tmp_layer = get_layer(int(iv[2]),SegmentLst)
                #iv_BSplineLst = trim_BSplineLst(tmp_layer.b_BSplineLst,iv[0],iv[1],tmp_layer.S1,tmp_layer.S2)
                iv_BSplineLst = trim_BSplineLst(self.a_BSplineLst,iv[0],iv[1],self.S1,self.S2)
                #iv_BSplineLst = self.a_BSplineLst
                isClosed = iv_BSplineLst[0].StartPoint().IsEqual(iv_BSplineLst[-1].EndPoint(),1e-5)
               
                if isClosed:
                    tmp_nodes = tmp_layer.b_nodes
                else: 
                    tmp_nodes = [tmp_layer.a_nodes[0]] + tmp_layer.b_nodes + [tmp_layer.a_nodes[-1]]
                
                disco_nodes = grab_nodes_on_BSplineLst(tmp_nodes,iv_BSplineLst)
                new_a_nodes.extend(disco_nodes)
        
        
        self.a_nodes = remove_duplicates_from_list_preserving_order(new_a_nodes)
        self.a_nodes = merge_nodes_if_too_close(self.a_nodes,self.a_BSplineLst,global_minLen,0.01)
        
        
    def mesh_layer(self, SegmentLst, global_minLen, proj_tol_1= 5e-2, 
                   proj_tol_2= 3e-1, crit_angle_1 = 115, alpha_crit_2 = 60, 
                   growing_factor=1.8, shrinking_factor=0.01, display=None):
        '''
        The mesh layer function discretizes the layer, which is composed of a 
        a_BsplineLst and a b_BsplineLst. Between the a_BsplineLst and the 
        b_BsplineLst the cells are created. First nodes on the a_BSplineLst are
        determined with the determine_a_nodes procedure. Subsequently the 
        a_nodes are projected onto the b_BsplineLst. The following functions 
        (modify_cornerstyle_one, modify_sharp_corners and
        second_stage_improvements) try to imporve the quality of the mesh. 
        everything is stored in the layer.cells and is returned
        
        Args:
            SegmentLst:               The overall list of Segments within the 
                                    segemet. This list is needed to determine
                                    the a_nodes
            global_minLen:          
            proj_tol_1 = 5e-2:      tolerance value to determine a distance, 
                                    in which the resulting projection point 
                                    has to be. 
                                    (mesh_by_projecting_nodes_on_BSplineLst)
            proj_tol_2 = 2e-1:      tolerance value to determine a distance, 
                                    in which the resulting projection point 
                                    has to be. (modify_sharp_corners)
            crit_angle_1 = 115:     is the critical angle to determine a corner 
                                    if 2 projection points are found.    
            alpha_crit_2 = 60:      is the critical angle to refine  a corner 
            growing_factor = 1.8:   critical growing factor of cell before 
                                    splitting 
            shrinking_factor = 0.10:  critical shrinking factor for cells 
                                    before merging nodes
            
        returns: self.cells: (list of cells) 
        '''
        
        self.determine_a_nodes(SegmentLst,global_minLen,display)
        self.a_nodes, self.b_nodes, self.cells = mesh_by_projecting_nodes_on_BSplineLst(self.a_BSplineLst,self.a_nodes,self.b_BSplineLst,self.thickness, proj_tol_1,crit_angle_1, LayerID = self.ID, display=display) 
        #enhanced_cells = modify_cornerstyle_one(cells,self.b_BSplineLst)
        self.cells, nb_nodes = modify_sharp_corners(self.cells, self.b_BSplineLst, global_minLen, self.thickness, self.ID, proj_tol_2, alpha_crit_2, display=display)
        self.b_nodes.extend(nb_nodes)
        try:
            self.cells, nb_nodes = second_stage_improvements(self.cells, self.b_BSplineLst, global_minLen, self.ID, growing_factor, shrinking_factor, display=display)
            self.b_nodes.extend(nb_nodes)
        except:
            pass
                
        #self.b_nodes = sorted(self.b_nodes, key=lambda Node: (Node.parameters[1],Node.parameters[2]))  
        
        for c in self.cells:
            c.calc_theta_1()
            c.theta_3 = self.Orientation
            c.MatID = int(self.MatID)
            c.structured = True
            
        return self.cells
    


#execute the following code if this file is executed as __main__   
if __name__ == '__main__':   
     L1 = Layer()
     pass