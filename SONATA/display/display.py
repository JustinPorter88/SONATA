## 2016 Tobias Pflumm (tobias.pflumm@tum.de)
## Note that this is adapted from pythonocc-Utils.
''' This Module describes all displying functionalietes of the code'''

import sys

from OCC.Display.SimpleGui import init_display
from OCC.gp import gp_Pnt
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Graphic3d import (Graphic3d_EF_PDF,
                           Graphic3d_EF_SVG,
                           Graphic3d_EF_TEX,
                           Graphic3d_EF_PostScript,
                           Graphic3d_EF_EnhPostScript)

#===========================================================================
# MENU FUNCTIONALITIES
#===========================================================================


#===========================================================================
# DISPLAY
#===========================================================================
global display

def init_viewer():
    display, start_display, add_menu, add_function_to_menu = init_display('wx')
    display.Context.SetDeviationAngle(0.000001)       # 0.001 default. Be careful to scale it to the problem.
    display.Context.SetDeviationCoefficient(0.000001) # 0.001 default. Be careful to scale it to the problem. 

    add_menu('screencapture')
    add_function_to_menu('screencapture', export_to_BMP)
    add_function_to_menu('screencapture', export_to_PNG)
    add_function_to_menu('screencapture', export_to_JPEG)
    add_function_to_menu('screencapture', export_to_TIFF)
    add_function_to_menu('screencapture', exit)              
    return display    

def export_to_BMP(event=None):
    display.View.Dump('./capture_bmp.bmp')
    
    
def export_to_PNG(event=None):
    display.View.Dump('./capture_png.png')
    
    
def export_to_JPEG(event=None):
    display.View.Dump('./capture_jpeg.jpeg')
    
    
def export_to_TIFF(event=None):
    display.View.Dump('./capture_tiff.tiff')

def exit(event=None):
    sys.exit()

def export_to_PDF(f,event=None):
    f.Export('SONATA_export.pdf', Graphic3d_EF_PDF)
    pass

def export_to_SVG(event=None):
    f.Export('SONATA_export.svg', Graphic3d_EF_SVG)


def export_to_PS(event=None):
    f.Export('SONATA_export.ps', Graphic3d_EF_PostScript)


def export_to_EnhPS(event=None):
    f.Export('SONATA_export_enh.ps', Graphic3d_EF_EnhPostScript)


def export_to_TEX(event=None):
    f.Export('SONATA_export.tex', Graphic3d_EF_TEX)

def exit(event=None):
    sys.exit()
    
def show_coordinate_system(display):
    '''CREATE AXIS SYSTEM for Visualization'''
    O  = gp_Pnt(0., 0., 0.)
    p1 = gp_Pnt(0.088,0.,0.)
    p2 = gp_Pnt(0.,0.088,0.)
    p3 = gp_Pnt(0.,0.,0.088)
    
    h1 = BRepBuilderAPI_MakeEdge(O,p1).Shape()
    h2 = BRepBuilderAPI_MakeEdge(O,p2).Shape()
    h3 = BRepBuilderAPI_MakeEdge(O,p3).Shape()

    display.DisplayShape(O,color='BLACK')
    display.DisplayShape(h1,color='RED')
    display.DisplayShape(h2,color='GREEN')
    display.DisplayShape(h3,color='BLUE')
    return display

def display_points_of_array(array,display):
    for j in range(array.Lower(),array.Upper()+1):
        p = array.Value(j)
        display.DisplayShape(p, update=False)         
                                              
if __name__ == '__main__':   
    pass