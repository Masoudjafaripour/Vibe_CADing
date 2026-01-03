from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism

# ---- sketch (rectangle) ----
p1, p2, p3, p4 = (
    gp_Pnt(0,0,0),
    gp_Pnt(1,0,0),
    gp_Pnt(1,1,0),
    gp_Pnt(0,1,0)
)

edges = [
    BRepBuilderAPI_MakeEdge(p1,p2).Edge(),
    BRepBuilderAPI_MakeEdge(p2,p3).Edge(),
    BRepBuilderAPI_MakeEdge(p3,p4).Edge(),
    BRepBuilderAPI_MakeEdge(p4,p1).Edge()
]

wire = BRepBuilderAPI_MakeWire()
for e in edges:
    wire.Add(e)

face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()

# ---- solid via extrusion ----
solid = BRepPrimAPI_MakePrism(face, gp_Vec(0,0,1)).Shape()


from OCC.Display.SimpleGui import init_display

display, start_display, _, _ = init_display()
display.DisplayShape(solid, update=True)
start_display()
