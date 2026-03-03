import bpy
from . import properties, operators, panels

bl_info = {
    "name": "3D NCA Simulator",
    "author": "Michal Repcik",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > NCA",
    "description": "3D Neural Cellular Automata Simulator",
    "category": "Object",
}

modules = (
    properties,
    operators,
    panels,
)

def register():
    for module in modules:
        module.register()

def unregister():
    for module in reversed(modules):
        module.unregister()

if __name__ == "__main__":
    register()