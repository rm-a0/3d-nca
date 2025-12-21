bl_info = {
    "name": "3D Neural Celular Automata Simulator",
    "author": "rm-a0",
    "description": "A Blender add-on for 3D neural cellular automata simulations",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D -> Sidebar -> 3D NCA Sim",
    "category": "Object"
}

from . import properties, operators, panels

def register():
    properties.register()
    operators.register()
    panels.register()

def unregister():
    panels.unregister()
    operators.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()