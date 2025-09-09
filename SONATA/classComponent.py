# Third party modules
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt


class Component:
    """Describes a Sonata Component Object

    Attributes
    ----------
    name : str
        name of the component

    Ax2 : gp_Ax2
        Describes a right-handed coordinate system in 3D space. It is part of
        the parent class 'Component'. It is an instance from the opencascade
        gp_Ax2 class

    """

    __slots__ = ("name", "Ax2")

    def __init__(self, name="NONAME", *args, **kwargs):
        self.name = name
        self.Ax2 = gp_Ax2(*args)

    def __repr__(self):
        """Return the official string representation of the object."""
        return f"Component: {self.name}"


if __name__ == "__main__":
    C = Component(gp_Pnt(0, 10, 0), gp_Dir(1, 0, 0), name="TestComponent")
