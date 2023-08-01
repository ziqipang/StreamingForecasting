import numpy as np


class Position:
    def __init__(self, array) -> None:
        """ xyz
        """
        self.xyz = np.array(array)
    
    def __str__(self):
        return self.xyz
    
    def copy(self, src):
        self.xyz = src.xyz
        return
    
    def pos2world(self, ego_matrix, inplace=False):
        xyz = ego_matrix[:3, :3] @ self.xyz.reshape((3, 1))
        xyz = xyz.reshape(-1) + ego_matrix[:3, 3]
        if inplace:
            self.xyz = xyz
        return Position(xyz)