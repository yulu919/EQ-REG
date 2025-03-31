from torch.nn import Module, Sequential
from abc import ABC, abstractmethod

from typing import Tuple

__all__ = ["MyModule"]


class MyModule(Module, ABC):

    def __init__(self):
        super(MyModule, self).__init__()
        self.rot_num = 1


    @abstractmethod
    def forward(self, *input):
        pass

    
    def rot_num_in(self, rot_num = 0):

        self.rot_num = rot_num
        for module in self.children():
            # print(module)
            if hasattr(module, 'rot_num'):
                # print(module)
                module.rot_num_in(rot_num)
        return self
    
    
    