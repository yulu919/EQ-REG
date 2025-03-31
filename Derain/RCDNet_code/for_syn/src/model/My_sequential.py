
from model.My_module import MyModule
import operator
from itertools import islice
from collections import OrderedDict

__all__ = ["MySequential"]


class MySequential(MyModule):
    
    def __init__(self, *args,):
        r"""
        
        A sequential container similar to :class:`torch.nn.Sequential`.
        
        The constructor accepts both a list or an ordered dict of :class:`~e2cnn.nn.EquivariantModule` instances.
        
        Example::
        
            # Example of SequentialModule
            s = e2cnn.gspaces.Rot2dOnR2(8)
            c_in = e2cnn.nn.FieldType(s, [s.trivial_repr]*3)
            c_out = e2cnn.nn.FieldType(s, [s.regular_repr]*16)
            model = e2cnn.nn.SequentialModule(
                      e2cnn.nn.R2Conv(c_in, c_out, 5),
                      e2cnn.nn.InnerBatchNorm(c_out),
                      e2cnn.nn.ReLU(c_out),
            )

            # Example with OrderedDict
            s = e2cnn.gspaces.Rot2dOnR2(8)
            c_in = e2cnn.nn.FieldType(s, [s.trivial_repr]*3)
            c_out = e2cnn.nn.FieldType(s, [s.regular_repr]*16)
            model = e2cnn.nn.SequentialModule(OrderedDict([
                      ('conv', e2cnn.nn.R2Conv(c_in, c_out, 5)),
                      ('bn', e2cnn.nn.InnerBatchNorm(c_out)),
                      ('relu', e2cnn.nn.ReLU(c_out)),
            ]))
        
        """
        
        super(MySequential, self).__init__()
        self.rot_num = 1

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                # assert isinstance(module, MySequential)
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                # assert isinstance(module, MySequential)
                self.add_module(str(idx), module)

    def __iter__(self):
        return iter(self._modules.values())


    def __getitem__(self,idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __len__(self) -> int:
        return len(self._modules)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))
        
    def forward(self, input):

        x = input
        for m in self._modules.values():
            x = m(x)

        return x

    
    