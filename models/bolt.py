# from .BolT.model import Model
from .BolT.bolT import BolT
import copy

def get_BolT(node_sz, out_channel, in_channel, nlayer=4, **kargs):
    
    hyperParams = {

        "weightDecay" : 0,

        "lr" : 2e-4,
        "minLr" : 2e-5,
        "maxLr" : 4e-4,

        # FOR BOLT
        # "nOfLayers" : 4,
        "nOfLayers" : nlayer,
        "dim" : node_sz,

        "numHeads" : 8,
        "headDim" : 5,
        # "numHeads" : 36,
        # "headDim" : 20,

        "windowSize" : 20,
        "shiftCoeff" : 2.0/5.0,            
        "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
        "focalRule" : "expand",

        "mlpRatio" : 1.0,
        "attentionBias" : True,
        "drop" : 0.1,
        "attnDrop" : 0.1,
        "lambdaCons" : 1,

        # extra for ablation study
        "pooling" : "cls", # ["cls", "gmp"]         
            

    }
    hyperParams = Option(hyperParams)

    return BolT(hyperParams, in_channel, out_channel, node_sz)


class Option(object):
      
    def __init__(self, my_dict):

        self.dict = my_dict

        for key in my_dict:
            setattr(self, key, my_dict[key])

    def copy(self):
        return Option(copy.deepcopy(self.dict))