from enum import Enum 

class PruneAlgo(Enum):
    ACT_PATCH = "act_patch"
    ATTR_PATCH = "attr_patch"
    ACDC = "acdc"
    CIRC_PROBE = "circ_probe"
    EDGE_PROBE = "edge_probe"