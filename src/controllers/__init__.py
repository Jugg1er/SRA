REGISTRY = {}

from .basic_controller import BasicMAC
from .my_controller import MyMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["my_mac"] = MyMAC