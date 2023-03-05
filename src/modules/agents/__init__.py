REGISTRY = {}

from .rnn_agent import RNNAgent
from .role_agent import RoleAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["sra"] = RoleAgent