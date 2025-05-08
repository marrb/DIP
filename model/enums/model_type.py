# Autor: Martin Bublavý [xbubla02]

from enum import Enum

class ModelType(Enum):
    """Enum for model types."""
    VIDEO_P2P = "video_p2p"
    VIDEO_P2P_EI = "video_p2p_ei"
    VIDEO_P2P_EI_PLUS = "video_p2p_ei_plus"
    