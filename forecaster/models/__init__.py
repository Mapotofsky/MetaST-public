from typing import Union

import torch

from .nbeats.nbeats import nbeats
from .metast.metast import metast
from .koopa.koopa import koopa
from .deeptime.DeepTIMe import deeptime


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'nbeats':
        model = nbeats(kwargs['device'])
    elif model_type == 'koopa':
        model = koopa(kwargs['mask_spectrum'])
    elif model_type == 'metast':
        model = metast(kwargs['device'])
    elif model_type == 'deeptime':
        model = deeptime()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
