from typing import List

import numpy as np


def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        v0 = outputs[0][k]
        # Handle Python scalars and NumPy scalar types
        if np.isscalar(v0) or isinstance(v0, np.generic):
            collated[k] = [o[k] for o in outputs]
        # Handle NumPy arrays
        elif isinstance(v0, np.ndarray) or (hasattr(v0, 'shape') and hasattr(v0, 'dtype')):
            try:
                collated[k] = np.vstack([np.asarray(o[k])[None] for o in outputs])
            except Exception:
                # Fallback: try simple stacking without adding a new axis first
                try:
                    collated[k] = np.stack([np.asarray(o[k]) for o in outputs])
                except Exception as e:
                    raise ValueError(f'Cannot collate key {k!r} with array-like values of shapes '
                                     f"{[getattr(np.asarray(o[k]), 'shape', None) for o in outputs]}: {e}")
        elif isinstance(v0, list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(v0)} for key {k!r}. '
                             f'Modify collate_outputs to add this functionality')
    return collated
