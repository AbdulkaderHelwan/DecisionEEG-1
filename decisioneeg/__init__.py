"""DecisionEEG-1: calibrated decisions from any EEG montage, on the NOMAD-EEG backbone."""
from . import paths                      # noqa: F401  (puts the NOMAD-EEG repo on sys.path)
from .tasks import TASKS


def __getattr__(name):                   # lazy: importing the package does not load torch models
    if name == "DecisionEEG":
        from .model import DecisionEEG
        return DecisionEEG
    raise AttributeError(name)


__all__ = ["DecisionEEG", "TASKS"]
