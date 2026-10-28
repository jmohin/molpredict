"""Model registry. Add a new model by implementing `Predictor` and listing it here."""

from .base import ModelInfo, Prediction, Predictor, PropertyInfo


def load_predictors() -> list[Predictor]:
    from .admet_ai import ADMETAI
    from .rdkit_descriptors import RDKitDescriptors
    from .soltrannet import SolTranNet

    return [RDKitDescriptors(), ADMETAI(), SolTranNet()]


__all__ = ["ModelInfo", "Prediction", "Predictor", "PropertyInfo", "load_predictors"]
