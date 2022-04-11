from abc import ABC
from typing import Generic

from kilroytorch.models.base import A, B, BaseModel, ForwardValidation


class DistributionModel(
    ForwardValidation[A, B], BaseModel[A, B], ABC, Generic[A, B]
):
    pass
