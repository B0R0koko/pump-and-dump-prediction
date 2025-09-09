from typing import Protocol


class BaseModelTrait(Protocol):

    def predict(self, *args, **kwargs):
        ...

    def predict_proba(self, *args, **kwargs):
        ...
