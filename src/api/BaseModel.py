from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):
    @abstractmethod
    def predict(self, img, url) -> dict:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def cleanup(self) -> None:
        pass
