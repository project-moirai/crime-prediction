from abc import ABC, abstractmethod
from typing import Literal


class PredictionModel(ABC):
    @abstractmethod
    def predict(
        self,
        date: str,  # MM-DD format
        time: str,  # HH:MM format
        lat: float,
        lng: float,
    ) -> Literal[
        "alcohol-related-incident",
        "arson",
        "assault",
        "criminal-damage",
        "domestic-violence",
        "drug-offense",
        "environmental-offense",
        "fraud",
        "gun-offense",
        "intrusion",
        "mountain-accident",
        "murder",
        "no-incident",
        "other",
        "public-disturbance",
        "robbery",
        "suicide",
        "theft",
        "traffic-accident",
        "vandalism",
    ]:
        pass
