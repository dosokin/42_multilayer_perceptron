import json
from pathlib import Path

from dataclasses import dataclass
from multilayer_perceptron.core.layer import LayerParameters

@dataclass
class Model:
    layers: list[LayerParameters]

    def to_struct(self):
        return {
            "layers": [
                {
                    "weights": l.weights.tolist(),
                    "biases": l.biases.tolist()
                } for l in self.layers]
        }

    def save(self, filename):
        output_file = Path(f"models/{filename}")

        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

        print(f"SAVING MODEL INTO {output_file.absolute()}")

        with output_file.open('w') as f:
            json.dump(self.to_struct(), f)

@dataclass
class ModelPerformance:
    loss: float|None
    model: Model|None

