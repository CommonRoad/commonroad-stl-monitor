import logging
from pathlib import Path

from crmonitor.mpr import DataLoader, ModelTrainer
from crmonitor.common import ScenarioType
from crmonitor.predicates.position import PredInFrontOf

logging.basicConfig(level=logging.INFO)

output_folder_path = Path(__file__).parents[2] / "output"

learning_data_path = output_folder_path / "learning_data" / "learning_data.csv"
models_output_path = output_folder_path / "models"


data_loader = DataLoader.create_from_file(learning_data_path)

trainer = ModelTrainer(data_loader, ScenarioType.INTERSTATE, training_iter=400)

model_containers = trainer.train([PredInFrontOf])

for predicate_name, model_container in model_containers.items():
    model_container.write_to_folder(models_output_path)
