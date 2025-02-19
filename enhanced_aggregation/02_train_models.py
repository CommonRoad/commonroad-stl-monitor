from pathlib import Path

from commonroad_mpr.learning.data_loader import DataLoader
from commonroad_mpr.learning.gp_regression import ModelTrainer

learning_data_path = (
    Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"
)

data_loader = DataLoader.create_from_file(learning_data_path)
raise RuntimeError("Training of new models is currently work in progress.")
trainer = ModelTrainer.create_from_config(data_loader)
