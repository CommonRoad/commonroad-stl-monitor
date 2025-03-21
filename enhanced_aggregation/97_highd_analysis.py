import pandas as pd

from pathlib import Path
from tqdm import tqdm

from commonroad_mpr.learning.data_generation import min_max_vel_acc_in_scenario

highd_scenarios = Path(__file__).parents[3] / "highD-scenarios"
save_path = Path(__file__).parents[1] / "output" / "metrics" / "min_max_velocity.csv"

df = pd.DataFrame(
    columns=["scenario", "min_velocity", "max_velocity", "min_acceleration", "max_acceleration"]
)

for scenario_path in tqdm(highd_scenarios.glob("*.xml"), desc="Processing scenarios"):
    min_vel, max_vel, min_acc, max_acc = min_max_vel_acc_in_scenario(scenario_path)
    new_row = pd.DataFrame(
        {
            "scenario": scenario_path.stem,
            "min_velocity": min_vel,
            "max_velocity": max_vel,
            "min_acceleration": min_acc,
            "max_acceleration": max_acc,
        },
        index=[0],
    )
    df = pd.concat([df, new_row], ignore_index=True)

df.to_csv(save_path, index=False)
