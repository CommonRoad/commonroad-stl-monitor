from ind_evaluation import IndEvaluator
import os


def generate_learning_data(scenario_path, filename):
    root_path = os.getcwd()
    evaluator = IndEvaluator(
        scenario_path=scenario_path,
        save_filename=filename,
        save_filepath=root_path,
        max_scenario_number=400,
    )
    # evaluator.evaluation_parallel()
    evaluator.evaluation()


if __name__ == "__main__":
    scenario_path = "/home/ge23lac/scenarios/AAH1_test/"
    # scenario_path = "/home/zekun/MA/scenarios/AAH1_test"
    filename = "ind_violation.csv"
    generate_learning_data(scenario_path, filename)
