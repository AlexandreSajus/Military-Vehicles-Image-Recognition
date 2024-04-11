import taipy as tp
from taipy import Config, Core
import pandas as pd

import train, detect


def launch_train(model_name: str = "yolov5s.pt", batch_size: int = 16):
    name = "gd_v5"
    model_size = model_name.split(".")[0].split("yolov5")[1]
    run_name = f"{name}{model_size}_{str(batch_size)}"
    train.run(
        data="arma.yaml",
        # Change to 50 in production
        epochs=50,
        weights=model_name,
        batch_size=batch_size,
        name=run_name,
    )
    return run_name


def get_results(run_name: str):
    results = pd.read_csv(f"runs/train/{run_name}/results.csv")
    return results


def run_inference(run_name: str):
    detect.run(
        source="./input/",
        weights=f"runs/train/{run_name}/weights/best.pt",
        conf_thres=0.5,
        name=run_name,
    )
    return f"runs/detect/{run_name}"


def configure():
    model_name_cfg = Config.configure_data_node("model_name")
    batch_size_cfg = Config.configure_data_node("batch_size")
    run_name_cfg = Config.configure_data_node("run_name")
    train_cfg = Config.configure_task(
        "train",
        function=launch_train,
        input=[model_name_cfg, batch_size_cfg],
        output=[run_name_cfg],
    )
    results_cfg = Config.configure_data_node("results")
    get_results_cfg = Config.configure_task(
        "get_results", function=get_results, input=[run_name_cfg], output=[results_cfg]
    )
    output_path_cfg = Config.configure_data_node("output_path")
    run_inference_cfg = Config.configure_task(
        "infer",
        function=run_inference,
        input=[run_name_cfg],
        output=[output_path_cfg],
    )
    scenario_cfg = Config.configure_scenario(
        "scenario_configuration",
        task_configs=[train_cfg, get_results_cfg, run_inference_cfg],
    )
    Config.export("scenario.toml")
    return scenario_cfg


if __name__ == "__main__":
    core = Core()
    default_scenario_cfg = configure()
    core.run()

    model_name = "yolov5s.pt"
    batch_size = 16

    scenario = tp.create_scenario(
        default_scenario_cfg, name=f"gd_v5_{model_name}_{batch_size}"
    )
    scenario.model_name.write(model_name)
    scenario.batch_size.write(batch_size)
    tp.submit(scenario)
    print(scenario.results.read())
    print(scenario.output_path.read())
