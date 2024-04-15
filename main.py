import taipy as tp
from taipy import Config, Core
from taipy.gui import Gui

from algos import launch_train, get_results, run_inference


def configure():
    Config.configure_job_executions(mode="cluster")
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

    for model_size in ["s", "m", "l"]:
        model_name = f"yolov5{model_size}.pt"
        run_name = f"YOLOv5{model_size}"
        batch_size = 1

        scenario = tp.create_scenario(default_scenario_cfg, name=run_name)
        scenario.model_name.write(model_name)
        scenario.batch_size.write(batch_size)
        tp.submit(scenario)
        print(f"Results: {scenario.results.read()}")
        print(f"Output path: {scenario.output_path.read()}")
    Gui("").run()
    # Before new batch of runs,
    # Remove labels cache, user_data, .taipy, previous runs
