import taipy as tp
from taipy.gui import Gui, State
import pandas as pd

scenarios = tp.get_scenarios()
scenario_names = [scenario.name for scenario in scenarios]
scenario_results = [scenario.results.read() for scenario in scenarios]
scenario_output_paths = [scenario.output_path.read() for scenario in scenarios]

selected_metric = "Validation mAP 0.5"
metrics_list = [
    "Box Loss",
    "Obj Loss",
    "Cls Loss",
    "Precision",
    "Recall",
    "Validation mAP 0.5",
    "Validation mAP 0.5:0.95",
]
metrics_dict = {
    "Epoch": "               epoch",
    "Box Loss": "      train/box_loss",
    "Obj Loss": "      train/obj_loss",
    "Cls Loss": "      train/cls_loss",
    "Precision": "   metrics/precision",
    "Recall": "      metrics/recall",
    "Validation mAP 0.5": "     metrics/mAP_0.5",
    "Validation mAP 0.5:0.95": "metrics/mAP_0.5:0.95",
}
chart_data = pd.DataFrame(
    {
        "Epoch": scenario_results[0][metrics_dict["Epoch"]].tolist(),
        "YOLOv5n": scenario_results[scenario_names.index("YOLOv5n")][
            metrics_dict[selected_metric]
        ].tolist(),
        "YOLOv5m": scenario_results[scenario_names.index("YOLOv5m")][
            metrics_dict[selected_metric]
        ].tolist(),
        "YOLOv5x": scenario_results[scenario_names.index("YOLOv5x")][
            metrics_dict[selected_metric]
        ].tolist(),
    }
)

selected_image = "test_crossroad.png"
image_list = [
    "test_crossroad.png",
    "test_marid_1.png",
    "test_marid_2.png",
    "test_zamak_1.png",
    "test_zamak_2.png",
]
image_path_1 = "runs/detect/gd_v5n_8/test_crossroad.png"
image_path_2 = "runs/detect/gd_v5x_8/test_crossroad.png"
selected_scenario_1 = "YOLOv5n"
selected_scenario_2 = "YOLOv5x"
scenario_list = ["YOLOv5n", "YOLOv5m", "YOLOv5x"]


def change_metric(state: State):
    state.chart_data = pd.DataFrame(
        {
            "Epoch": scenario_results[0][metrics_dict["Epoch"]].tolist(),
            "YOLOv5n": scenario_results[scenario_names.index("YOLOv5n")][
                metrics_dict[state.selected_metric]
            ].tolist(),
            "YOLOv5m": scenario_results[scenario_names.index("YOLOv5m")][
                metrics_dict[state.selected_metric]
            ].tolist(),
            "YOLOv5x": scenario_results[scenario_names.index("YOLOv5x")][
                metrics_dict[state.selected_metric]
            ].tolist(),
        }
    )


def change_image(state: State):
    state.image_path_1 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_1)]}/{state.selected_image}"
    state.image_path_2 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_2)]}/{state.selected_image}"
    print(state.image_path_1)


def change_scenario_1(state: State):
    state.image_path_1 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_1)]}/{state.selected_image}"


def change_scenario_2(state: State):
    state.image_path_2 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_2)]}/{state.selected_image}"


page = """
<|container|

# 🛰️ Vehicle Image Recognition

<intro_card|card|

## Comparing the performance of different YOLOv5 models on drone imagery

In this application, we compare the performance of different YOLOv5 models for detecting and recognizing 
military vehicles in drone imagery.

Learn more about this project <a href="https://github.com/AlexandreSajus/Military-Vehicles-Image-Recognition" target="_blank">here</a>.

<br/>

<p align="center">
  <img src="example_inference.png" alt="Example Inference" width="40%"/>
</p>

<p align="center">
  <img src="model_comparison.png" alt="YOLOv5 Model Sizes" width="50%"/>
</p>


|intro_card>

<br/>

<chart_card|card|

## Metrics over Epochs 📈

<|{selected_metric}|selector|dropdown|lov={metrics_list}|on_change=change_metric|>
<|{chart_data}|chart|type=line|x=Epoch|y[3]=YOLOv5n|y[2]=YOLOv5m|y[1]=YOLOv5x|title=Metric over Epochs for YOLOv5 Models|rebuild|>

|chart_card>

<br/>

<image_card|card|

## Compare Results 🖼️

<|{selected_image}|selector|dropdown|lov={image_list}|on_change=change_image|>

<images|layout|columns=1 1|

<|{selected_scenario_1}|selector|dropdown|lov={scenario_list}|on_change=change_scenario_1|class_name=fullwidth|><br/>
<center><|{image_path_1}|image|width=60vh|></center>

<|{selected_scenario_2}|selector|dropdown|lov={scenario_list}|on_change=change_scenario_2|class_name=fullwidth|><br/>
<center><|{image_path_2}|image|width=60vh|></center>

|images>

|image_card>

|>
"""

gui = Gui(page)
gui.run(dark_mode=False, debug=True, title="🛰️Vehicle Image Recognition")
