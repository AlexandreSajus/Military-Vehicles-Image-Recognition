import pandas as pd

import train, detect


def launch_train(model_name: str = "yolov5s.pt", batch_size: int = 16):
    name = "gd_v5"
    model_size = model_name.split(".")[0].split("yolov5")[1]
    run_name = f"{name}{model_size}_{str(batch_size)}"
    train.run(
        data="arma.yaml",
        # Change to 50 in production
        epochs=3,
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
