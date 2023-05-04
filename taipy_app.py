""" Creating a Taipy App to detect military vehicles in an image """

import os
from taipy.gui import Gui
from detect import run

input_path = ""
output_path = "temp/output.png"


def on_upload_action(state):
    """
    Runs YOLOv5 on the uploaded image and displays the output image.

    Args:
        state (dict): The state of the app.
    """
    output_folder = run(
        source=state.input_path,
        weights="military-reco/runs/train/yolo_arma4/weights/best.pt",
        conf_thres=0.5,
        name="yolo_arma",
    )
    image_path = os.path.join(output_folder, os.listdir(output_folder)[0])
    state.output_path = image_path


page = """
# Military Vehicles Image Recognition
### Finds military vehicles in an aerial drone image using YOLOv5.

<|{input_path}|file_selector|label=Upload Image Here|on_action=on_upload_action|extensions=".PNG"|>

________________________________________________________

<|{output_path}|image|label=Output Image|width=50%|>
"""

app = Gui(page=page)
app.run(use_reloader=True)
