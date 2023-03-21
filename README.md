# Military Vehicles Image Recognition

Fine-Tuning YOLOv5 to detect Military Vehicles in Aerial ARMA 3 Imagery


<p align="center">
  <img src="media/compound.png" alt="Compound" width="80%"/>
</p>

<p align="center">
  <img src="media/surveillance.gif" alt="Surveillance gif" width="80%"/>
</p>

Currently, the model is able to detect the following classes:
- CSAT Varsuk
- CSAT Marid
- CSAT Zamak (Transport)

The model is a YOLOv5 fine tuned using 100 images of each class using various environments and angles at noon clear sky using a UAV at around 100 meters altitude.

The dataset used is available on [Kaggle](https://www.kaggle.com/datasets/alexandresajus/arma3cvdataset).

## How to use

1. Clone the repository

```bash
git clone https://github.com/AlexandreSajus/Military-Vehicles-Image-Recognition.git
```

2. Install the requirements

```bash
pip install -r requirements.txt
```

3. Add your images to the `input` folder

4. Run the model

```bash	
python detect.py --source ./input/ --weights runs/train/yolo_arma4/weights/best.pt --conf 0.5 --name yolo_arma
```

5. The results will be available in a `runs/detect/yolo_armaX` folder