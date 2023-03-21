# Military Vehicles Image Recognition

Fine-Tuning YOLOv5 to detect Military Vehicles in Aerial ARMA 3 Imagery

<p align="center">
  <img src="media/compound.png" alt="Compound" width="70%"/>
</p>

<p align="center">
  <img src="media/crossroad.png" alt="Crossroad" width="70%"/>
</p>

Currently, the model is able to detect the following classes:
- CSAT Varsuk
- CSAT Marid
- CSAT Zamak (Transport)

The model is a YOLOv5 fine tuned using 100 images of each class using various environments and angles at noon clear sky using a UAV at around 100 meters altitude.

The dataset used is available on [Kaggle](https://www.kaggle.com/datasets/alexandresajus/arma3cvdataset).