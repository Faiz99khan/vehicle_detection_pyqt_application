
# vehicle_detection_pyqt_application

## overview
It is a vehicle detection application which is made using [PyQt5(GUI toolkit for python)](https://www.riverbankcomputing.com/software/pyqt/). It can detect 4 types of vehicles (Car, Bus, Truck, Motorclycle). In this project, I wanted to add a feature that is if two vehicles come close and cross a safety distance, locate those vehicles in a frame( i locate such vehicles with bounding box of red color and a line b/w them) which work on approximation not accurately because this feature requires exact information of what focal length or FOV(Field of view), height and angle of installed camera are to work well. I have developed different method to find approximate distance b/w cars to just demonstrate my feature.

## Results
<p align="center">
<img src="results/screenshot1.PNG" >
</p>
<img src="results/screenshot2.PNG" >
<p align="center">
<img src="results/screenshot3.PNG" >
</p>
