# CARLA-GENERATOR
This is a co-simulator for autonomous driving that can be used to generate KITTI-compatible and semantic3D-compatible datasets. If you want to realize real-time network simulation, please cooperate with our sub-module [NS3-DSRC](https://github.com/S-kewen/NS3-DSRC), [NS3-NR-C-V2X](https://github.com/S-kewen/NS3-NR-C-V2X).

If you are interested in error concealment, please visit our [LiDAR Point Cloud Error Concealment System](https://github.com/S-kewen/lidar-base-point-cloud-error-concealment).


<div align=center>
<img src="doc/sequence_diagram.svg">
</div>
<!-- Code release for the paper XXXX. -->

## Installation
### Requirements
All the codes are tested in the following environment:
* OS (Ubuntu 20.04 or Windows 10)
* CARLA 0.9.12
* Python 3.7.13+

## Quick demo
### a. Build environment
```
conda create -n carla-generator python==3.7.13 -y
conda activate carla-generator
```
### b. Clone repository
```
git clone https://github.com/S-kewen/carla-generator
cd carla-generator
pip install -r requirements.txt
```
### c. Install CARLA
We recommend using [the binary release](https://github.com/carla-simulator/carla/releases/tag/0.9.12/), you can also [building CARLA from source code](https://carla.readthedocs.io/en/0.9.12/build_linux/).
- Windows 10
```
curl -0 https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/CARLA_0.9.12.zip --output CARLA_0.9.12.zip
tar -xf CARLA_0.9.12.zip
pip3 install WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.12-cp37-cp37m-win_amd64.whl
WindowsNoEditor\CarlaUE4.exe
```
- Ubuntu 20.04
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.12.tar.gz
tar -zxvf CARLA_0.9.12.tar.gz
cd CARLA_0.9.12
pip3 install CARLA_0.9.12/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg
./CARLA_0.9.12/CarlaUE4.sh
```
### d. Run
```
python generator.py --s NS3.ENABLE False # simple demo
python generator.py --s NS3.ENABLE True # with NS-3 simulator
```
For specific configuration, please refer to [config.yaml](config.yaml).

## Offline simulation
You also can store all the data and then run the NS-3 sub-module for network simulation.
```
python exp_offline_ns3.py --zmqPort {your_zmq_port} --path {your_dataset_path} --config {your_config_file}
```
Moreover, we support network simulation for the KITTI odometry dataset.
```
python exp_offline_ns3_kitti.py --zmqPort {your_zmq_port} --path {your_dataset_path} --config {your_config_file}
```
## Output structures
```
├── save_directory_name
│   ├── ImageSets
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── trainval.txt
│   │   ├── val.txt
│   ├── object
│   │   │   ├── training
│   │   │   │   ├── calib
│   │   │   │   ├── carla_label
│   │   │   │   ├── image_2
│   │   │   │   ├── image_label_2
│   │   │   │   ├── label_2
│   │   │   │   ├── location
│   │   │   │   ├── ns3
│   │   │   │   ├── packet
│   │   │   │   ├── planes
│   │   │   │   ├── ply
│   │   │   │   ├── semantic3d_label
│   │   │   │   ├── semantic3d_xyzirgb
│   │   │   │   ├── velodyne
│   │   │   │   ├── velodyne_compression
│   │   │   │   ├── velodyne_fg
│   │   │   │   ├── velodyne_ground_removal
│   │   │   ├── CAR2
│   │   │   │   ├── ...
│   │   │   ├── ...
│   ├── config.yaml
```

<!-- ## Citation 
If you find this project useful in your research, please consider cite:
```
XXXX
``` -->

## Limitations 
We are happy to improve this project together, please submit your pull request if you fixed these limitations.
- [ ] Calib: Our sensors are installed in a fixed location and can not provide calibration replacement.
- [ ] LiDAR Intensity: The CARLA LiDAR sensor only provides virtual intensity without considering the material.

## Acknowledgment

Part of our code refers to the work [DataGenerator](https://github.com/mmmmaomao/DataGenerator).

## Contribution
welcome to contribute to this repo, please feel free to contact us with any potential contributions.