# Bandit

## Experiment Dependency

* python3.5
* numpy
* scipy, with community/atlas-lapack-base
* matplotlib, with cairocffi
* colorama

## Experiment Data

* Movielens
    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
    unzip ml-20m.zip -d movielens
* Rocketfuel
    wget http://research.cs.washington.edu/networking/rocketfuel/maps/rocketfuel_maps_cch.tar.gz
    tar -xvf rocketfuel_maps_cch.tar.gz -C isp

## Experiment

* Before running, please modify "flowtest.py" accordingly, then 
    python3 flowtest.py
* Logs in human-understandable format are attached in "log"
