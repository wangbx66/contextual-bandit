# Contextual Bandit 

The repo features the experimental analysis on [Contextual Combinatorial Cascading Bandits](http://icml.cc/2016/?page_id=1839#581). We highly recommend cite the the paper, using the following bib

```
@inproceedings{li2016contextual,
  title={Contextual Combinatorial Cascading Bandits},
  author={Li, Shuai and Wang, Baoxiang and Zhang, Shengyu and Chen, Wei},
  booktitle={Proceedings of The 33rd International Conference on Machine Learning},
  pages={1245--1253},
  year={2016}
}
```

## Experiment Dependency

* python3.5
* numpy
* scipy, with community/atlas-lapack-base
* matplotlib, with cairocffi
* colorama

## Experiment Data

* Movielens

``` 
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip -d movielens
```

* Rocketfuel

```
wget http://research.cs.washington.edu/networking/rocketfuel/maps/rocketfuel_maps_cch.tar.gz
wget http://research.cs.washington.edu/networking/rocketfuel/maps/weights-dist.tar.gz
wget http://research.cs.washington.edu/networking/rocketfuel/maps/rocketfuel-traces.tgz
tar -xvf rocketfuel_maps_cch.tar.gz -C isp
tar -xvf weights-dist.tar.gz -C isp-weight
tar -xvf rocketfuel-traces.tgz -C isp-trace
```

## Experiment

* Before running, please modify the code *bleedtest.py* accordingly, then

```
python3 bleedtest.py
```

* Logs in human-understandable format are attached in *log*
