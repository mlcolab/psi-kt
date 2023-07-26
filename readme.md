# Hierarchical State Space Model 

Simple overview of use/purpose.

## Description 

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

Draft: 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ipdb
pip install tqdm
pip install networkx

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* Baseline models
```
cd baseline
python exp_baseline.py --dataset assistment17 --model_name DKT --random_seed 2023
```

* Our model
```
cd exp
python exp_learner_predict.py --dataset assistment17 --model_name AmortizedHSSM --random_seed 2023
```


## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [XKT](https://github.com/tswsxk/XKT)