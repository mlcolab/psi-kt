# Hierarchical State Space Model 

Simple overview of use/purpose.

## Description 
- [ ] An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

Dependencies are in the `envrionment.yml` file.  

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* Baseline models
```
cd script
python predict_learner_performance_baseline.py --dataset assistment17 --model_name DKT --random_seed 2023
```

* Our model
```
cd script
python predict_learner_performance_groupkt.py --dataset assistment17 --model_name AmortizedGroupKT --random_seed 2023
```


## Authors

Contributors and contact info


## Version History

* 0.2  
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [XKT](https://github.com/tswsxk/XKT)