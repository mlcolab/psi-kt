# Predictive, Scalable and Interpretable Knowledge Tracing (PSI-KT)

## Description 
- [ ] An in-depth paragraph about your project and overview of use.
- [ ] Paper link. 

## Getting Started

### Dependencies

Dependencies are in the `envrionment.yml` file.  

### Executing program

* Data preprocess
We follow the preprocess as in the Hawkes KT model: https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb

* Baseline models
```
python predict_learner_performance_baseline.py --dataset assistment17 --model_name DKT --random_seed 2023
```

* PSI-KT
```bash
python predict_learner_performance_psikt.py --dataset assistment17 --model_name AmortizedPSIKT --random_seed 2023
```

* Continual learning 

```bash
python predict_learner_performance_psikt.py --dataset assistment17 --model_name AmortizedPSIKT --random_seed 2023 --vcl 1
```

## Authors

Contributors and contact info

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration:
* [XKT](https://github.com/tswsxk/XKT)  
Code review: 