# Predictive, Scalable and Interpretable Knowledge Tracing (PSI-KT)

## Description 
- [ ] An in-depth paragraph about your project and overview of use.
- [ ] Paper link. 

## Getting Started

### Dependencies

Dependencies are in the `envrionment.yml` file.  

### Executing program

* Data preprocessing
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

To reproduce the results in the paper, the five random seeds we use are 2023, 2022, 2021, 2020, 2019.  
We report the mean and standard deviation of results from these five experiments.  

## Authors

- [ ] Contributors and contact info

## License

- [ ] This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration:
* [XKT](https://github.com/tswsxk/XKT)  
Code review: 