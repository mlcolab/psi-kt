# Predictive, Scalable and Interpretable Knowledge Tracing (PSI-KT)

## Description 
- [ ] An in-depth paragraph about your project and overview of use.
- [ ] Paper link. 

## Getting Started

### Dependencies

Dependencies are in the `envrionment.yml` file.  

### Data preprocessing

We follow the preprocess as in the [HawkesKT](https://github.com/THUwangcy/HawkesKT) model.

### Baseline models
```
python predict_learner_performance_baseline.py --dataset assistment17 --model_name DKT --random_seed 2023
```

### PSI-KT
Running PSI-KT for prediction on bucket data:
```bash
python predict_learner_performance_psikt.py --dataset assistment17 --model_name AmortizedPSIKT --random_seed 2023
```
Running PSI-KT for continual learning can be set by specifying `--vcl 1`.


## Authors

- [ ] Contributors and contact info

## License

This project is licensed under the GNU Affero General Public License - see the LICENSE.md file for details

## Acknowledgments

The training architectures follow [HawkesKT](https://github.com/THUwangcy/HawkesKT).  
The baselines follow [XKT](https://github.com/tswsxk/XKT) and [pyKT](https://github.com/pykt-team/pykt-toolkit).  
The logging modules follow [AmortizedCausalDiscovery](https://github.com/loeweX/AmortizedCausalDiscovery).
