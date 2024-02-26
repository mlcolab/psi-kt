# Predictive, Scalable and Interpretable Knowledge Tracing (PSI-KT)

[![Project Status: Active – The project has reached a usable state and is under active development.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)


## About The Project

Knowledge Tracing (KT) is a fundamental task in the realm of intelligent tutoring systems, aimed at predicting how students learn and progress over time. PSI-KT introduces a novel approach to KT, leveraging a hierarchical state-space model to improve the prediction, scalability, and interpretability of knowledge tracing on structured domains. This is the official repository for our work [Predictive, scalable and interpretable knowledge tracing on structured domains](https://openreview.net/forum?id=NgaLU2fP5D&referrer=%5Bthe%20profile%20of%20Hanqi%20Zhou%5D(%2Fprofile%3Fid%3D~Hanqi_Zhou1)), offering a unique blend of predictive accuracy, efficient scalability, and insightful interpretability.

> **Abstract**
>
> Intelligent tutoring systems optimize the selection and timing of learning materials to enhance understanding and long-term retention. This requires estimates of both the learner's progress ("knowledge tracing"; KT), and the prerequisite structure of the learning domain ("knowledge mapping"). While recent deep learning models achieve high KT accuracy, they do so at the expense of the interpretability of psychologically-inspired models. In this work, we present a solution to this trade-off. PSI-KT is a hierarchical generative approach that explicitly models how both individual cognitive traits and the prerequisite structure of knowledge influence learning dynamics, thus achieving interpretability by design. Moreover, by using scalable Bayesian inference, PSI-KT targets the real-world need for efficient personalization even with a growing body of learners and interaction data. Evaluated on three datasets from online learning platforms, PSI-KT achieves superior multi-step predictive accuracy and scalable inference in continual-learning settings, all while providing interpretable representations of learner-specific traits and the prerequisite structure of knowledge that causally supports learning. In sum, predictive, scalable and interpretable knowledge tracing with solid knowledge mapping lays a key foundation for effective personalized learning to make education accessible to a broad, global audience.

## Getting Started

### Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/mlcolab/psi-kt.git
cd psi-kt
conda env create -f environment.yml
conda activate psi-kt
```

### Data preprocessing

We follow the preprocessing steps outlined in the [HawkesKT](https://github.com/THUwangcy/HawkesKT) model.

### Training and evaluation
To train different models and reproduce the results in Figures 2 and 3:

Running PSI-KT for prediction on bucket data:
```bash
python predict_learner_performance_psikt.py 
--dataset assistment17 
--model_name AmortizedPSIKT 
--random_seed 2023
--num_learner 100
```
Arguments Explained:
- `random_seed`: Sets the seed for random number generation to ensure reproducibility of results. You can specify any integer value, but we recommend using one of the following for consistent experimentation: 2023, 2022, 2021, 2020, or 2019. 
- `dataset`: Specifies the dataset to be used for training and evaluation. We support several predefined datasets, including assistment12, assistment17, and junyi15. You can also use your custom Knowledge Tracing (KT) dataset by providing its name here. 
- `model_name`: Specifies the model to be used for training and evaluation. We support HLR, PPE, DKT, DKTForgetting, AKT, HKT, GKT, QIKT. 
- `vcl`: Enables Variational Continual Learning (VCL) by setting this argument to 1. VCL helps in adapting the model to new data over time without forgetting previously learned information. 
- `early_stop`: Determines whether to stop the training process early if the performance on the validation dataset does not improve. 



## License

This project is licensed under the GNU Affero General Public License - see the [LICENSE.md](https://github.com/mlcolab/psi-kt/blob/main/LICENSE) for details. 

## Acknowledgments

The training architectures follow [HawkesKT](https://github.com/THUwangcy/HawkesKT).  
The baselines follow [XKT](https://github.com/tswsxk/XKT) and [pyKT](https://github.com/pykt-team/pykt-toolkit).  
The logging modules follow [AmortizedCausalDiscovery](https://github.com/loeweX/AmortizedCausalDiscovery).


## Citation
Following is the Bibtex if you would like to cite our paper:

```bibtex
@inproceedings{
  zhou2024predictive,
  title={Predictive, scalable and interpretable knowledge tracing on structured domains},
  author={Hanqi Zhou and Robert Bamler and Charley M Wu and {\'A}lvaro Tejero-Cantero},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=NgaLU2fP5D}
}
```