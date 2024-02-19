import sys

sys.path.append("..")

import pickle
import argparse

import pandas as pd
import numpy as np

import knowledge_tracing.utils.visualize as visualize
import knowledge_tracing.utils.utils as utils

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def parse_args(parser):
    parser.add_argument(
        "--base_pth",
        type=str,
    )
    parser.add_argument("--dataset", type=str, default=1, help="Name of dataset")
    parser.add_argument(
        "--gap",
        type=int,
        default=1,
        help="The order of transition in evaluation causal support",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=1,
        help="Whether using test data in evaluation causal support",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=1e6,
        help="Number of samples in estimating causal support",
    )

    return parser


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser(description="Global")
    parser = parse_args(parser)
    args, extras = parser.parse_known_args()

    # assistment12
    inter = pd.read_csv(
        f"{args.base_pth}/{args.dataset}/multi_skill/interactions.csv", sep="\t"
    )
    with open(f"{args.base_pth}/{args.dataset}/multi_skill/Corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    skill_id = list(inter.skill_id.unique())
    num_node = len(skill_id)

    skill_list = []
    for i in range(len(skill_id)):
        text = list(inter.loc[inter["skill_id"] == i]["skill_text"])[0]
        skill_list.append(text)

    # ----- Calculate transition matrix -----
    gap = args.gap
    start = 10 if args.test else 0
    T = np.zeros((num_node, num_node, 4))  # 0-1, 0-0, 1-1, 1-0
    N = np.zeros((num_node, num_node))
    for l in range(len(corpus.user_seq_df)):
        correct = corpus.user_seq_df["correct_seq"][l]
        index = corpus.user_seq_df["skill_seq"][l]

        for i in range(start, start + 10 - gap):
            if index[i + gap] != index[i]:
                if correct[i] == 0:
                    if correct[i + gap] == 1:
                        T[index[i], index[i + gap], 0] += 1
                    else:
                        T[index[i], index[i + gap], 1] += 1
                else:
                    if correct[i + gap] == 1:
                        T[index[i], index[i + gap], 2] += 1
                    else:
                        T[index[i], index[i + gap], 3] += 1
                N[index[i], index[i + gap]] += 1
    success_transition = abs(T[..., 2]) / (T[..., 2] + T[..., 3] + 1e-6)
    mask = T[..., 2] + T[..., 3] + T[..., 0] + T[..., 1] > 1

    Nc_minus = T[..., 0] + T[..., 1]
    Nc_plus = T[..., 2] + T[..., 3]
    Ne_minus = T[..., 1] + T[..., 3]
    Ne_plus = T[..., 0] + T[..., 2]

    # ----- Compute causal support -----
    # P(D|G0)
    num_sample = args.num_sample
    w0 = np.arange(0, num_sample, 1) / num_sample
    w0 = w0.reshape(num_sample, 1, 1).repeat(num_node, 1).repeat(num_node, -1)
    p0 = np.power(w0, np.expand_dims(Ne_plus, 0).repeat(num_sample, 0)) * np.power(
        1 - w0, np.expand_dims(Ne_minus, 0).repeat(num_sample, 0)
    )

    # P(D|G1)
    w0 = np.arange(0, num_sample, 1) / num_sample
    w0 = w0.reshape(num_sample, 1, 1).repeat(num_node, 1).repeat(num_node, -1)
    w0 = w0.repeat(num_sample, 0)
    w1 = (
        np.arange(0, num_sample, 1) / num_sample
    )  # w1 = p_edge.reshape(1, num_node, num_node).repeat(num_sample, 0)
    w1 = w1.reshape(num_sample, 1, 1).repeat(num_node, 1).repeat(num_node, -1)
    w1 = np.tile(w1, (num_sample, 1, 1))
    N_e1_c1 = np.expand_dims(T[..., 2], 0).repeat(num_sample * num_sample, 0)
    p_e1_c1 = np.power(np.multiply(w0, 1 - w1), N_e1_c1)

    N_e1_c0 = np.expand_dims(T[..., 0], 0).repeat(num_sample * num_sample, 0)
    p_e1_c0 = np.power(w0, N_e1_c0)

    p1 = np.multiply(p_e1_c1, p_e1_c0)

    # Support
    support = np.log(p1.mean(0) + 1e-6) - np.log(p0.mean(0) + 1e-6)
