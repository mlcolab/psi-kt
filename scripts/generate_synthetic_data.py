import sys
sys.path.append("..")

import os
import argparse
import datetime

import numpy as np
from numpy.random import default_rng
import pandas as pd
import networkx as nx

import torch

from knowledge_tracing.baseline.learner_model import *
from knowledge_tracing.utils.visualize import *


def parse_args(parser):
    # ----- global -----
    parser.add_argument(
        "--random_seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_sequence", type=int, default=1, help="Number of sequences to generate"
    )
    parser.add_argument(
        "--learner_model",
        type=str,
        default="graph_ou",
        choices=["hlr", "ou", "graph_ou", "egraph_ou", "ppe"],
        help="Type of learner model: hlr, ou, graph_ou, egraph_ou, ppe",
    )

    # ----- time points -----
    parser.add_argument(
        "--time_random_type",
        type=str,
        default="random",
        choices=["random", "uniform"],
        help="Type of time distribution: random or uniform",
    )
    parser.add_argument(
        "--time_step", type=int, default=20, help="Time step between points"
    )
    parser.add_argument(
        "--max_time_step", type=int, default=250, help="Maximum time step"
    )

    # ----- random graph -----
    parser.add_argument(
        "--num_node", type=int, default=2, help="Number of nodes in the random graph"
    )
    parser.add_argument(
        "--edge_prob",
        type=float,
        default=0.4,
        help="Probability of an edge between nodes",
    )

    # ----- ou process -----
    parser.add_argument(
        "--mean_rev_speed",
        type=float,
        default=0.02,
        help="Mean reversion speed parameter",
    )
    parser.add_argument(
        "--mean_rev_level",
        type=float,
        default=0.7,
        help="Mean reversion level parameter",
    )
    parser.add_argument("--vola", type=float, default=0.01, help="Volatility parameter")
    parser.add_argument("--gamma", type=float, default=0.75, help="Gamma parameter")
    parser.add_argument("--rho", type=float, default=2, help="Rho parameter")
    parser.add_argument("--omega", type=float, default=0.75, help="Omega parameter")

    # ----- hlr process -----
    parser.add_argument(
        "--theta",
        type=list,
        default=[1 / 4, 1 / 2, -1 / 3],
        help="List of theta parameters",
    )

    # ----- ppe process -----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the PPE process",
    )
    parser.add_argument(
        "--decay_rate", type=float, default=0.2, help="Decay rate for the PPE process"
    )

    # ----- save path -----
    parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/qb/work/mlcolab/hzhou52/kt/synthetic",
        help="Path to save results",
    )

    return parser


def save_as_unified_format(
    args: argparse.Namespace,
    path: str,
    times: np.ndarray,
    items: np.ndarray,
    adj: np.ndarray,
) -> None:
    """
    Save data in a unified format.

    This function takes various data components and saves them in a unified format.

    Args:
        args (argparse.Namespace): Command-line arguments.
        path (str): Path to save the data.
        times (numpy.ndarray): Array of timestamps.
        items (numpy.ndarray): Array of item IDs.
        adj (numpy.ndarray): Adjacency matrix.

    Returns:
        None

    """
    df = []
    timestamp = times.flatten()
    dwell_time = np.zeros_like(timestamp)

    correct = (path >= 0.5) * 1  # (sigmoid(p_items) >= 0.5)*1 # TODO???

    problem_id = items.flatten()
    skill_id = items.flatten()

    user_id = np.tile(
        np.arange(args.num_sequence).reshape(-1, 1), (1, args.time_step)
    ).flatten()

    df = np.stack([timestamp, dwell_time, correct, problem_id, skill_id, user_id], -1)
    df = pd.DataFrame(
        df,
        columns=[
            "timestamp",
            "dwell_time",
            "correct",
            "problem_id",
            "skill_id",
            "user_id",
        ],
    )

    df = df.astype(
        {
            "timestamp": np.float64,
            "dwell_time": np.float64,
            "correct": np.float64,
            "problem_id": np.int64,
            "skill_id": np.int64,
            "user_id": np.int64,
        }
    )

    # Save
    adj_path = os.path.join(args.log_path, "adj.npy")
    np.save(adj_path, adj)
    df_path = os.path.join(args.log_path, "interactions_{}.csv".format(args.time_step))
    df.to_csv(df_path, sep="\t", index=False)


def time_point_generate():
    """
    Generate random or uniform time points for interactions.

    This function generates time points for interactions based on the specified method (uniform or random).

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - time_random_type (str): Type of time point generation ('uniform' or 'random').
            - max_time_step (int): Maximum time step for generating time points.
            - time_step (int): Interval between two time points (used only if time_random_type is 'uniform').
            - num_sequence (int): Number of sequences.

    Returns:
        numpy.ndarray: Array containing time points for interactions.
    """
    if args.time_random_type == "uniform":
        times = np.arange(0, args.max_time_step, args.max_time_step // args.time_step)
        times = np.tile(
            np.expand_dims(times, 0), (args.num_sequence, 1)
        )  # [num_deq, time_step]

    elif args.time_random_type == "random":
        rng = default_rng(args.random_seed)
        times = []
        for i in range(args.num_sequence):
            time = rng.choice(np.arange(args.max_time_step), args.time_step, False)
            time.sort()
            times.append(time)
        times = np.stack(times)

    return times


def review_item_generate():
    """
    Generate review items for each sequence.

    This function generates review items for each sequence based on the provided path.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - random_seed (int): Seed for random number generation.
            - num_sequence (int): Number of sequences.
            - num_node (int): Total number of nodes (items).
            - time_step (int): Number of time steps in each sequence.
        path (numpy.ndarray): Array representing the path or sequence of items.

    Returns:
        numpy.ndarray: Array containing review items for each sequence.
    """

    rng = default_rng(args.random_seed)
    items = []
    # p_items = []
    for i in range(args.num_sequence):
        item = rng.choice(np.arange(args.num_node), args.time_step, True)
        items.append(item)
        # p_items.append(path[i, np.arange(args.time_step), item])
    items = np.stack(items)
    # p_items = np.stack(p_items).flatten()

    return items


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function for a given input.

    The sigmoid function is commonly used in machine learning for mapping input values to a range between 0 and 1.

    Args:
        x (numpy.ndarray): Input values.

    Returns:
        numpy.ndarray: Output values after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # ----- args -----
    parser = argparse.ArgumentParser(description="Global")
    parser = parse_args(parser)

    global args
    args, extras = parser.parse_known_args()

    args.time = datetime.datetime.now().isoformat()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.log_path = os.path.join(
        args.save_path, args.time + "_" + "node_" + str(args.num_node)
    )
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # -- generate random graphs
    graph = nx.erdos_renyi_graph(
        args.num_node, args.edge_prob, seed=args.random_seed, directed=True
    )
    adj = nx.adjacency_matrix(graph).toarray()
    # draw_graph(graph, args)

    # -- generate time points & reviewing items
    times = torch.tensor(time_point_generate(), device=args.device)
    items = torch.tensor(review_item_generate(), device=args.device)

    # -- Vanilla OU process
    x0 = torch.zeros((args.num_sequence, args.num_node), device=args.device)
    ou_vanilla_generator = VanillaOU(
        mean_rev_speed=args.mean_rev_speed,
        mean_rev_level=args.mean_rev_level,
        vola=args.vola,
        num_seq=args.num_sequence,
        mode="synthetic",
        device=args.device,
    )
    vanilla_path, params = ou_vanilla_generator.simulate_path(x0, times)
    draw_path(vanilla_path[0], args, times[0], items=items[0], prefix="ou")

    # -- HLR process
    weight_total = [1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3]
    weight_success = [1 / 2]
    weight_failure = [-1 / 3]
    x0 = torch.zeros((args.num_sequence, args.num_node), device=args.device)
    for w in weight_total:
        theta = [w] + weight_success + weight_failure
        theta = np.array(theta)
        hlr_generator = HLR(
            theta=theta, num_seq=args.num_sequence, mode="synthetic", device=args.device
        )
        hlr_path, params = hlr_generator.simulate_path(x0=x0, t=times, items=items)
        prefix = "hlr_theta_{:.2f}_{:.2f}_{:.2f}".format(theta[0], theta[1], theta[2])

        draw_path(hlr_path[0], args, times[0], items=items[0], prefix=prefix)
        # draw_params(params, args, times, items, prefix=prefix) TODO to test
        ipdb.set_trace()

    # -- PPE process
    for learning_rate in [0.01, 0.05, 0.1, 0.2, 0.5, 1]:
        ppe_generator = PPE(learning_rate, num_seq=args.num_sequence)
        ppe_path, params = ppe_generator.simulate_path(
            np.zeros((args.num_node, 1)), times, items
        )
        prefix = "ppe_lr_{}".format(learning_rate)

        draw_path(
            ppe_path[0], args, times[0], items=items[0], prefix=prefix
        )  # , scatter=True)
        draw_params(params, args, times, items, prefix=prefix)

    # -- Extend Graph OU process
    # -- Graph OU process
    # for speed in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
    for gamma in [0.1, 0.2, 0.5, 0.75, 1]:
        ou_generator = GraphOU(
            args.mean_rev_speed,
            args.mean_rev_level,
            args.vola,
            args.num_sequence,
            nx_graph=graph,
            gamma=gamma,
            rho=args.rho,
            omega=args.omega,
        )  # test
        ou_path, params = ou_generator.simulate_path(
            np.zeros((args.num_node, 1)), times, items=items
        )
        prefix = "GraphOU_speed_{}_vola_{}_gamma_{}_rho_{}_omega_{}".format(
            args.mean_rev_speed, args.vola, gamma, args.rho, args.omega
        )
        # args.mean_rev_speed

        draw_path(ou_path[0], args, times[0], items=items[0], prefix=prefix)
        draw_path(
            sigmoid(ou_path[0]),
            args,
            times[0],
            items=items[0],
            prefix=prefix + "sigmoid",
        )
        draw_params(params, args, times, items, prefix=prefix)

    save_as_unified_format(args, path, times, items, adj)

    # TODO debug: visualize
    draw_path(path[0], args, times[0], prefix="graph")
    draw_path(vanilla_path[0], args, times[0], prefix="vanilla")
    draw_path(sigmoid(path[0]), args, times[0], prefix="sigmoid")
    visualize_ground_truth(graph, args, adj)
