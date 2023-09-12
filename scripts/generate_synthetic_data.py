import sys

sys.path.append("..")

import argparse
import datetime

from pathlib import Path
from scipy.special import expit
import numpy as np
import networkx as nx

import torch

from knowledge_tracing.baseline.halflife_regression import hlr as HLR
from knowledge_tracing.baseline.learner_model import ou as VanillaOU
from knowledge_tracing.baseline.learner_model import graph_ou as GraphOU
from knowledge_tracing.baseline.ppe import ppe as PPE

import knowledge_tracing.utils.visualize as visualize
import knowledge_tracing.utils.utils as utils


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
        default="..kt_data/synthetic",
        help="Path to save results",
    )

    return parser


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser(description="Global")
    parser = parse_args(parser)
    args, extras = parser.parse_known_args()

    # set up
    args.time = datetime.datetime.now().isoformat()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.log_path = Path(args.save_path, args.time + "_" + "node_" + str(args.num_node))
    Path(args.log_path).touch()

    # -- generate random graphs
    graph = nx.erdos_renyi_graph(
        args.num_node, args.edge_prob, seed=args.random_seed, directed=True
    )
    adj = nx.adjacency_matrix(graph).toarray()
    visualize.draw_graph(graph, args)

    # -- generate time points & reviewing items
    times = torch.tensor(utils.generate_time_point(args), device=args.device)
    items = torch.tensor(utils.generate_review_item(args), device=args.device)

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
    visualize.draw_path(vanilla_path[0], args, times[0], items=items[0], prefix="ou")

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

        visualize.draw_path(hlr_path[0], args, times[0], items=items[0], prefix=prefix)

    # -- PPE process
    for learning_rate in [0.01, 0.05, 0.1, 0.2, 0.5, 1]:
        ppe_generator = PPE(learning_rate, num_seq=args.num_sequence)
        ppe_path, params = ppe_generator.simulate_path(
            np.zeros((args.num_node, 1)), times, items
        )
        prefix = "ppe_lr_{}".format(learning_rate)

        visualize.draw_path(
            ppe_path[0], args, times[0], items=items[0], prefix=prefix
        )  # , scatter=True)
        visualize.draw_params(params, args, times, items, prefix=prefix)

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

        visualize.draw_path(ou_path[0], args, times[0], items=items[0], prefix=prefix)
        visualize.draw_path(
            expit(ou_path[0]),
            args,
            times[0],
            items=items[0],
            prefix=prefix + "sigmoid",
        )
        visualize.draw_params(params, args, times, items, prefix=prefix)

    path = ou_path  # Or other paths
    utils.save_as_unified_format(args, path, times, items, adj)

    visualize.draw_path(path[0], args, times[0], prefix="graph")
    visualize.draw_path(vanilla_path[0], args, times[0], prefix="vanilla")
    visualize.draw_path(expit(path[0]), args, times[0], prefix="sigmoid")
    visualize.visualize_ground_truth(graph, args, adj)
