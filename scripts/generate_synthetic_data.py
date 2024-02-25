import sys

sys.path.append("..")

import argparse
import datetime
from pathlib import Path

import torch

from knowledge_tracing.baseline.halflife_regression import hlr as HLR
from knowledge_tracing.baseline.learner_model import ou as VanillaOU
from knowledge_tracing.baseline.learner_model import graph_ou as GraphOU
from knowledge_tracing.baseline.ppe import ppe as PPE
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
        choices=["hlr", "ppe", "ou", "graph_ou"],
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
    parser.add_argument("--rho", type=float, default=2, help="Rho parameter")
    parser.add_argument("--omega", type=float, default=0.75, help="Omega parameter")
    parser.add_argument(
        "--gamma", type=float, default=[0.1, 0.2, 0.5, 0.75, 1], help="Gamma parameter"
    )

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
        default=[0.01, 0.05, 0.1, 0.2, 0.5, 1],
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

    # -- generate time points & reviewing items
    times = torch.tensor(utils.generate_time_point(args), device=args.device)
    items = torch.tensor(utils.generate_review_item(args), device=args.device)

    # -- simulate learning processes
    if args.learner_model == "ou":
        ou_path, params = utils.simulate_ou_learning_path(
            args, times, items, model=VanillaOU, vis=True
        )
    elif args.learner_model == "graph_ou":
        # -- generate random graphs
        graph_adj = utils.generate_random_graph(args, vis=True)
        graph_ou_path, params = utils.simulate_ou_learning_path(
            args, times, items, graph_adj, model=GraphOU, vis=True
        )
    elif args.learner_model == "hlr":
        hlr_path, params = utils.simulate_hlr_learning_path(
            args, times, items, model=HLR, vis=True
        )
    elif args.learner_model == "ppe":
        ppe_path, params = utils.simulate_ppe_learning_path(
            args, times, items, model=PPE, vis=True
        )
    else:
        raise NotImplementedError
