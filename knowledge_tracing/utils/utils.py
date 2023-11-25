import os, pickle, datetime, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import default_rng
from scipy.special import expit

import torch

from knowledge_tracing.utils import visualize


def get_theta_shape(num_seq: int, num_node: int, other) -> dict:
    """
    Get the shape of theta parameters based on the training mode.

    This function returns a dictionary containing the shape of theta parameters based on the
    number of sequences, number of nodes, and other parameters.

    Args:
        num_seq (int): The number of sequences.
        num_node (int): The number of nodes.
        other: Other parameters or arguments.

    Returns:
        dict: A dictionary with keys representing different training modes and values
              representing the shape of theta parameters.
    """
    return dict(
        simple_split_time=(1, 1, other),
        simple_split_learner=(1, 1, other),
        ls_split_time=(num_seq, 1, other),
        ns_split_time=(1, num_node, other),
        ns_split_learner=(1, num_node, other),
        ln_split_time=(num_seq, num_node, other),
    )


def load_corpus(logs, args):
    """
    Load corpus from the corpus path, and split the data into k folds.

    Args:
        logs: An object to write logs to.
        args: An object that contains command-line arguments.

    Returns:
        The corpus object that contains the loaded data.
    """

    # Construct the path to the corpus file based on the dataset and max_step arguments.
    corpus_path = Path(
        args.data_dir, args.dataset, "Corpus_{}.pkl".format(args.max_step)
    )
    logs.write_to_log_file(f"Load corpus from {corpus_path}")

    # Load the corpus object from the pickle file at the specified path.
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    # Check the value of the train_mode argument to determine the type of data split.
    if "split_learner" in args.train_mode:
        corpus.gen_fold_data(args.fold)
        logs.write_to_log_file("# Training mode splits LEARNER")

    elif "split_time" in args.train_mode:
        corpus.gen_time_split_data(
            args.train_time_ratio,
            args.test_time_ratio,
            args.val_time_ratio * args.validate,
        )
        logs.write_to_log_file("# Training mode splits TIME")

    logs.write_to_log_file(
        "# Train: {}, # val: {}, # Test: {}".format(
            len(corpus.data_df["train"]),
            len(corpus.data_df["val"]),
            len(corpus.data_df["test"]),
        )
    )

    return corpus


def get_feed_general(
    keys: dict, data: pd.DataFrame, start: int, batch_size: int, pad_list: bool = False
) -> dict:
    """
    Creates a PyTorch feed_dict for a batch of data.

    Args:
        keys: A dictionary mapping keys to their corresponding column names in the input data.
        data: A pandas DataFrame containing the input data.
        start: The starting index of the batch.
        batch_size: The size of the batch.
        pad_list: A boolean indicating whether to pad the sequences in the input data.

    Returns:
        A dictionary containing the input data as PyTorch tensors.
    """

    # Create an empty dictionary to hold the feed_dict values
    feed_dict = {}

    # Iterate over the keys in the provided list
    for key, value in keys.items():
        # Extract the sequence of values for the current key from the input data
        seq = data[value][start : start + batch_size].values

        # If the key ends in '_seq' and the pad_list flag is True, pad the sequence
        if "_seq" in key or "num_" in key:
            seq = pad_lst(seq)

        # Convert the sequence to a PyTorch tensor and add it to the feed_dict dictionary
        feed_dict[key] = torch.as_tensor(seq)

    return feed_dict


def get_feed_continual(
    keys: dict, data: pd.DataFrame, idx: int, pad_list: bool = False
) -> dict:
    """
    Create a feed dictionary for a given index from a DataFrame (for continual learning tasks).

    Args:
        keys (dict): A dictionary where keys are target variable names and values
                     are column names in the DataFrame `data` containing the
                     corresponding sequence data.
        data (pd.DataFrame): The DataFrame containing the sequence data.
        idx (int): The index up to which the sequences should be extracted.
        pad_list (bool, optional): Whether to pad sequences if their keys end with
                                   '_seq' or contain 'num_'. Default is False.

    Returns:
        dict: A dictionary with target variable names as keys and PyTorch tensors
              containing sequences of data up to the specified index as values.

    Note:
        - If `pad_list` is True, sequences whose keys end with '_seq' or contain
          'num_' will be padded to the length of the longest sequence in the batch.
    """
    # Create an empty dictionary to hold the feed_dict values
    feed_dict = {}

    # Iterate over the keys in the provided list
    for key, value in keys.items():
        # Extract the sequence of values for the current key from the input data
        if "_seq" in key or 'num_' in key:
            seq = data[value][:, : idx + 1]
        else:
            seq = data[value]

        # Convert the sequence to a PyTorch tensor and add it to the feed_dict dictionary
        feed_dict[key] = seq

    return feed_dict


def format_arg_str(
    args: argparse.Namespace,
    exclude_lst: list = ["device", "log_path", "log_file", "log_file_name"],
    max_len: int = 20,
) -> str:
    """
    Format the command-line arguments as a string.
    Args:
        args: The command-line arguments.
        exclude_lst: A list of arguments to exclude from the formatted string.
        max_len: The maximum length of the formatted string.
    Returns:
        A string containing the formatted command-line arguments.
    """
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = "Arguments", "Values"
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = (
        max([len(key_title), key_max_len]),
        max([len(value_title), value_max_len]),
    )
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + "=" * horizon_len + linesep
    res_str += (
        " "
        + key_title
        + " " * (key_max_len - len(key_title))
        + " | "
        + value_title
        + " " * (value_max_len - len(value_title))
        + " "
        + linesep
        + "=" * horizon_len
        + linesep
    )
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace("\t", "\\t")
            value = value[: max_len - 3] + "..." if len(value) > max_len else value
            res_str += (
                " "
                + key
                + " " * (key_max_len - len(key))
                + " | "
                + value
                + " " * (value_max_len - len(value))
                + linesep
            )
    res_str += "=" * horizon_len
    return res_str


def format_metric(metric: dict) -> str:
    """
    Format the metric as a string.
    Args:
        metric: The metric to format.
    Returns:
        A string containing the formatted metric.
    """
    assert type(metric) == dict
    format_str = []
    for name in np.sort(list(metric.keys())):
        m = metric[name]
        if (
            type(m) is float
            or type(m) is np.float
            or type(m) is np.float32
            or type(m) is np.float64
        ):
            format_str.append("{}:{:<.4f}".format(name, m))
        elif (
            type(m) is int
            or type(m) is np.int
            or type(m) is np.int32
            or type(m) is np.int64
        ):
            format_str.append("{}:{}".format(name, m))
    return ",".join(format_str)


def check_dir(file_name: str) -> None:
    """
    Checks if the directory containing the specified file exists, and creates it if necessary.

    Args:
        file_name: The name of the file to check.

    Returns:
        None
    """

    # Get the path to the directory containing the specified file.
    dir_path = Path(file_name).parents[0]
    dir_path.touch()


def get_time():
    """
    Get the current time.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def strictly_increasing(l: list) -> bool:
    """
    Check if the elements in the list are strictly increasing.
    """
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l: list) -> bool:
    """
    Check if the elements in the list are strictly decreasing.
    """
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l: list) -> bool:
    """
    Check if the elements in the list are non-increasing.
    """
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l: list) -> bool:
    """
    Check if the elements in the list are non-decreasing.
    """
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l: list) -> bool:
    """
    Check if the elements in the list are monotonic.
    """
    return non_increasing(l) or non_decreasing(l)


def as_list(obj) -> list:
    """
    Convert the input object to a list.
    https://github.com/tswsxk/longling/blob/2cc45688e183b3395ada129ec54db7bd00959cbb/longling/lib/candylib.py#L17
    """
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def create_rel_rec_send(num_atoms: int, device: torch.device) -> tuple:
    """
    Based on https://github.com/ethanfetaya/NRI (MIT License).
    """

    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_atoms, num_atoms])
    rel_rec = np.array(np.where(off_diag)[0])
    rel_send = np.array(np.where(off_diag)[1])
    rel_rec = torch.tensor(rel_rec).to(device)
    rel_send = torch.tensor(rel_send).to(device)

    return rel_rec, rel_send


def distribute_over_GPUs(
    args: argparse.Namespace, model: torch.nn.Module, num_GPU: int = None
) -> tuple:
    """
    Distribute the model over multiple GPUs.
    Args:
        args: An object that contains command-line arguments.
        model: The model to distribute over multiple GPUs.
        num_GPU: The number of GPUs to use.
    Returns:
        A tuple containing the distributed model and the number of GPUs used.
    """
    ## distribute over GPUs
    # if torch.device("cpu") not in args.device:
    if args.device.type != "cpu":
        if num_GPU is None:
            model = torch.nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            args.batch_size_multiGPU = args.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = torch.nn.DataParallel(model, device_ids=list(range(num_GPU)))
            args.batch_size_multiGPU = args.batch_size * num_GPU
    else:
        model = torch.nn.DataParallel(model)
        args.batch_size_multiGPU = args.batch_size

    model = model.to(args.device)

    return model, num_GPU


def pad_lst(lst: list, value: int = 0, dtype: type = np.int64) -> np.ndarray:
    """
    Pad a list of lists with a specified value.
    Args:
        lst: The list of lists to pad.
        value: The value to pad the lists with.
        dtype: The data type of the padded lists.
    Returns:
        A numpy array containing the padded lists.
    """
    # Find the maximum length of any row in the input list
    inner_max_len = max(map(len, lst))

    # Create a new array with the same number of rows as the input list
    # and the maximum row length as the number of columns
    result = np.ones([len(lst), inner_max_len], dtype) * value

    # Iterate over each row of the input list
    for i, row in enumerate(lst):
        # Iterate over each element in the row
        for j, val in enumerate(row):
            # Copy the element value to the corresponding position in the new array
            result[i][j] = val

    return result


def save_as_unified_format(
    args: argparse.Namespace,
    path: str,
    params: dict,
    times: np.ndarray,
    items: np.ndarray,
    adj: np.ndarray,
    prefix: str = "",
    vis: bool = True,
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

    correct = (path >= 0.5) * 1

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
    adj_path = Path(args.log_path, prefix, "adj.npy")
    np.save(adj_path, adj)
    df_path = Path(args.log_path, "interactions_{}.csv".format(args.time_step))
    df.to_csv(df_path, sep="\t", index=False)

    # Visualization
    if vis:
        visualize.draw_path(path[0], args, times[0], items=items[0], prefix=prefix)
        visualize.draw_path(expit(path[0]), args, times[0], prefix="sigmoid" + prefix)
        visualize.draw_params(params, args, times, items, prefix=prefix)


def generate_time_point(args: argparse.Namespace) -> np.ndarray:
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
        for _ in range(args.num_sequence):
            time = rng.choice(np.arange(args.max_time_step), args.time_step, False)
            time.sort()
            times.append(time)
        times = np.stack(times)

    return times


def generate_review_item(args: argparse.Namespace) -> np.ndarray:
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

    for _ in range(args.num_sequence):
        item = rng.choice(np.arange(args.num_node), args.time_step, True)
        items.append(item)
    items = np.stack(items)

    return items


def generate_random_graph(args: argparse.Namespace, vis: bool = True) -> np.ndarray:
    """
    Generate a random graph.
    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - num_node (int): Number of nodes in the random graph.
            - edge_prob (float): Probability of an edge between two nodes.
            - random_seed (int): Seed for random number generation.
        vis (bool): A boolean indicating whether to visualize the graph.
    Returns:
        numpy.ndarray: Adjacency matrix of the generated graph.
    """

    graph = nx.erdos_renyi_graph(
        args.num_node, args.edge_prob, seed=args.random_seed, directed=True
    )
    adj = nx.adjacency_matrix(graph).toarray()
    if vis:
        visualize.visualize_ground_truth(graph, args, adj)


def simulate_ou_learning_path(
    args: argparse.Namespace,
    times: np.ndarray,
    items: np.ndarray,
    model: torch.nn.Module,
    vis: bool = True,
) -> np.ndarray:
    """
    Simulate a learning path using the OU model.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - mean_rev_speed (float): Mean reversion speed.
            - mean_rev_level (float): Mean reversion level.
            - vola (float): Volatility.
            - num_sequence (int): Number of sequences.
            - num_node (int): Number of nodes.
            - device (torch.device): Device to use for computation.
        times (numpy.ndarray): Array containing time points for interactions.
        items (numpy.ndarray): Array containing review items for each sequence.

    Returns:
        numpy.ndarray: Array containing the simulated learning path.
    """

    # Initialize the initial state
    x0 = torch.zeros((args.num_sequence, args.num_node), device=args.device)

    # Create an OU model generator
    ou_vanilla_generator = model(
        mean_rev_speed=args.mean_rev_speed,
        mean_rev_level=args.mean_rev_level,
        vola=args.vola,
        num_seq=args.num_sequence,
        mode="synthetic",
        device=args.device,
    )

    # Simulate the learning path
    vanilla_path, params = ou_vanilla_generator.simulate_path(x0, times)

    # Save the simulated path
    save_as_unified_format(
        args, vanilla_path, times, items, adj=None, prefix="vanilla_ou", vis=vis
    )

    return vanilla_path, params


def simulate_graph_ou_learning_path(
    args: argparse.Namespace,
    times: np.ndarray,
    items: np.ndarray,
    graph_adj: np.ndarray,
    model: torch.nn.Module,
    vis: bool = True,
) -> np.ndarray:
    """
    Simulate a learning path using the graph OU model.
    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - mean_rev_speed (float): Mean reversion speed.
            - mean_rev_level (float): Mean reversion level.
            - vola (float): Volatility.
            - num_sequence (int): Number of sequences.
            - num_node (int): Number of nodes.
            - device (torch.device): Device to use for computation.
        times (numpy.ndarray): Array containing time points for interactions.
        items (numpy.ndarray): Array containing review items for each sequence.
        graph_adj (numpy.ndarray): Adjacency matrix of the graph.
    Returns:
        numpy.ndarray: Array containing the simulated learning path.
    """
    for gamma in args.gamma:
        graph_ou_generator = model(
            args.mean_rev_speed,
            args.mean_rev_level,
            args.vola,
            args.num_sequence,
            nx_graph=graph_adj,
            gamma=gamma,
            rho=args.rho,
            omega=args.omega,
        )
        graph_ou_path, params = graph_ou_generator.simulate_path(
            np.zeros((args.num_node, 1)), times, items=items
        )
        prefix = "GraphOU_speed_{}_vola_{}_gamma_{}_rho_{}_omega_{}".format(
            args.mean_rev_speed, args.vola, gamma, args.rho, args.omega
        )

        save_as_unified_format(
            args,
            graph_ou_path,
            params,
            times,
            items,
            adj=graph_adj,
            prefix=prefix,
            vis=vis,
        )

    return graph_ou_path, params


def simulate_hlr_learning_path(
    args: argparse.Namespace,
    times: np.ndarray,
    items: np.ndarray,
    model: torch.nn.Module,
    vis: bool = True,
) -> np.ndarray:
    """
    Simulate a learning path using the HLR (Hierarchical Linear Regression) model.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - num_sequence (int): Number of sequences.
            - num_node (int): Number of nodes.
            - device (torch.device): Device to use for computation.
            - theta (list): List of theta parameters.
        times (numpy.ndarray): Array containing time points for interactions.
        items (numpy.ndarray): Array containing review items for each sequence.
        model (torch.nn.Module): The HLR model to use for simulation.
        vis (bool, optional): Whether to visualize the results. Defaults to True.

    Returns:
        numpy.ndarray: Array containing the simulated learning path.
    """

    # Initialize the initial state
    x0 = torch.zeros((args.num_sequence, args.num_node), device=args.device)
    theta = np.array(args.theta)

    # Create an HLR model generator
    hlr_generator = model(
        theta=theta, num_seq=args.num_sequence, mode="synthetic", device=args.device
    )

    # Simulate the learning path
    hlr_path, params = hlr_generator.simulate_path(x0=x0, t=times, items=items)

    # Create a prefix for saving
    prefix = "hlr_theta_{:.2f}_{:.2f}_{:.2f}".format(theta[0], theta[1], theta[2])

    # Save the simulated path
    save_as_unified_format(
        args, hlr_path, params, times, items, adj=None, prefix=prefix, vis=vis
    )


def simulate_ppe_learning_path(
    args: argparse.Namespace,
    times: np.ndarray,
    items: np.ndarray,
    model: torch.nn.Module,
    vis: bool = True,
) -> np.ndarray:
    """
    Simulate a learning path using the PPE (Power Parameter Estimation) process.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - num_sequence (int): Number of sequences.
            - num_node (int): Number of nodes.
            - learning_rate (list): List of learning rates to simulate.
        times (numpy.ndarray): Array containing time points for interactions.
        items (numpy.ndarray): Array containing review items for each sequence.
        model (torch.nn.Module): The PPE model to use for simulation.
        vis (bool, optional): Whether to visualize the results. Defaults to True.

    Returns:
        numpy.ndarray: Array containing the simulated learning path.
    """

    # Simulate for each learning rate
    for learning_rate in args.learning_rate:
        ppe_generator = model(learning_rate, num_seq=args.num_sequence)
        ppe_path, params = ppe_generator.simulate_path(
            np.zeros((args.num_node, 1)), times, items
        )

        # Create a prefix for saving
        prefix = "ppe_lr_{}".format(learning_rate)

        # Save the simulated path
        save_as_unified_format(
            args, ppe_path, params, times, items, adj=None, prefix=prefix, vis=vis
        )
