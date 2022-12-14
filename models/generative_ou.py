from data.ou_process import *


class GenerativeOU():
    def __init__(self, mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph):
        super().__init__(mean_rev_speed, mean_rev_level, vola, num_seq, nx_graph)