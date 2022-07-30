# -*- coding: UTF-8 -*-

import time, os
import numpy as np
from tqdm import tqdm
import networkx as nx
import joblib, json

import torch
import torch.nn.functional as F

from models.BaseModel import BaseModel
from utils import utils

import ipdb

class MyGRUCell(torch.nn.Module): # rnn.py
    def __init__(self, hidden_num):
        super(MyGRUCell, self).__init__()
        self.hidden_num = hidden_num
        self.i2h = torch.nn.Linear(3*hidden_num, 3*hidden_num)
        self.h2h = torch.nn.Linear(hidden_num, 3*hidden_num)
        self.reset_act = torch.nn.Sigmoid()
        self.update_act = torch.nn.Sigmoid()
        self.act = torch.nn.Tanh()

    def forward(self, inputs, states):
        prev_state_h = states

        i2h = self.i2h(inputs)

        h2h = self.h2h(prev_state_h)
        i2h_r, i2h_z, i2h = torch.split(i2h, self.hidden_num, dim=-1)
        h2h_r, h2h_z, h2h = torch.split(h2h, self.hidden_num, dim=-1)

        reset_gate = self.reset_act(i2h_r + h2h_r)
        update_gate = self.update_act(i2h_z + h2h_z)
        next_h_tmp = self.act(i2h + reset_gate * h2h)
        ones = torch.ones_like(update_gate)
        next_h = (ones - update_gate) * next_h_tmp + update_gate * prev_state_h

        return next_h, [next_h]


class Graph(object):
    def __init__(self, skill_num, directed_graphs, undirected_graphs):
        self.skill_num = skill_num
        self.directed_graphs = utils.as_list(directed_graphs)
        self.undirected_graphs = utils.as_list(undirected_graphs)

    @staticmethod
    def _info(graph: nx.Graph):
        return {"edges": len(graph.edges)}
    @property
    def info(self):
        return {
            "directed": [self._info(graph) for graph in self.directed_graphs],
            "undirected": [self._info(graph) for graph in self.undirected_graphs]
        }

    def neighbors(self, x, ordinal=True, merge_to_one=True, with_weight=False, excluded=None):
        excluded = set() if excluded is None else excluded

        if isinstance(x, torch.Tensor):
            x = x.numpy().tolist()
        if isinstance(x, list):
            return [self.neighbors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                if len(self.undirected_graphs) == 0:
                    return None if not merge_to_one else []
                elif len(self.undirected_graphs) == 1:
                    return [v for v in self.undirected_graphs[0].neighbors(int(x)) if v not in excluded]
                else:
                    if not merge_to_one:
                        return [[v for v in graph.neighbors(int(x)) if v not in excluded] for graph in
                                self.undirected_graphs]
                    else:
                        _ret = []
                        for graph in self.undirected_graphs:
                            _ret.extend([v for v in graph.neighbors(int(x)) if v not in excluded])
                        return _ret
            else:  # ordinal
                if not merge_to_one:
                    if len(self.undirected_graphs) == 0:
                        return None
                    elif len(self.undirected_graphs) == 1:
                        graph = self.undirected_graphs[0]
                        _ret = [0] * self.skill_num
                        for i in graph.neighbors(int(x)):
                            if i in excluded:
                                continue
                            if with_weight:
                                _ret[i] = graph[x][i].get('weight', 1)
                            else:
                                _ret[i] = 1
                        return _ret
                    else:
                        _ret = []
                        for graph in self.undirected_graphs:
                            __ret = [0] * self.skill_num
                            for i in graph.neighbors(int(x)):
                                if i in excluded:
                                    continue
                                if with_weight:
                                    __ret[i] = graph[x][i].get('weight', 1)
                                else:
                                    __ret[i] = 1
                            _ret.append(__ret)
                else:
                    if len(self.undirected_graphs) == 0:
                        return [0] * self.skill_num
                    else:
                        _ret = [0] * self.skill_num
                        for graph in self.undirected_graphs:
                            for i in graph.neighbors(int(x)):
                                if i in excluded:
                                    continue
                                if with_weight:
                                    _ret[i] += graph[x][i].get('weight', 1)
                                else:
                                    _ret[i] = 1
                        return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    def successors(self, x, ordinal=True, merge_to_one=True, excluded=None):
        excluded = set() if excluded is None else excluded
        if isinstance(x, torch.Tensor):
            x = x.numpy().tolist()
        if isinstance(x, list):
            return [self.successors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                if len(self.directed_graphs) == 0:
                    return None if not merge_to_one else []
                elif len(self.directed_graphs) == 1:
                    return [v for v in self.directed_graphs[0].successors(int(x)) if v not in excluded]
                else:
                    if not merge_to_one:
                        return [[v for v in graph.successors(int(x)) if v not in excluded] for graph in
                                self.directed_graphs]
                    else:
                        _ret = []
                        for graph in self.directed_graphs:
                            _ret.extend([v for v in graph.successors(int(x)) if v not in excluded])
                        return _ret
            else:
                if not merge_to_one:
                    if len(self.directed_graphs) == 0:
                        return None
                    elif len(self.directed_graphs) == 1:
                        _ret = [0] * self.skill_num
                        for i in self.directed_graphs[0].successors(int(x)):
                            if i in excluded:
                                continue
                            _ret[i] = 1
                        return _ret
                    else:
                        _ret = []
                        for graph in self.directed_graphs:
                            __ret = [0] * self.skill_num
                            for i in graph.successors(int(x)):
                                if i in excluded:
                                    continue
                                _ret[i] = 1
                            _ret.append(__ret)
                else:
                    if len(self.directed_graphs) == 0:
                        return [0] * self.skill_num
                    else:
                        _ret = [0] * self.skill_num
                        for graph in self.directed_graphs:
                            for i in graph.successors(int(x)):
                                if i in excluded:
                                    continue
                                _ret[i] = 1
                        return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    @classmethod
    def from_file(cls, graph_nodes_num, graph_params):
        directed_graphs = []
        undirected_graphs = []
        for graph_param in graph_params:
            # graph_path = os.path.join(args.path, args.dataset, graph_param)
            graph, directed = load_graph(graph_nodes_num, *utils.as_list(graph_param))
            if directed:
                directed_graphs.append(graph)
            else:
                undirected_graphs.append(graph)
        return cls(graph_nodes_num, directed_graphs, undirected_graphs)




class SKTF1(BaseModel):
    extra_log_args = ['graph_params']

    @staticmethod
    def parse_model_args(parser, model_name='SKTF1'):
        parser.add_argument('--graph_params', default=[['correct_transition_graph.json', True], ['transition_graph.json', False]])
        parser.add_argument('--hidden_num', default=16)
        parser.add_argument('--latent_dim', default=None)
        parser.add_argument('--concept_dim', default=None)
        parser.add_argument('--alpha', default=0.5)

        parser.add_argument('--activation', default=None)
        parser.add_argument('--sync_activation', default='relu')
        parser.add_argument('--prop_activation', default='relu')
        parser.add_argument('--agg_activation', default='relu')

        parser.add_argument('--skt_dropout', default=0.0)
        parser.add_argument('--self_dropout', default=0.5)
        parser.add_argument('--sync_dropout', default=0.0)
        parser.add_argument('--prop_dropout', default=0.0)
        parser.add_argument('--agg_dropout', default=0.0)

        parser.add_argument('--prefix', default=None) # TODO what is this
        parser.add_argument('--valid_length', default=None)
        parser.add_argument('--compressed_out', default=True)

        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus): # TODO super(SKT, self).__init__(prefix=prefix, params=params)
        self.dataset = args.dataset
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills) # =self.ku_num
        self.hidden_num = self.skill_num if args.hidden_num is None else int(args.hidden_num)
        self.latent_dim = self.hidden_num if args.latent_dim is None else int(args.latent_dim)
        self.concept_dim = self.hidden_num if args.concept_dim is None else int(args.concept_dim)
        graph_params = args.graph_params if args.graph_params is not None else []
        graph_path = []
        for graph in graph_params:
            graph_path.append([])
            graph_path[-1].append(os.path.join(args.path, args.dataset, 'graph', graph[0]))
            graph_path[-1].append(graph[1])

        self.graph = Graph.from_file(self.skill_num, graph_path) 

        self.alpha = args.alpha

        self.sync_activation = args.sync_activation if args.activation is None else args.activation
        self.prop_activation = args.prop_activation if args.activation is None else args.activation
        self.agg_activation = args.agg_activation if args.activation is None else args.activation

        self.self_dp = args.self_dropout
        self.dp = args.skt_dropout
        self.sync_dropout = args.sync_dropout
        self.prop_dropout = args.prop_dropout
        self.agg_dropout = args.agg_dropout
        
        #self.time_log = args.time_log
        self.gpu = args.gpu
        self.prefix = args.prefix
        self.valid_length = args.valid_length
        self.compressed_out = args.compressed_out

        super().__init__(model_path=args.model_path)

    def _init_weights(self):
        self.rnn = MyGRUCell(self.hidden_num).to(torch.float64)
        self.response_embedding = torch.nn.Embedding(2 * self.skill_num, self.latent_dim)
        self.concept_embedding = torch.nn.Embedding(self.skill_num, self.concept_dim) #TODO no gradient
        self.f_self = torch.nn.GRUCell(self.hidden_num, self.hidden_num).to(torch.float64)
        self.self_dropout = torch.nn.Dropout(self.self_dp)
        self.f_prop = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_num, self.hidden_num),
            torch.nn.ReLU(),
            # torch.nn.Activation(self.prop_activation), # TODO
            torch.nn.Dropout(self.prop_dropout),
        ).to(torch.float64)
        self.f_sync = torch.nn.Sequential(
            torch.nn.Linear(2*self.hidden_num, self.hidden_num),
            torch.nn.ReLU(),
            # torch.nn.Activation(self.sync_activation),
            torch.nn.Dropout(self.sync_dropout),
        ).to(torch.float64)
        self.f_agg = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_num, self.hidden_num),
            torch.nn.ReLU(), 
            # torch.nn.Activation(self.agg_activation),
            torch.nn.Dropout(self.agg_dropout),
        ).to(torch.float64)
        self.dropout = torch.nn.Dropout(self.dp)
        self.out = torch.nn.Linear(self.hidden_num, 1).to(torch.float64) # TODO loss function

        self.fin = torch.nn.Linear(3, self.hidden_num)

        self.loss_function = torch.nn.BCELoss()


    def neighbors(self, x, ordinal=True):
        return torch.as_tensor(self.graph.neighbors(x, ordinal))

    def successors(self, x, ordinal=True):
        return torch.as_tensor(self.graph.successors(x, ordinal))

    def begin_states(self, shapes, prefix, func=torch.zeros): # func=mx.nd.zeros # in rnn.py
        states = []
        for i, shape in enumerate(utils.as_list(shapes)):
            state = func(shape, dtype=torch.float64)
            # state = func(name='%sbegin_state_%d' % (prefix, i), shape=shape) # TODO: why they use this?
            states.append(state)
        return states

    def get_states(self, indexes, states): # in rnn.py
        tmp = []
        if isinstance(indexes, torch.Tensor):
            indexes = indexes.numpy().tolist()
        if isinstance(indexes, list):
            for (index, state) in zip(indexes, states): 
                tmp.append(self.get_states(index, state))
            return torch.stack(tmp).to(torch.float64)
        elif isinstance(indexes, (int, float)):
            return states[int(indexes)]
        else:
            raise TypeError("cannot handle %s" % type(indexes))
            
    def forward(self, feed_dict):
    # def forward(self, states=None, layout='NTC', compressed_out=True,
    #             *args, **kwargs):
    #     ctx = questions.context # cpu or gpu?

        problems = feed_dict['problem_seq']  # [batch_size, seq_len] #=questions 
        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        repeated_time_gap_seq = feed_dict['repeated_time_gap_seq']  # [batch_size, max_step]
        sequence_time_gap_seq = feed_dict['sequence_time_gap_seq']  # [batch_size, max_step]
        past_trial_counts_seq = feed_dict['past_trial_counts_seq']  # [batch_size, max_step]

        batch_size = problems.shape[0]
        length = problems.shape[1]
        
        # if states is None: #TODO
            # initialize the hidden state of concepts???
            # states = self.begin_states(shapes=[(batch_size, self.skill_num, self.hidden_num)], prefix=self.prefix)[0]
        states = self.begin_states(shapes=[(batch_size, self.skill_num, self.hidden_num)], prefix=self.prefix)[0]

        # states = states.as_in_context(ctx) TODO put states to gpu???
        outputs = []
        all_states = []

        for i in range(length-1):
            # paper III.B temporal effect
            # inputs[i]: (bs, ); question_id with batch size
            # self - influence
            inputs = skills[:, i] # (bs,)
            answers = labels[:, i] # (bs,)
            # - current state is indexed by the skill_id(inputs)
            _self_state = self.get_states(inputs, states)  # (bs, hidden_num)

            # - time embedding
            fin = self.fin(torch.cat((repeated_time_gap_seq[:,i], sequence_time_gap_seq[:,i], past_trial_counts_seq[:,i]), dim=-1))
            
            # - use current answer to update current states(from last step)
            answer_index = inputs * (answers+1)
            _next_self_state = self.f_self(self.response_embedding(answer_index), _self_state)  # TODO test answer_index
            _next_self_state = self.self_dropout(_next_self_state) # (bs, hidden_num)

            # get self mask
            # - give each skill a one-hot vector
            # https://stackoverflow.com/questions/55549843/pytorch-doesnt-support-one-hot-vector
            _self_mask = torch.zeros((batch_size, self.skill_num))
            _self_mask[range(batch_size), inputs] = 1
            _self_mask = _self_mask.unsqueeze(-1)  # (bs, ku_num, 1)
            _self_mask = torch.broadcast_to(_self_mask, (batch_size, self.skill_num, self.hidden_num))

            # find neighbors
            _neighbors = self.neighbors(inputs) # (bs, num_skills)
            _neighbors_mask = _neighbors.unsqueeze(-1) 
            _neighbors_mask = torch.broadcast_to(_neighbors_mask, (batch_size, self.skill_num, self.hidden_num))

            # synchronization
            _broadcast_next_self_states = _next_self_state.unsqueeze(1)
            _broadcast_next_self_states = torch.broadcast_to(_broadcast_next_self_states, (batch_size, self.skill_num, self.hidden_num))
            # - fuse [original states + answer fused states]
            _sync_diff = torch.cat((states, _broadcast_next_self_states), dim=-1)
            _sync_inf = _neighbors_mask * self.f_sync(_sync_diff)

            # reflection on current vertex
            _reflec_inf = torch.sum(_sync_inf, dim=1)
            _reflec_inf = torch.broadcast_to(_reflec_inf.unsqueeze(1), (batch_size, self.skill_num, self.hidden_num))
            _sync_inf = _sync_inf + _self_mask * _reflec_inf

            # find successors
            _successors = self.successors(inputs)
            _successors_mask = _successors.unsqueeze(-1)
            _successors_mask = torch.broadcast_to(_successors_mask, (batch_size, self.skill_num, self.hidden_num))

            # propagation
            _prop_diff = _next_self_state - _self_state

            # 1
            _prop_inf = self.f_prop(_prop_diff)
            _prop_inf = _successors_mask * torch.broadcast_to(_prop_inf.unsqueeze(1), (batch_size, self.skill_num, self.hidden_num))
            # concept embedding
            concept_embeddings = self.concept_embedding.weight.data
            concept_embeddings = torch.broadcast_to(concept_embeddings.unsqueeze(0), _prop_inf.shape)

            # aggregate
            fin = torch.broadcast_to(fin.unsqueeze(1), states.shape)
            _inf = self.f_agg(self.alpha * _sync_inf + (1 - self.alpha) * _prop_inf)
            next_states, _ = self.rnn(torch.cat((_inf, concept_embeddings, fin), dim=-1), states)
            states = next_states
            
            output = torch.sigmoid(torch.squeeze(self.out(self.dropout(states)), dim=-1))

            next_skill = skills[:, i+1]
            next_index = torch.cat((torch.arange(0, output.shape[0]).unsqueeze(-1), next_skill.unsqueeze(-1)), dim=-1)
            output = output[next_index[:,0], next_index[:,1]]
            outputs.append(output)
            if self.valid_length is not None and not self.compressed_out:
                all_states.append([states])

        outputs = torch.swapaxes(torch.stack(outputs), 0, 1)
        out_dict = {
            'prediction': outputs,
            'label': labels[:, 1:].double()
        }
        return out_dict

    def loss(self, feed_dict, outdict):
        prediction = outdict['prediction'].flatten()
        label = outdict['label'].flatten()
        mask = label > -1
        loss = self.loss_function(prediction[mask], label[mask])
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        skill_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values
        problem_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values

        user_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values

        sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq = \
            self.get_time_features(user_seqs, time_seqs)
        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)

        feed_dict = {
            'skill_seq': torch.from_numpy(utils.pad_lst(skill_seqs)),            # [batch_size, seq_len]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, seq_len]
            'problem_seq': torch.from_numpy(utils.pad_lst(problem_seqs)),        # [batch_size, seq_len]
            'time_seq': torch.from_numpy(utils.pad_lst(time_seqs)),              # [batch_size, seq_len]
            'repeated_time_gap_seq': torch.from_numpy(repeated_time_gap_seq[indice]),  # [batch_size, max_step]
            'sequence_time_gap_seq': torch.from_numpy(sequence_time_gap_seq[indice]),  # [batch_size, max_step]
            'past_trial_counts_seq': torch.from_numpy(past_trial_counts_seq[indice]),  # [batch_size, max_step]
            'length': torch.from_numpy(lengths[indice]),  # [batch_size]
            'inverse_indice': torch.from_numpy(inverse_indice),
            'indice': torch.from_numpy(indice),
        }
        return feed_dict

    @staticmethod
    def get_time_features(user_seqs, time_seqs):
        skill_max = max([max(i) for i in user_seqs])
        inner_max_len = max(map(len, user_seqs))
        repeated_time_gap_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.double)
        sequence_time_gap_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.double)
        past_trial_counts_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.double)
        for i in range(len(user_seqs)):
            last_time = None
            skill_last_time = [None for _ in range(skill_max)]
            skill_cnt = [0 for _ in range(skill_max)]
            for j in range(len(user_seqs[i])):
                sk = user_seqs[i][j] - 1
                ti = time_seqs[i][j]

                if skill_last_time[sk] is None:
                    repeated_time_gap_seq[i][j][0] = 0
                else:
                    repeated_time_gap_seq[i][j][0] = ti - skill_last_time[sk]
                skill_last_time[sk] = ti

                if last_time is None:
                    sequence_time_gap_seq[i][j][0] = 0
                else:
                    sequence_time_gap_seq[i][j][0] = (ti - last_time)
                last_time = ti

                past_trial_counts_seq[i][j][0] = (skill_cnt[sk])
                skill_cnt[sk] += 1

        repeated_time_gap_seq[repeated_time_gap_seq < 0] = 1
        sequence_time_gap_seq[sequence_time_gap_seq < 0] = 1
        repeated_time_gap_seq[repeated_time_gap_seq == 0] = 1e4
        sequence_time_gap_seq[sequence_time_gap_seq == 0] = 1e4
        past_trial_counts_seq += 1
        sequence_time_gap_seq *= 1.0 / 60
        repeated_time_gap_seq *= 1.0 / 60

        sequence_time_gap_seq = np.log(sequence_time_gap_seq)
        repeated_time_gap_seq = np.log(repeated_time_gap_seq)
        past_trial_counts_seq = np.log(past_trial_counts_seq)
        return sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq





# https://github.com/apache/incubator-mxnet/blob/9491442d147e51e883708ebdf3a2b4cbf5b5247a/python/mxnet/gluon/rnn/rnn_cell.py#L52
def format_sequence(length, inputs, layout, merge, in_layout=None):
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    batch_axis = layout.find('N')
    batch_size = 0
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, np.ndarray):
        batch_size = inputs.shape[batch_axis]
        if merge is False:
            assert length is None or length == inputs.shape[in_axis]
            inputs = _as_list(npx.slice_channel(inputs, axis=in_axis,
                                                num_outputs=inputs.shape[in_axis],
                                                squeeze_axis=1))
    else:
        assert isinstance(inputs, (list, tuple)), \
            "Only support MXNet numpy ndarray or list of MXNet numpy ndarrays as inputs"
        assert length is None or len(inputs) == length
        batch_size = inputs[0].shape[0]
        if merge is True:
            inputs = np.stack(inputs, axis=axis)
            in_axis = axis

    if isinstance(inputs, np.ndarray) and axis != in_axis:
        inputs = np.swapaxes(inputs, axis, in_axis)

    return inputs, axis, batch_size

def load_graph(graph_nodes_num, filename=None, directed: bool = True, threshold=0.0):
    directed = bool(directed)
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(range(graph_nodes_num))
    if threshold < 0.0:
        for i in range(graph_nodes_num):
            for j in range(graph_nodes_num):
                graph.add_edge(i, j)
    else:
        assert filename is not None
        with open(filename) as f:
            for data in json.load(f):
                pre, suc = data[0], data[1]
                if len(data) >= 3 and float(data[2]) < threshold:
                    continue
                elif len(data) >= 3:
                    weight = float(data[2])
                    graph.add_edge(pre, suc, weight=weight)
                    continue
                graph.add_edge(pre, suc)
    return graph, directed
