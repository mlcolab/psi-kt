import time, os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib


from models.BaseModel import BaseModel
from utils import utils

from models.modules import *
from models.variational_distributions import *
from models.graph_message_passing import * #GraphTargetMessage, GraphWholeMessage, GraphRecurrent, GraphPerformanceTrace, GraphOUProcess, GraphOUProcessDebug

from collections import defaultdict
import ipdb

# TODO: only whole_graph change the data shape to [bs, num_graph, ...]

class CausalKT(BaseModel):
    extra_log_args = ['time_log']

    @staticmethod
    def parse_model_args(parser, model_name='CausalKT'):
        parser.add_argument('--tau_gumbel', type=float, default=1.0, help="Temperature for Gumbel softmax.")

        parser.add_argument('--emb_size', type=int, default=2, help='Size of embedding vectors.')
        parser.add_argument('--time_log', type=float, default=np.e, help='Log base of time intervals.')
        parser.add_argument('--time_lag', type=int, default=30, )

        parser.add_argument('--latent_rep', type=str, default='dibs', help='[constant, basic, enco, dibs]')  
        parser.add_argument('--latent_dim', type=int, default=32, help='embedding vector dimension for dibs')  
        parser.add_argument('--num_graph', type=int, default=1, help='the amount of graphs when sampling')  
        

        parser.add_argument('--emb_history', type=int, help='for debug use! \
                                        0: label=1/-1 and multiply the emb history; \
                                        1: label=1/0, and concatenate after the emb vector.')
        parser.add_argument('--decoder_type', type=str, default='ou', help='[target_msg, whole_msg, rnn, perf_trace]')

        # for debugging
        parser.add_argument('--dense_init', type=int, default=0, help='whether the graph starts from sparse(0) or dense(1)')  
        parser.add_argument('--adj_or_prob', type=str, default='b_adj', help=['adj=b*p', 'b_adj', 'p_adj'])  
        parser.add_argument('--diagonal', type=int, default=1, help='for debug use! 0: wo_diagonal; 1: w_diagonal')   
        parser.add_argument('--skill_base', type=int, default=0, help='for debug use! 0: wo_problem_base; 1: w_problem_base')     
        parser.add_argument('--problem_base', type=int, default=0, help='for debug use! 0: wo_problem_base; 1: w_problem_base')     

        
        return BaseModel.parse_model_args(parser, model_name)


    def __init__(self, args, corpus, logs):
        self.args = args
        self.device = args.device
        self.logs = logs
        
        self.num_bias = int(corpus.n_problems)
        self.num_nodes = int(corpus.n_skills)

        self.emb_size = args.emb_size
        self.tau_gumbel = args.tau_gumbel
        self.time_lag = args.time_lag

        self.decoder_type = args.decoder_type

        self.diagonal = args.diagonal

        self.latent_rep = args.latent_rep
        self.dense_init = args.dense_init
        self.adj_or_prob = args.adj_or_prob

        # self.dev0 = dev0
        # self.dev1 = dev1

        super().__init__(model_path=os.path.join(args.log_path, 'Model_{}.pt'))
        
    
    def _init_weights(self):
        self.loss_function = torch.nn.BCELoss()

        # ----- initialize the latents over the graph -----
        if self.latent_rep == 'gt_fixedgraph': # -> load gt adj
            self.var_dist_A = VarGT(
                device=self.device,
                num_nodes=self.num_nodes,
                gt_adj_path=self.args.gt_adj_path,
            )
        
        if self.latent_rep == 'dibs':
            self.var_dist_A = VarDIBS(
                device=self.device,
                num_nodes=self.num_nodes,
                dense_init=self.dense_init,
                latent_dim=self.args.latent_dim, 
                tau_gumbel=self.tau_gumbel,
            )
        elif self.latent_rep == 'enco': 
            self.var_dist_A = VarENCO(
                device=self.device,
                num_nodes=self.num_nodes,
                dense_init=self.dense_init,
                tau_gumbel=self.tau_gumbel,
            )
        elif self.latent_rep == 'basic':
            self.var_dist_A = VarBasic(
                device=self.device,
                num_nodes=self.num_nodes,
                dense_init=self.dense_init,
                tau_gumbel=self.tau_gumbel,
            )
        elif self.latent_rep == 'constant':
            self.var_dist_A = VarConstant(
                device=self.device,
                num_nodes=self.num_nodes,
            )
        # --------------------------------------------------

        # ----- initialize the node embeddings -----
        if self.latent_rep == 'dibs':
            u, v = self.var_dist_A.u, self.var_dist_A.v
            self.node_embeddings = torch.cat([u, v], dim=-1).unsqueeze(0) # [1, num_nodes, num_emb]
            self.emb_size = self.node_embeddings.shape[-1]
            # ipdb.set_trace()
        else:
            node_embeddings = torch.randn(1, self.num_nodes, self.emb_size, device=self.device)
            self.node_embeddings = nn.Parameter(node_embeddings, requires_grad=True)

        # node_base = torch.randn(1, self.num_nodes, device=self.device)
        # self.node_base = nn.Parameter(node_base, requires_grad=True)
        # node_bias_base = torch.randn(1, self.num_bias, device=self.device)
        # self.node_bias_base = nn.Parameter(node_bias_base, requires_grad=True) # from the problem
        # -------------------------------------------

        if self.decoder_type == 'target_msg':
            self.decoder = GraphTargetMessage(self.device, self.num_nodes, self.args, self.emb_size)
        elif self.decoder_type == 'whole_msg':
            self.decoder = GraphWholeMessage(self.device, self.num_nodes, self.args, self.emb_size)
        elif self.decoder_type == 'rnn':
            self.decoder = GraphRecurrent(self.device, self.num_nodes, self.args, self.emb_size, self.dev0, self.dev1)
        elif self.decoder_type == 'perf_trace':
            self.decoder = GraphPerformanceTrace(self.device, self.num_nodes, self.args, self.emb_size)
        elif self.decoder_type == 'ou':
            self.decoder = GraphOUProcess(self.device, self.num_nodes, self.args, self.emb_size)
        elif self.decoder_type == 'oudebug':
            self.decoder = GraphOUProcessDebug(self.device, self.num_nodes, self.args)
        elif self.decoder_type == 'graphou_fixedgraph':
            self.decoder = GraphOU_FixedGraph(self.device, self.num_nodes, self.args)
        
    def get_weighted_adjacency(self):
        if self.diagonal == 0:
            W_adj = self.W * (1.0 - torch.eye(self.num_nodes, device=self.device))  # Shape (num_nodes, num_nodes)
        else: W_adj = self.W
        return W_adj


    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        '''
        Prepare the batch data and add model-specific data features
        Dont need to move the data to device in this step (the function is implemented in the BaseModel)
        Args:
            corpus:       DataReader instance containing all of the data information
            data:         specific data to be batched (train/test/dev)
            batch_start:  the index of current batch
            batch_size:   size of batch
        Return:
            feed_dict:    a dictionary containing all of the data needed for training the model
        '''
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        
        skill_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values
        problem_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values
        num_history = data['num_history'][batch_start: batch_start + real_batch_size].values
        num_success = data['num_success'][batch_start: batch_start + real_batch_size].values
        num_failure = data['num_failure'][batch_start: batch_start + real_batch_size].values

        gt_adj = corpus.adj
        feed_dict = {
            'skill_seq': torch.from_numpy(utils.pad_lst(skill_seqs)),            # [batch_size, seq_len] # TODO isn't this -1?
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, seq_len]
            'problem_seq': torch.from_numpy(utils.pad_lst(problem_seqs)),        # [batch_size, seq_len]
            'time_seq': torch.from_numpy(utils.pad_lst(time_seqs)),              # [batch_size, seq_len]
            'gt_adj': torch.from_numpy(gt_adj),
            'num_history': torch.from_numpy(utils.pad_lst(num_history)), 
            'num_success': torch.from_numpy(utils.pad_lst(num_success)), 
            'num_failure': torch.from_numpy(utils.pad_lst(num_failure)), 
        }
        
        return feed_dict


    def forward(self, feed_dict):
        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        bs, _ = labels.shape
        
        _, p_adj, b_adj = self.var_dist_A.sample_A(num_graph=self.args.num_graph)
        
        if self.adj_or_prob == 'b_adj':
            adj = b_adj.tile((1, bs, 1, 1))
        elif self.adj_or_prob == 'adj':
            adj = b_adj*p_adj
            adj = adj.tile((1, bs, 1, 1))
        else:
            adj = p_adj.tile((1, bs, 1, 1)) # [num_graph, 1, num_nodes, num_nodes]
            
        
        # adj = feed_dict['gt_adj'].reshape(1,1,5,5).tile(self.args.num_graph,bs,1,1,)
        # adj = self.var_dist_A
        # W_adj = adj * self.get_weighted_adjacency() 
        preds = self.decoder(feed_dict, adj=adj, node_embeddings=self.node_embeddings)
        
        # ipdb.set_trace()
        preds = torch.stack(preds, dim=-1)# [:,:,0,0] ???
        if skills.is_cuda:
            preds = preds.cuda()
        labels = labels[:, self.args.time_lag:]#.unsqueeze(1).tile((1, preds.shape[1], 1)) # TODO in real data should be labels[:, 1:]

        return {
            'prediction': preds.double(),
            'label': labels.double(),
        }


    def loss(self, feed_dict, outdict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        prediction = outdict['prediction'].flatten()
        label = outdict['label'].flatten()
        
        mask = label > -1
        _, _, adj = self.var_dist_A.sample_A(num_graph=self.args.num_graph)
        # ipdb.set_trace()
        # # TODO for debugging
        # weight = (label==0)+1
        # loss = torch.nn.BCELoss(weight=weight.to(self.device))
        # losses['loss_pred'] = loss(prediction[mask], label[mask])
        losses['loss_pred'] = self.loss_function(prediction[mask], label[mask])
        losses['loss_spasity'] = F.relu(1 * adj.sum() / self.args.num_graph) * 1e-7 #
        
        losses['loss_total'] = losses['loss_pred'] + losses['loss_spasity']

        # ipdb.set_trace()
        # TODO DEBUG
        losses['speed'] = torch.mean(self.decoder.learner_model.mean_rev_speed)
        losses['level'] = torch.mean(self.decoder.learner_model.mean_rev_level)
        losses['vola'] = torch.mean(self.decoder.learner_model.vola)

        if metrics != None:
            pred = outdict['prediction'].detach().cpu().data.numpy()
            gt = outdict['label'].detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
        
        return losses