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
from models.variational_distributions import VarConstant, VarBasic, VarENCO, VarDIBS
from models.graph import InterventionalGraph

from collections import defaultdict
import ipdb

class CausalKT(BaseModel):
    extra_log_args = ['time_log']

    @staticmethod
    def parse_model_args(parser, model_name='CausalKT'):
        parser.add_argument('--tau_gumbel', type=float, default=0.5, help="Temperature for Gumbel softmax.")

        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--time_log', type=float, default=np.e, help='Log base of time intervals.')
        parser.add_argument('--time_lag', type=int, default=30, )

        parser.add_argument('--latent_rep', type=str, default='dibs', help='[constant, basic, enco, dibs]')  
        parser.add_argument('--latent_dim', type=int, default=64, help='embedding vector dimension for dibs')  
        parser.add_argument('--dense_init', type=int, default=0, help='whether the graph starts from sparse(0) or dense(1)')  
        parser.add_argument('--num_graph', type=int, default=1, help='the amount of graphs when sampling')  
        parser.add_argument('--adj_or_prob', type=str, default='adj')  


        parser.add_argument('--emb_history', type=int, help='for debug use! \
                                        0: label=1/-1 and multiply the emb history; \
                                        1: label=1/0, and concatenate after the emb vector.')


        parser.add_argument('--diagonal', type=int, default=1, help='for debug use! 0: wo_diagonal; 1: w_diagonal')      
        parser.add_argument('--problem_base', type=int, default=0, help='for debug use! 0: wo_problem_base; 1: w_problem_base')     

        
        return BaseModel.parse_model_args(parser, model_name)



    def __init__(self, args, corpus, logs):
        self.args = args
        
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.time_log = args.time_log
        self.device = args.device
        self.logs = logs

        self.tau_gumbel = args.tau_gumbel
        self.num_nodes = self.skill_num
        self.time_lag = args.time_lag

        self.norm_layer = LayerNorm
        self.res_connection = True
        self.encoder_layer_sizes = None
        self.decoder_layer_sizes = None # TODO
        
        self.dropout = args.dropout
        self.embedding_size = args.emb_size # TODO

        self.emb_history = args.emb_history
        
        self.diagonal = args.diagonal

        self.latent_rep = args.latent_rep
        self.dense_init = args.dense_init
        self.adj_or_prob = args.adj_or_prob

        super().__init__(model_path=os.path.join(args.log_path, 'Model_{}.pt'))
        
    
    def _init_weights(self):
        if self.latent_rep == 'dibs':
            self.var_dist_A = VarDIBS(
                device=self.device,
                num_nodes=self.num_nodes,
                dense_init=self.dense_init,
                latent_dim=self.args.latent_dim, 
                tau_gumbel=self.tau_gumbel,
            )
        elif self.latent_rep == 'enco': # TODO did not test
            self.var_dist_A = VarENCO(
                device=self.device,
                num_nodes=self.num_nodes,
                dense_init=self.dense_init,
                tau_gumbel=self.tau_gumbel,
            )
        elif self.latent_rep == 'basic': # TODO did not test
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
        ipdb.set_trace()
        self.loss_function = torch.nn.BCELoss()

        if self.latent_rep == 'dibs':
            latents = self.var_dist_A.latents
            self.alpha_skill_embeddings = torch.cat([latents[0], latents[1]], dim=-1).unsqueeze(0) # [1, num_nodes, num_emb]
            self.embedding_size = self.alpha_skill_embeddings.shape[-1]
        else:
            alpha_skill_embeddings = torch.randn(1, self.num_nodes, self.embedding_size, device=self.device)
            self.alpha_skill_embeddings = nn.Parameter(alpha_skill_embeddings, requires_grad=True)
        
        skill_base = torch.randn(1, self.num_nodes, device=self.device)
        self.skill_base = nn.Parameter(skill_base, requires_grad=True)
        # problem_base = torch.randn(1, self.problem_num, device=self.device)
        # self.problem_base = nn.Parameter(problem_base, requires_grad=True)

        # if self.dense_w == 0:
        #     self.W = nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes, device=self.device), requires_grad=True)
        # elif self.dense_w == 1:
        #     self.W = nn.Parameter(torch.ones(self.num_nodes, self.num_nodes, device=self.device), requires_grad=True)
        # elif self.dense_w == 2:
        #     self.W = torch.ones(self.num_nodes, self.num_nodes, device=self.device)


        a = max(self.embedding_size, 64)
        
        layers_f = self.decoder_layer_sizes or [a, a]
        in_dim_f = self.embedding_size
        layers_g = self.encoder_layer_sizes or [a, a]
        in_dim_g = self.embedding_size*2

        if self.emb_history == 1:

            self.g = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=self.embedding_size,
                hidden_dims=layers_g,
                p_dropout=self.dropout,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=2,
            hidden_dims=layers_f,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )

        self.interv_edges = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=self.embedding_size,
            hidden_dims=layers_g,
            p_dropout=self.dropout,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=self.norm_layer,
            res_connection=self.res_connection,
        )
        
        
    def get_weighted_adjacency(self):
        if self.diagonal == 0:
            W_adj = self.W * (1.0 - torch.eye(self.num_nodes, device=self.device))  # Shape (num_nodes, num_nodes)
        else: W_adj = self.W
        return W_adj

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        skill_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values
        problem_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values

        feed_dict = {
            'skill_seq': torch.from_numpy(utils.pad_lst(skill_seqs)),            # [batch_size, seq_len] # TODO isn't this -1?
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, seq_len]
            'problem_seq': torch.from_numpy(utils.pad_lst(problem_seqs)),        # [batch_size, seq_len]
            'time_seq': torch.from_numpy(utils.pad_lst(time_seqs)),              # [batch_size, seq_len]
        }
        return feed_dict


    def forward(self, feed_dict):
        
        skills = feed_dict['skill_seq']      # [batch_size, seq_len]
        problems = feed_dict['problem_seq']  # [batch_size, seq_len]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        labels = feed_dict['label_seq']      # [batch_size, seq_len]

        bs, time_steps = labels.shape
        time_lag = self.time_lag
        
        _, p_adj, b_adj = self.var_dist_A.sample_A(num_graph=self.args.num_graph)
        if self.adj_or_prob == 'adj':
            adj = b_adj*p_adj
            adj = adj.tile((1, bs, 1, 1))
        else:
            adj = p_adj.tile((1, bs, 1, 1)) # [num_graph, 1, num_nodes, num_nodes]
        
        # adj = self.var_dist_A
        # W_adj = adj * self.get_weighted_adjacency() 

        bs_skill_emb = torch.tile(self.alpha_skill_embeddings, (bs, 1, 1)).to(self.device)
        bs_skill_base = torch.tile(self.skill_base, (bs, 1)).to(self.device)
        # bs_problem_base = torch.tile(self.problem_base, (bs, 1)).to(self.device)

        if self.emb_history == 0:
            labels_pro = torch.where(labels==0, -2, labels)
            labels_pro = torch.where(labels_pro==-1, 0, labels_pro)
            labels_pro = torch.where(labels_pro==-2, -1, labels_pro)
        else: labels_pro = labels
        
        delta_t = (times[:, :, None] - times[:, None, :]).abs().double() # [bs, seq_len, seq_len] # symmetry ranging from 13.47e8-13.74e8
        delta_t = torch.log(delta_t + 1e-10) / torch.log(torch.tensor(self.time_log)) # ??? time_log # ranging from 0 to 17
        
        preds = []

        # TODO
        for t in range(time_lag, time_steps):
            # ipdb.set_trace()
            label_history = labels_pro[:, t - time_lag:t] # [bs, time_lag]
            skill_history = skills[:, t - time_lag:t] # [bs, time_lag]
        
            skill_target = skills[:, t:t+1] # [bs, 1]
            problem_target = problems[:, t:t+1]
            skill_base_target = bs_skill_base[torch.arange(0, bs), skill_target[:, 0]].unsqueeze(-1) # [bs, 1]
            # problem_base_target = bs_problem_base[torch.arange(0, bs), problem_target[:, 0]].unsqueeze(-1) # [bs, 1]

            delta_t_weight = delta_t.transpose(-1,-2)[:, t, t - time_lag:t] # [bs, time_lag]  range from 0 to around 10
            emb_history = [bs_skill_emb[torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            emb_history = torch.stack(emb_history, dim=1) # [bs, time_lag, emd_size]

            if self.emb_history == 0:
                emb_history *= label_history.unsqueeze(-1)
            elif self.emb_history == 1:
                label_history = torch.tile(label_history.unsqueeze(-1), (1, 1, self.embedding_size))
                emb_history = torch.cat([emb_history, label_history], dim=-1).double()
                emb_history = self.g(emb_history)

            # w_ij means the effect from i to j 
            # column j has all of the effects coming to j
            # so index in j-th row of w_transpose  
            graph_weight = adj.transpose(-1,-2)[:, torch.arange(0, bs), skill_target[:, 0]] # [num_graph, bs, num_nodes]
            cross_weight = [graph_weight[:, torch.arange(0, bs), skill_history[:, i]] for i in range(time_lag)]
            cross_weight = torch.stack(cross_weight, -1) # [bs, time_lag]
            ipdb.set_trace()
            emb_target = torch.tile(bs_skill_emb[torch.arange(0, bs), skill_target[:,0]].unsqueeze(1), (1, time_lag, 1))
            edge_feat = torch.cat([emb_target, emb_history], dim=-1).double() # [bs, time_lag, emb_size*2]

            msg = self.interv_edges(edge_feat) # [bs, time_lag, emb_size]

            # use a softmax here? TODO
            cross_effect = (cross_weight * torch.exp(-delta_t_weight)).unsqueeze(-1) * msg
            cross_effect = cross_effect.sum(-2) # [num_graph, bs, emb_size]

            # pred = (self.f(cross_effect) + problem_base_target + skill_base_target).sigmoid()
            # pred = (self.f(cross_effect)).sigmoid()
            pred = F.softmax(self.f(cross_effect), dim=-1)[..., 1:] # problem_base_target + skill_base_target
            preds.append(pred)

        preds = torch.cat(preds, dim=-1)
        if skills.is_cuda:
            preds = preds.cuda()
        labels = labels[:, self.time_lag:].unsqueeze(0).tile((preds.shape[0], 1, 1))

        return {
            'prediction': preds.double(),
            'label': labels.double(),
        }

    def loss(self, feed_dict, outdict, metrics=None):
        losses = defaultdict(lambda: torch.zeros((), device=self.device))
        prediction = outdict['prediction'].flatten()
        label = outdict['label'].flatten()
        mask = label > -1
        
        # # TODO for debugging
        # weight = (label==0)+1
        # loss = torch.nn.BCELoss(weight=weight.to(self.device))
        # losses['loss_pred'] = loss(prediction[mask], label[mask])
        losses['loss_pred'] = self.loss_function(prediction[mask], label[mask])
        
        _, _, adj = self.var_dist_A.sample_A(num_graph=self.args.num_graph)

        losses['loss_spasity'] = F.relu(1 * adj.sum()) * 1e-7

        losses['loss_total'] = losses['loss_spasity'] + losses['loss_pred']

        if metrics != None:
            pred = outdict['prediction'].detach().cpu().data.numpy()
            gt = outdict['label'].detach().cpu().data.numpy()
            evaluations = BaseModel.pred_evaluate_method(pred, gt, metrics)
        for key in evaluations.keys():
            losses[key] = evaluations[key]
        
        return losses