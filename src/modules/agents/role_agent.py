import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RoleAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RoleAgent, self).__init__()
        self.args = args

        # 共享参数
        self.traj_encoder1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.traj_encoder2 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.query_builder = nn.Linear(args.rnn_hidden_dim, args.embed_dim)
        self.q1 = nn.Linear(args.rnn_hidden_dim, args.hidden_dim)
        self.hyper_net = nn.Sequential(nn.Linear(args.embed_dim, args.hidden_dim * args.n_actions // 2), nn.ReLU(),
                                        nn.Linear(args.hidden_dim * args.n_actions // 2, args.hidden_dim * args.n_actions))
        # self.hyper_bias
    
    def init_hidden(self):
        return self.traj_encoder1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, role_embed):
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        traj_embed = self.traj_encoder2(F.relu(self.traj_encoder1(inputs)), h_in)   # h
        query = self.query_builder(traj_embed)
        role_weight = F.softmax(th.mm(query, role_embed.T), dim=-1)    # (bs * n_agents, n_roles)
        q = F.relu(self.q1(traj_embed))   # (bs * n_agents, hidden_dim)
        hyper_w = self.hyper_net(role_embed).view(self.args.n_roles, self.args.hidden_dim, self.args.n_actions)
        q = th.bmm(q.unsqueeze(0).expand(self.args.n_roles, -1, -1), hyper_w)   # (n_roles, bs * n_agents, n_actions)
        q = th.bmm(role_weight.unsqueeze(1), q.permute(1, 0, 2)).squeeze(1)   # (bs * n_agents, n_actions)
        return q, traj_embed, role_weight