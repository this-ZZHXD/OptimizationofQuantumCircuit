import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.data import Batch
from torch.distributions.categorical import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None,
                 masks=torch.tensor([]), device="cpu"):
        self.device = device
        self.masks = masks.to(self.device)
        if not len(self.masks) == 0:
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)


class ActorGNN(nn.Module):
    def __init__(self, args):
        super(ActorGNN, self).__init__()
        self.gat1 = geom_nn.GATv2Conv(args.c_in_p, args.hidden_width, edge_dim=args.edge_dim, add_self_loops=True)
        self.gat2 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim, add_self_loops=True)
        self.gat3 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim, add_self_loops=True)
        self.gat4 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim, add_self_loops=True)
        self.gat5 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim, add_self_loops=True)

        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, 1)

        self.active_func = nn.Tanh()

    def forward(self, s):
        """ Forward propagation """
        x, edge_index, edge_attr = s.x, s.edge_index, s.edge_attr
        self.gat1(x, edge_index, edge_attr)
        x = self.active_func(self.gat1(x, edge_index, edge_attr))
        x = self.active_func(self.gat2(x, edge_index, edge_attr))
        x = self.active_func(self.gat3(x, edge_index, edge_attr))
        x = self.active_func(self.gat4(x, edge_index, edge_attr))
        x = self.active_func(self.gat5(x, edge_index, edge_attr))

        x = self.active_func(self.fc1(x))
        x = self.active_func(self.fc2(x))
        return x


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        self.active_func = nn.Tanh()

        self.gat1 = geom_nn.GATv2Conv(args.c_in_critic, args.hidden_width, edge_dim=args.edge_dim_critic, add_self_loops=True)
        self.gat2 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim_critic, add_self_loops=True)
        self.gat3 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim_critic, add_self_loops=True)
        self.gat4 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim_critic, add_self_loops=True)
        self.gat5 = geom_nn.GATv2Conv(args.hidden_width, args.hidden_width, edge_dim=args.edge_dim_critic, add_self_loops=True)

        self.global_attention = geom_nn.GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(args.hidden_width, args.hidden_width),
                nn.Tanh(),
                nn.Linear(args.hidden_width, args.hidden_width),
                nn.Tanh(),
                nn.Linear(args.hidden_width, 1),
            ),
            nn=nn.Sequential(
                nn.Linear(args.hidden_width, args.hidden_width),
                nn.Tanh(),
                nn.Linear(args.hidden_width, args.hidden_width),
                nn.Tanh(),
            ),
        )

        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

    def forward(self, s, batch_ids):
        """ Forward propagation """
        x, edge_index, edge_attr = s.x, s.edge_index, s.edge_attr
        x = self.active_func(self.gat1(x, edge_index, edge_attr))
        x = self.active_func(self.gat2(x, edge_index, edge_attr))
        x = self.active_func(self.gat3(x, edge_index, edge_attr))
        x = self.active_func(self.gat4(x, edge_index, edge_attr))
        x = self.active_func(self.gat5(x, edge_index, edge_attr))

        x = self.global_attention(x, batch_ids)

        x = self.active_func(self.fc1(x))
        x = self.active_func(self.fc2(x))
        x = self.active_func(self.fc3(x))

        return x


class PPO:
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient

        self.max_num = 16

        self.actor = ActorGNN(args)  # get logits instead of probs
        self.critic = Critic(args)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)

    def choose_action(self, s, with_grad=False):
        """
        choose an action
        :param s: state (observation)
        :return: action identifier
        :param with_grad:
        """
        # Batch processing
        batch_size = s.batch.max().item() + 1
        a_logits = self.actor(s).to(self.device)  # logit
        batch = torch.tensor(s.batch)

        # data
        a_logits_masked = torch.zeros([batch_size, self.max_num]).to(self.device)
        action_mask = torch.zeros([batch_size, self.max_num], dtype=torch.bool).to(self.device)
        identifiers = torch.zeros([batch_size, self.max_num], dtype=torch.int).to(self.device)

        # Sample Action
        for batch_id in range(batch_size):
            all_nodes = s.identifiers[batch == batch_id].to(self.device)
            action_nodes = torch.where(torch.ne(all_nodes, -1))[0].to(self.device)  # real action(index)
            logits = a_logits[batch == batch_id][action_nodes]  # probs of action nodes
            a_logits_masked[batch_id, : logits.shape[0]] = logits.flatten()
            action_mask[batch_id, : logits.shape[0]] = torch.tensor([True] * logits.shape[0])
            identifiers[batch_id, : action_nodes.shape[0]] = all_nodes[action_nodes]

        # get Distribution of all candidate actions
        dist = CategoricalMasked(logits=a_logits_masked, masks=action_mask, device=self.device)
        a = dist.sample()  # node id in graph
        identifier = identifiers[torch.arange(batch_size), a]  # identifier of action
        a_logprob = dist.log_prob(a)
        entropy = dist.entropy()
        if with_grad:
            return a, a_logprob.view(-1, 1), entropy.view(-1, 1), identifier
        else:
            return (a.cpu().numpy()[0],
                    a_logprob.detach().cpu().numpy()[0],
                    entropy.detach().cpu().numpy()[0],
                    identifier.cpu().numpy()[0])

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data

        batch_s = Batch.from_data_list(s.tolist())
        from torch_geometric.data import Data
        data_list = []
        for item in s_:
            data = Data(
                x=item['x'],
                edge_index=item['edge_index'],
                edge_attr=item['edge_attr'],
                identifiers=item['identifiers']
            )
            data_list.append(data)
        try:
            batch_s_ = Batch.from_data_list(data_list)
        except Exception as e:
            print(f"Failed to create Batch object: {e}")
        else:
            print("Batch object created successfully.")
       
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(batch_s, batch_s.batch)
            vs_ = self.critic(batch_s_, batch_s_.batch)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5)).to(self.device)  # Trick:advantage normalization

        # Optimize policy for K epochs:
        scale_actor_loss, scale_critic_loss = 0, 0
        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                batch_sample = Batch.from_data_list(s[index].tolist())
                _, a_logprob_now, dist_entropy, _ = self.choose_action(batch_sample, with_grad=True)
                a_logprob_now = a_logprob_now.to(self.device)
                dist_entropy = dist_entropy.to(self.device)
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(batch_sample, batch_sample.batch)
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                # record loss
                scale_actor_loss += actor_loss.mean().item()
                scale_critic_loss += critic_loss.item()
                
        print("")

        self.lr_decay(total_steps)
        return scale_actor_loss / self.K_epochs, scale_critic_loss / self.K_epochs

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
