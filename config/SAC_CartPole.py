from utils import Config
from models import CNN, MLP
from agents import SAC
"""
    Compared with QLearning, alpha instead of eps
"""
algo_args = Config()

algo_args.q_update_interval=4
algo_args.pi_update_interval=4
algo_args.n_warmup=int(2e3)
algo_args.replay_size=int(1e5)
algo_args.max_ep_len=500
algo_args.test_interval = int(1e4)
algo_args.seed=0
algo_args.batch_size=256
algo_args.save_interval=int(1e6)
algo_args.log_interval=int(2e3)
algo_args.n_step=int(1e8)

agent_args=Config()
agent_args.agent=SAC
agent_args.gamma=0.99
agent_args.alpha=0.2 
agent_args.target_sync_rate=algo_args.q_update_interval/1000

q_args=Config()
q_args.network = MLP
q_args.activation=torch.nn.ReLU
q_args.lr=2e-4
q_args.sizes = [4, 16, 32, 3] # 2 actions, dueling q learning

pi_args=Config()
pi_args.network = MLP
pi_args.activation=torch.nn.ReLU
pi_args.lr=2e-4
pi_args.sizes = [4, 16, 32, 2] 

args = Config()
args.env_name=env_name
args.name=f"{args.env_name}_{agent_args.agent}"
device = 0

q_args.env_fn = env_fn
agent_args.env_fn = env_fn
algo_args.env_fn = env_fn

agent_args.q_args = q_args
agent_args.pi_args = pi_args
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set