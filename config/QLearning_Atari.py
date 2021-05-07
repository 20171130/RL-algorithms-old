from utils import Config
from models import CNN
from agents import QLearning
"""
    update_interval, save_interval, etc are counted per sample
    "epoch" is only used for logging 
    
    the configs are the same as rainbow,
    batchsize *8, lr * 4, update frequency/ 8
    no noisy q and therefore eps of 3e-2
"""
algo_args = Config()

algo_args.max_ep_len=2000
algo_args.q_update_interval=32
algo_args.batch_size=256
algo_args.n_warmup=int(2e5)
algo_args.replay_size=int(1e6)
algo_args.test_interval = int(3e4)
algo_args.seed=0
algo_args.save_interval=int(1e6)
algo_args.log_interval=int(2e3)
algo_args.n_step=int(1e8)

agent_args=Config()
agent_args.agent=QLearning
agent_args.gamma=0.99
agent_args.eps=3e-2
agent_args.target_sync_rate=algo_args.q_update_interval/32000

q_args=Config()
q_args.network = CNN
q_args.activation=torch.nn.ReLU
q_args.lr=2e-4
q_args.strides = [2]*6
q_args.kernels = [3]*6
q_args.paddings = [1]*6
q_args.sizes = [4, 16, 32, 64, 128, 128, 5] # 4 actions, dueling q learning

args = Config()
args.env_name="Breakout-v0"
args.name=f"{args.env_name}_{agent_args.agent}"
device = 0

agent_args.q_args = q_args
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set