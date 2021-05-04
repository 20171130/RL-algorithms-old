args = Config()
args.env=env_name #discrete action
args.algorithm="sac"
args.name=f"{args.env}_{args.algorithm}"
args.gpu=0
args.seed=0
args.cpu=4

algo_args = Config()
algo_args.n_step=4096
algo_args.n_update=50
algo_args.batch_size=2048
algo_args.epochs=9999
algo_args.start_steps=20000
algo_args.update_after=20000

model_args=Config()
model_args.hidden_sizes=[256]*4
model_args.activation=torch.nn.ReLU
model_args = Config()
model_args.gamma=0.99
model_args.polyak=0.995
model_args.lr=3e-5
model_args.alpha=0.02
model_args.eps=0
model_args.dqn=False

args.algo_args = algo_args.toDict()
args.model_args = model_args.toDict()

run=wandb.init(
    project="RL",
    config=args,
    name=args.name,
    group=args.env,
)
logger = Logger(run)
env = gym.make(args.env)
model = core.MLPDQActorCritic(env.observation_space, env.action_space, logger=logger, **(model_args.toDict()))
device=0
model.to(device)
result =sac(lambda : gym.make(args.env), model=model, logger=logger,  device=device, **(algo_args.toDict()))
run.finish()