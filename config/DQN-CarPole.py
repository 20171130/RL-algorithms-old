args = Config()
args.env=env_name #discrete action
args.algorithm="dqn"
args.name=f"{args.env}_{args.algorithm}"
args.gpu=0
args.seed=0
args.cpu=4
args.steps_per_epoch=5000
args.epochs=500

model_args=Config()
model_args.hidden_sizes=[256]*4
model_args.activation=torch.nn.ReLU
model_args = Config()
model_args.gamma=0.99
model_args.polyak=0.995
model_args.lr=3e-5
model_args.alpha=0
model_args.eps=0.01
model_args.dqn=True
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
result =sac(lambda : gym.make(args.env), model=model, logger=logger, 
           steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
run.finish()