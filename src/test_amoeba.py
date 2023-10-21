import argparse
import torch
from torch import optim
from test_args import test_args_dict
from torch.utils.data import DataLoader
from dataset import VariableRecordDataset
from discriminators import DecisionTreeDiscriminator
from discriminators import RandomForestDiscriminator, CUMULDisciminator
from discriminators import DFDiscriminator, SDAE, LSTMDiscriminator
from adversarial_env import AdvEnv
from seq2seq import StateEncoder
from stable_baselines3 import PPO
from utils import parse_data_config
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import pickle
import warnings


def init_discriminator_optimizer(args):
    dis_optimizer = None
    if args.dis == "dt":
        discriminator = DecisionTreeDiscriminator(args.device, args.stats_scaler, args.MAX_UNIT, args.MAX_DELAY)
    elif args.dis == "rf":
        discriminator = RandomForestDiscriminator(args.device, args.stats_scaler, args.MAX_UNIT, args.MAX_DELAY)
    elif args.dis == "df":
        discriminator = DFDiscriminator(args.device)
    elif args.dis == "sdae":
        discriminator = SDAE(args.device)
    elif args.dis == "lstm":
        discriminator = LSTMDiscriminator(args.device)
    elif args.dis == "cumul":
        discriminator = CUMULDisciminator(args.device)
    else:
        raise ValueError("unknown discriminator")
    if args.dis in nn_dis:
        discriminator = discriminator.to(args.device)
        dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=1e-3, weight_decay=1e-4)
    return discriminator, dis_optimizer


def evaluate_dis(dis, dataset, args):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for x, y in tqdm(dataset):
            x = x.to(args.device)
            y = y.to(args.device)
            if args.dis == "sdae":
                pred = dis(x)[0].cpu().numpy()
            else:
                pred = dis(x).cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_true = np.concatenate(y_true, axis=0)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"F1: {f1:.3f} accuracy: {accuracy:.3f}")


def evaluate_gen_flow(input, output):
    concat_flow = torch.concat(output, dim=0).cpu().numpy()
    input = np.array(input).reshape((-1, 2))
    data_overhead_rate = (np.abs(concat_flow[:, 0]).sum() - np.abs(input[:, 0]).sum()) / np.abs(concat_flow[:, 0]).sum()
    time_overhead_rate = (concat_flow[:, 1].sum() - input[:, 1].sum()) / concat_flow[:, 1].sum()
    return data_overhead_rate, time_overhead_rate


def evaluate(env, model, episodes, discriminator, args, mode="val"):
    results = []
    completed = 0
    data_overhead_rate = 0
    time_overhead_rate = 0
    done_num = 0
    for i in range(episodes):
        state = env.reset(mode=mode)
        while True:
            action, _ = model.predict(state, deterministic=True)
            action[1] = 0
            state, reward, done, info = env.step(action)

            if done:
                done_num += 1
                original_flow = info["original_flow"].cpu().numpy()
                flow_completed = info["completed"]
                virtual_adv_flows = info["virtual_adv_flows"]
                flow_do, flow_to = evaluate_gen_flow(original_flow, virtual_adv_flows)
                completed += flow_completed
                if flow_completed == 1:
                    data_overhead_rate += flow_do
                    time_overhead_rate += flow_to
                for flow in virtual_adv_flows:
                    flow = flow.view((1, -1, 2))
                    with torch.no_grad():
                        if args.dis == "sdae":
                            score = discriminator(flow)[0].cpu().numpy()
                        else:
                            score = discriminator(flow).cpu().numpy()
                        results.append(score)
                break

    results = np.array(results)
    ASR = (results > 0.5).sum() / results.shape[0] * 100
    data_overhead_rate = data_overhead_rate / (completed + 1) * 100
    time_overhead_rate = time_overhead_rate / (completed + 1) * 100
    completed = completed / episodes * 100

    print(args.dis + f" mode: {mode}")
    print(f"ASR: {ASR:.2f}%")
    print(f"DO: {data_overhead_rate:.2f}%")
    print(f"TO: {time_overhead_rate:.2f}%")
    return ASR, completed, data_overhead_rate, time_overhead_rate


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # args setup
    parser = argparse.ArgumentParser(description="Amoeba Parser",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ROOT", type=str, help="The directory of Amoeba")
    parser.add_argument("--dis", type=str, default="dt",
                        help="censoring classifier type, supports {'dt', 'rf', 'cumul', 'df', 'lstm', 'sdae'}")
    parser.add_argument("--dataset", type=str, default="v2ray", help="dataset type, supports {'tor', 'v2ray'}")

    args = parser.parse_args()
    if not args.ROOT.endswith("/"):
        args.ROOT = args.ROOT + "/"

    test_args = test_args_dict[args.dataset][args.dis]

    # append ROOT dir
    test_args.trained_dis_path = args.ROOT + test_args.trained_dis_path
    test_args.trained_amoeba_path = args.ROOT + test_args.trained_amoeba_path
    test_args.encoder_path = args.ROOT + test_args.encoder_path
    test_args.data_config = args.ROOT + test_args.data_config

    # parse data config
    data = parse_data_config(args.ROOT, test_args.data_config)
    test_args.stats_scaler = data.stats_scaler
    test_args.train_dataset = data.train_dataset
    test_args.agent_train_dataset = data.agent_train_dataset
    test_args.val_dataset = data.val_dataset
    test_args.test_dataset = data.test_dataset

    test_args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(test_args.device)
    nn_dis = ["df", "sdae", "lstm"]

    # discriminator training/loading
    observation_num = 2
    discriminator, optimizer = init_discriminator_optimizer(test_args)
    X_path, y_path = test_args.test_dataset["X"], test_args.test_dataset["y"]
    test_set = DataLoader(VariableRecordDataset(X_path, y_path, test_args.MAX_UNIT, test_args.MAX_DELAY, target=-1),
                          batch_size=1, shuffle=True, drop_last=True)
    if test_args.dis in nn_dis:
        discriminator.load_state_dict(torch.load(test_args.trained_dis_path, map_location=test_args.device))
        discriminator = discriminator.to(test_args.device)
    else:
        with open(test_args.trained_dis_path, 'rb') as f:
            discriminator.model = pickle.load(f)
    discriminator.eval()
    # evaluate_dis(discriminator, test_set, test_args)

    # loading state encoder
    print("state encoder loading")
    state_encoder = StateEncoder(test_args.device, state_num=observation_num, num_layers=test_args.layer_num,
                                 hidden_size=test_args.enc_dim)
    state_encoder.load_state_dict(torch.load(test_args.encoder_path, map_location=test_args.device))
    state_encoder = state_encoder.to(test_args.device)

    test_args.max_actions = (test_args.adv_pkt_clip, test_args.adv_iat_clip)
    single_env = AdvEnv(discriminator=discriminator, state_encoder=state_encoder,
                        train_dataset=test_args.agent_train_dataset,
                        val_dataset=test_args.val_dataset, test_dataset=test_args.test_dataset, device=test_args.device,
                        max_actions=test_args.max_actions, MAX_DELAY=test_args.MAX_DELAY,
                        MAX_UNIT=test_args.MAX_UNIT, enc_dim=test_args.enc_dim, action_mode=test_args.action_mode)

    # evaluate
    model = PPO.load(test_args.trained_amoeba_path, env=single_env)
    evaluate(single_env, model, test_args.test_num, discriminator, test_args, mode='test')
