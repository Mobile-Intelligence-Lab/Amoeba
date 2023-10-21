import argparse
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import VariableRecordDataset
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.env_util import make_vec_env
from discriminators import DecisionTreeDiscriminator
from discriminators import RandomForestDiscriminator, CUMULDisciminator
from discriminators import DFDiscriminator, SDAE, LSTMDiscriminator
from adversarial_env import AdvEnv
from seq2seq import StateEncoder
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from logger import SummaryWriterCallback
from utils import parse_data_config, Timer
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import pickle
import warnings


def dis_loss(y_hat, y):
    if args.dis == "sdae":
        pred, recon, x = y_hat
        pred = pred.view((-1,))
        y = y.view((-1,))
        return F.binary_cross_entropy(pred, y, reduction="sum") + F.mse_loss(recon, x, reduction="sum")
    else:
        y_hat = y_hat.view((-1,))
        y = y.view((-1,))
        return F.binary_cross_entropy(y_hat, y, reduction="sum")


def train_discriminator(discriminator, optimizer, train_dataset, args):
    if isinstance(discriminator, (LSTMDiscriminator, DFDiscriminator, SDAE)):
        train_nn_discriminator(discriminator, optimizer, train_dataset, args)
        torch.save(discriminator.state_dict(), args.trained_dis_path)
    else:
        train_X = train_dataset.dataset.x
        train_y = train_dataset.dataset.y
        pred_y, true_y = discriminator.fit(train_X, train_y)
        f1 = f1_score(y_true=true_y, y_pred=pred_y)
        accuracy = accuracy_score(true_y, pred_y)
        args.summary_writer.add_scalar('Dis_Train/F1', f1, 0)
        args.summary_writer.add_scalar('Dis_Train/accuracy', accuracy, 0)
        print(f"f1 {f1}, accuracy: {accuracy}")
        with open(args.trained_dis_path, 'wb') as f:
            pickle.dump(discriminator.model, f)
    args.summary_writer.flush()


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


def train_nn_discriminator(discriminator, optimizer, train_dataset, args):
    discriminator.train()
    batch_size = 32
    max_seq_len = args.max_episode_length

    for epoch_i in range(args.train_epochs):
        epoch_D_x = 0
        epoch_D_G_z = 0
        epoch_loss = 0
        real_num = 0
        fake_num = 0
        count = 0
        batch_count = 0
        batch_x = torch.zeros((batch_size, max_seq_len, 2), dtype=torch.float32, device=args.device)
        batch_x_lengths = []
        batch_y = torch.zeros((batch_size,), dtype=torch.float32, device=args.device)
        for x, y in tqdm(train_dataset):
            count += 1
            x = x.to(args.device)
            y = y.to(args.device)
            if y.item() == 1:
                x_len = x.size(1) if x.size(1) < max_seq_len else max_seq_len
                batch_x[batch_count, :x_len] = x[0, :x_len]
                batch_x_lengths.append(x_len)
                batch_y[batch_count] = y
            else:
                G_z = x[0] if x.size(1) < max_seq_len else x[0, :max_seq_len]
                batch_x[batch_count, :G_z.size(0)] = G_z
                batch_x_lengths.append(G_z.size(0))
                batch_y[batch_count] = y
            batch_count += 1
            if batch_count == batch_size:
                batch_pred = discriminator(batch_x, batch_size, batch_x_lengths)
                loss = dis_loss(batch_pred, batch_y)
                if args.dis == "sdae":
                    batch_pred = batch_pred[0]

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # statistics
                epoch_loss += loss
                real_num += (batch_y == 1).sum()
                fake_num += (batch_y == 0).sum()
                epoch_D_x += batch_pred[batch_y == 1].sum()
                epoch_D_G_z += batch_pred[batch_y == 0].sum()

                # empty batch
                batch_count = 0
                batch_x = torch.zeros((batch_size, max_seq_len, 2), dtype=torch.float32, device=args.device)
                batch_x_lengths = []
                batch_y = torch.zeros((batch_size,), dtype=torch.float32, device=args.device)

        epoch_D_x /= real_num
        epoch_D_G_z /= fake_num
        epoch_loss /= count

        args.summary_writer.add_scalar('Dis_Train/D(x)_mean', epoch_D_x.item(), epoch_i)
        args.summary_writer.add_scalar('Dis_Train/D(z)_mean', epoch_D_G_z.item(), epoch_i)
        args.summary_writer.add_scalar('Dis_Train/loss_mean', epoch_loss.item(), epoch_i)
        print(f"train D(x): {epoch_D_x:.2f}, train D(G(z)): {epoch_D_G_z:.2f}")


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
    print(f"F1: {f1:.2f}")


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

    print(args.exp_name + f" mode: {mode}")
    print(f"ASR: {ASR:.2f}%")
    print(f"DO: {data_overhead_rate:.2f}%")
    print(f"TO: {time_overhead_rate:.2f}%")
    return ASR, completed, data_overhead_rate, time_overhead_rate


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    exp_timer = Timer("exp timer")
    # args setup
    parser = argparse.ArgumentParser(description="Amoeba Parser",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ROOT", type=str, help="The directory of Amoeba")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--data_config", type=str, help="the json file of dataset paths")
    parser.add_argument("--dis", type=str, default="dt",
                        help="censoring classifier type, supports {'dt', 'rf', 'cumul', 'df', 'lstm', 'sdae'}")
    parser.add_argument("--trained_dis_path", type=str, default=None, help="path of trained censoring classifier")
    parser.add_argument("--train_epochs", type=int, default=10,
                        help="epoch size to training censoring classifiers if a trained model is not provided")
    parser.add_argument("--encoder_path", type=str, help="trained StateEncoder path")
    parser.add_argument("--enc_dim", type=int, default=256, help="StateEncoder hidden dim")
    parser.add_argument("--layer_num", type=int, default=2, help="StateEncoder layer num")
    parser.add_argument("--max_episode_length", type=int, default=60, help="the max encoding length of StateEncoder")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--data_penalty", type=float, default=1, help="data penalty")
    parser.add_argument("--truncate_penalty", type=float, default=0.05, help="truncate penalty")
    parser.add_argument("--time_penalty", type=float, default=2, help="time penalty")
    parser.add_argument("--MAX_UNIT", type=int, default=1448, help="supports {1448, 16500}. 1448: tor, 16500: v2ray")
    parser.add_argument("--MAX_DELAY", type=int, default=1, help="in second")
    parser.add_argument("--binary_signal", type=bool, default=False,
                        help="use classification scores for training if False")
    parser.add_argument("--adv_pkt_clip", type=float, default=1, help="no clipping by default")
    parser.add_argument("--adv_iat_clip", type=float, default=0.005,
                        help="per-packet added delay is limited within 0.05")
    parser.add_argument("--timesteps", type=int, default=300000, help="training steps")
    parser.add_argument("--test_num", type=int, default=3000, help="number of test samples")
    parser.add_argument("--mask_rate", type=float, default=0, help="reward mask rate")
    parser.add_argument(
        "--action_mode",
        type=str,
        default="direction_limited",
        help="""
            {`addition_only`, `direction_limited`, `direction_unlimited`} are supported.
            `addition_only`: allow padding only;
            `direction_limited`: allow padding and truncation;
            `direction_unlimited`: allow padding and truncation, dummy packets with different directions can be generated 
                                between truncated packets.
            The final option would potentially introduce the largest overhead as well as highest ASR.
        """
    )

    args = parser.parse_args()
    # parse data config
    data = parse_data_config(args.ROOT, args.data_config)
    args.stats_scaler = data.stats_scaler
    args.train_dataset = data.train_dataset
    args.agent_train_dataset = data.agent_train_dataset
    args.val_dataset = data.val_dataset
    args.test_dataset = data.test_dataset

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(args.device)
    args.log_dir = args.ROOT + "logs/" + args.exp_name + "/"
    if not os.path.isdir(args.log_dir):
        print("log dir not exist, create one.")
        os.mkdir(args.log_dir)

    args.model_save_dir = args.ROOT + "saved_models/"
    args.model_save_path = args.ROOT + "saved_models/" + args.exp_name + ".zip"
    args.summary_writer = SummaryWriter(log_dir=args.log_dir)
    nn_dis = ["df", "sdae", "lstm"]

    # discriminator training/loading
    observation_num = 2
    discriminator, optimizer = init_discriminator_optimizer(args)
    X_path, y_path = args.train_dataset["X"], args.train_dataset["y"]
    clf_train_set = DataLoader(VariableRecordDataset(X_path, y_path, args.MAX_UNIT, args.MAX_DELAY, target=-1),
                               batch_size=1, shuffle=True, drop_last=True)
    X_path, y_path = args.test_dataset["X"], args.test_dataset["y"]
    test_set = DataLoader(VariableRecordDataset(X_path, y_path, args.MAX_UNIT, args.MAX_DELAY, target=-1),
                          batch_size=1, shuffle=True, drop_last=True)
    if args.trained_dis_path is None or not os.path.isfile(args.trained_dis_path):
        print("did not find trained model, start training.")
        train_discriminator(discriminator, optimizer, clf_train_set, args)
    elif args.dis in nn_dis:
        discriminator.load_state_dict(torch.load(args.trained_dis_path, map_location=args.device))
        discriminator = discriminator.to(args.device)
    else:
        with open(args.trained_dis_path, 'rb') as f:
            discriminator.model = pickle.load(f)
    discriminator.eval()
    evaluate_dis(discriminator, test_set, args)

    # loading state encoder
    print("state encoder loading")
    state_encoder = StateEncoder(args.device, state_num=observation_num, num_layers=args.layer_num,
                                 hidden_size=args.enc_dim)
    state_encoder.load_state_dict(torch.load(args.encoder_path, map_location=args.device))
    state_encoder = state_encoder.to(args.device)

    # training PPO agent
    args.max_actions = (args.adv_pkt_clip, args.adv_iat_clip)
    single_env = AdvEnv(discriminator=discriminator, state_encoder=state_encoder,
                        train_dataset=args.agent_train_dataset,
                        val_dataset=args.val_dataset, test_dataset=args.test_dataset, device=args.device,
                        data_penalty=args.data_penalty, time_penalty=args.time_penalty,
                        truncate_penalty=args.truncate_penalty, max_actions=args.max_actions, MAX_DELAY=args.MAX_DELAY,
                        MAX_UNIT=args.MAX_UNIT, binary_signal=args.binary_signal, enc_dim=args.enc_dim,
                        mask_rate=args.mask_rate, action_mode=args.action_mode)
    print("training PPO")
    vec_env = make_vec_env(AdvEnv, 8, env_kwargs=dict(discriminator=discriminator, state_encoder=state_encoder,
                                                      train_dataset=args.agent_train_dataset,
                                                      val_dataset=args.val_dataset,
                                                      test_dataset=args.test_dataset, device=args.device,
                                                      data_penalty=args.data_penalty, time_penalty=args.time_penalty,
                                                      truncate_penalty=args.truncate_penalty,
                                                      max_actions=args.max_actions, MAX_DELAY=args.MAX_DELAY,
                                                      MAX_UNIT=args.MAX_UNIT,
                                                      binary_signal=args.binary_signal, enc_dim=args.enc_dim,
                                                      mask_rate=args.mask_rate, action_mode=args.action_mode))
    policy_kwargs = dict(activation_fn=nn.ReLU,
                         net_arch=[256, 64, 32])
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=args.learning_rate, policy_kwargs=policy_kwargs,
                n_steps=60)
    new_logger = configure(args.log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=args.timesteps, callback=SummaryWriterCallback(
        args.exp_name, args.model_save_dir, single_env, evaluate, discriminator, args
    ))
    model.save(args.model_save_path)

    # evaluate
    model = PPO.load(args.model_save_path, env=single_env)
    evaluate(single_env, model, args.test_num, discriminator, args, mode='test')
    exp_timer.end()
