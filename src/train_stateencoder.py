import os
import shutil
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import RandomRecordDataset
from seq2seq import Seq2Seq, StateEncoder
import numpy as np
from utils import Timer
from utils import parse_data_config, setup_logger
from tqdm import tqdm
import argparse
import pandas as pd


def recon_loss(X, y_hat):
    y = torch.flip(X, dims=[1])
    loss = torch.mean((y - y_hat) ** 2)
    return loss


def train(epoch, model, optimizer, dataset):
    epoch_loss = 0
    for X in tqdm(dataset):
        X = X.to(args.device)
        batch_len = torch.randint(low=1, high=args.max_len, size=(1,)).item()
        partial_X = X[:, :batch_len, :]
        if len(partial_X.size()) == 2:
            partial_X = partial_X.unsqueeze(dim=1)
        partial_y_hat = model(partial_X)
        loss = recon_loss(partial_X, partial_y_hat)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update loss
        epoch_loss += loss.item()

    epoch_loss /= len(dataset) * args.max_len
    args.logger.info(f"[{epoch}] train epoch loss: {epoch_loss:.4f}")
    print(f"[{epoch}] train epoch loss: {epoch_loss:.4f}")
    return epoch_loss


def test(epoch, model, dataset):
    epoch_loss = 0
    for X in tqdm(dataset):
        X = X.to(args.device)
        for i in range(1, args.max_len + 1):
            partial_X = X[:, :i, :]
            if len(partial_X.size()) == 2:
                partial_X = partial_X.unsqueeze(dim=1)
            partial_y_hat = model(partial_X)
            loss = recon_loss(partial_X, partial_y_hat)

            epoch_loss += loss.item()

    epoch_loss /= len(dataset) * args.max_len
    args.logger.info(f"[{epoch}] test epoch loss: {epoch_loss:.4f}")
    print(f"[{epoch}] test epoch loss: {epoch_loss:.4f}")
    return epoch_loss


def get_MAE(model, device, min=1, max=100):
    r_max = 1
    r_min = -1
    total_num = 1000
    model.eval()
    with torch.no_grad():
        nmaes = []
        errs = []
        for test_len in tqdm(range(min, max + 1)):
            max_len = test_len
            timestamp = torch.rand(total_num, max_len, 1)
            packet_size = (r_max - r_min) * torch.rand(total_num, max_len, 1) + r_min
            X = torch.cat([packet_size, timestamp], dim=2)
            X = X.to(device)
            y = model(X)
            y = torch.flip(y, dims=[1])
            # X: shape (total_num, len, 2)
            # (|X - y| / X).mean(dim=(1, 2)): NMAE per sample
            # (|X - y| / X).mean(dim=(1, 2)).mean(): MMAE
            # (|X - y| / X).mean(dim=(1, 2)).std(): std of MAE
            per_sample_nmae = torch.mean(torch.abs(X - y) / torch.abs(X), dim=(1, 2))
            nmae = torch.mean(per_sample_nmae * 100)
            nmae_err = torch.std(per_sample_nmae * 100) / per_sample_nmae.size(0)
            nmaes.append(nmae.cpu().numpy())
            errs.append(nmae_err.cpu().numpy())
    nmaes = np.array(nmaes)
    errs = np.array(errs)
    return nmaes, errs


def generate_nmae_csv(maes, errs, save_dir):
    x = np.arange(len(maes))
    mae_dict = {
        "x": x + 1,
        "means": maes,
        "upper": maes + errs,
        "lower": maes - errs
    }
    df = pd.DataFrame.from_dict(mae_dict)
    df.to_csv(save_dir + "tor_recon.csv", index=False)


if __name__ == "__main__":
    # parse passed-in arguments
    parser = argparse.ArgumentParser(description="Seq2Seq Parser",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ROOT", type=str, help="absolute directory of Amoeba")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="the hidden size of GRU")
    parser.add_argument("--num_layers", type=int, default=2, help="num layers")
    parser.add_argument("--epoch_num", type=int, default=1000, help="epoch num")
    parser.add_argument("--state_num", type=int, default=2,
                        help="2: encode both packet size and IAT.\n 1: encode packet size only")
    parser.add_argument("--max_len", type=int, default=70, help="max flow len to encode")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="training device: {'cuda', 'cpu'}")
    parser.add_argument("--train_num", type=int, default=100000, help="size of randomly generated train set")
    parser.add_argument("--test_num", type=int, default=3000, help="size of randomly generated test set")

    args = parser.parse_args()
    args.device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    args.logger = setup_logger(args.ROOT + f"logs/{args.exp_name}.log")

    # init save dir
    args.working_dir = args.ROOT + "models/" + args.exp_name + "/"
    if os.path.isdir(args.working_dir):
        shutil.rmtree(args.working_dir)
    os.mkdir(args.working_dir)

    exp_name = args.exp_name
    exp_timer = Timer("EXP Timer")
    # for plotting use
    train_loss = []
    test_loss = []

    # init
    model = Seq2Seq(device=args.device, state_num=args.state_num, num_layers=args.num_layers,
                    hidden_size=args.hidden_size)
    model = model.to(args.device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # load dataset
    train_set = DataLoader(
        RandomRecordDataset(state_num=args.state_num, total_num=args.train_num, max_len=args.max_len),
        batch_size=args.batch_size, shuffle=True,
        drop_last=True)
    test_set = DataLoader(RandomRecordDataset(state_num=args.state_num, total_num=args.test_num, max_len=args.max_len),
                          batch_size=args.batch_size, shuffle=True,
                          drop_last=True)
    min_test_loss = float("inf")
    for round_i in range(args.epoch_num):
        round_timer = Timer("round timer", args.logger)
        print(f"round: {round_i}")
        args.logger.info(f"round: {round_i}")
        args.logger.info("train\n")
        train_epoch_loss = train(round_i, model, optimizer, train_set)
        train_loss.append(train_epoch_loss)
        args.logger.info("test\n")
        test_epoch_loss = test(round_i, model, test_set)
        if round_i % 5 == 0 and min_test_loss > test_epoch_loss:
            min_test_loss = test_epoch_loss
            args.logger.info(f"saving at Epoch {round_i}")
            torch.save(model.state_dict(), args.working_dir + "seq2seq.pth")
        test_loss.append(test_epoch_loss)
        round_timer.end()

    model.load_state_dict(torch.load(args.working_dir + "seq2seq.pth"))
    maes, errs = get_MAE(model, args.device, max=args.max_len)
    generate_nmae_csv(maes, errs, args.working_dir)

    state_encoder = StateEncoder(args.device, 2, 2, 256)
    for target_param, param in zip(state_encoder.encoder.parameters(), model.encoder.parameters()):
        target_param.data.copy_(param.data)

    torch.save(state_encoder.state_dict(), args.working_dir + "stateencoder.pth")
    exp_timer.end()
