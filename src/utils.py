import os
import logging
import time
import json
import numpy as np
from scipy.stats import kurtosis, skew


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class Timer(object):
    def __init__(self, name="Timer", logger=None):
        self.name = name
        self.logger = logger
        self.start_time = time.time()

    def end(self):
        duration = time.time() - self.start_time
        if self.logger is None:
            print("{} : {}".format(self.name, time.strftime("%H:%M:%S", time.gmtime(duration))))
        else:
            self.logger.info("{} : {}".format(self.name, time.strftime("%H:%M:%S", time.gmtime(duration))))


def parse_data_config(ROOT, json_file_path):
    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)
    args = AttributeAccessibleDict(arguments_dict)
    args.stats_scaler = ROOT + args.stats_scaler
    args.train_dataset["X"] = ROOT + args.train_dataset["X"]
    args.train_dataset["y"] = ROOT + args.train_dataset["y"]
    args.agent_train_dataset["X"] = ROOT + args.agent_train_dataset["X"]
    args.val_dataset["X"] = ROOT + args.val_dataset["X"]
    args.val_dataset["y"] = ROOT + args.val_dataset["y"]
    args.test_dataset["X"] = ROOT + args.test_dataset["X"]
    args.test_dataset["y"] = ROOT + args.test_dataset["y"]
    return args


def setup_logger(log_file):
    if os.path.isfile(log_file):
        os.remove(log_file)
    logger = logging.getLogger("application")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def gen_global_statistics(flow, direction):
    if direction == "fwd":
        target_flow = flow[flow[:, 0] > 0]
    elif direction == "bwd":
        target_flow = np.abs(flow[flow[:, 0] < 0])
    else:
        target_flow = np.abs(flow)
    pkt_sizes = target_flow[:, 0]
    timestamps = target_flow[:, 1]
    if len(target_flow) == 0:
        return np.zeros((32,))

    mean_size = np.mean(pkt_sizes)
    std_size = np.std(pkt_sizes)
    var_size = np.var(pkt_sizes)
    kurtosis_size = kurtosis(pkt_sizes, fisher=False)
    skew_size = skew(pkt_sizes)
    max_size = np.max(pkt_sizes)
    min_size = np.min(pkt_sizes)
    percentiles = np.percentile(pkt_sizes, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    pkt_stats = np.array([mean_size, std_size, var_size, kurtosis_size, skew_size, max_size, min_size])
    pkt_stats = np.concatenate([pkt_stats, percentiles], axis=0)

    mean_size = np.mean(timestamps)
    std_size = np.std(timestamps)
    var_size = np.var(timestamps)
    kurtosis_size = kurtosis(timestamps, fisher=False)
    skew_size = skew(timestamps)
    max_size = np.max(timestamps)
    min_size = np.min(timestamps)
    percentiles = np.percentile(timestamps, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    t_stats = np.array([mean_size, std_size, var_size, kurtosis_size, skew_size, max_size, min_size])
    t_stats = np.concatenate([t_stats, percentiles], axis=0)

    stats = np.concatenate([pkt_stats, t_stats], axis=0)

    return stats


def gen_burst(pkt_sizes):
    burst = []
    burst_count = []
    prev_direction = 1  # 1: fwd, -1:bwd
    for pkt in pkt_sizes:
        if len(burst) == 0:
            burst.append(pkt)
            burst_count.append(1)
            prev_direction = -1 if pkt < 0 else 1
            continue
        if pkt * prev_direction > 0:
            burst[-1] += pkt
            burst_count[-1] += 1
        else:
            burst.append(pkt)
            burst_count.append(1)
            prev_direction = prev_direction * -1
    return np.array(burst), np.array(burst_count)


def gen_burst_statistics(burst, burst_count, direction):
    if direction == "fwd":
        target_burst = burst[burst > 0]
        target_burst_count = burst_count[burst > 0]
    else:
        target_burst = - burst[burst < 0]
        target_burst_count = burst_count[burst < 0]

    if len(target_burst) == 0:
        return np.zeros((32,))

    tot_bursts_count = target_burst_count.shape[0]
    mean_bursts_count = np.mean(target_burst_count)
    std_bursts_count = np.std(target_burst_count)
    var_bursts_count = np.var(target_burst_count)
    max_bursts_count = np.max(target_burst_count)
    kurtosis_bursts_count = kurtosis(target_burst_count, fisher=False)
    skew_bursts_count = skew(target_burst_count)
    percentiles_count = np.percentile(target_burst_count, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    burst_count_statistics = np.array(
        [tot_bursts_count, mean_bursts_count, std_bursts_count, var_bursts_count, max_bursts_count,
         kurtosis_bursts_count,
         skew_bursts_count])

    mean_bursts = np.mean(target_burst)
    std_bursts = np.std(target_burst)
    var_bursts = np.var(target_burst)
    max_bursts = np.max(target_burst)
    min_bursts = np.min(target_burst)
    kurtosis_bursts = kurtosis(target_burst, fisher=False)
    skew_bursts = skew(target_burst)
    percentiles = np.percentile(target_burst, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    burst_statistics = np.array([mean_bursts, std_bursts, var_bursts, max_bursts, min_bursts, kurtosis_bursts,
                                 skew_bursts])

    return np.concatenate([burst_count_statistics, percentiles_count, burst_statistics, percentiles], axis=0)


def generate_statistics(flow: np.ndarray):
    pkt_sizes = flow[:, 0]
    abs_sizes = np.abs(pkt_sizes)
    # summary statistics
    tot_records = pkt_sizes.shape[0]
    tot_fwd_records = np.sum(pkt_sizes > 0)
    tot_bwd_records = np.sum(pkt_sizes < 0)
    tot_bytes = np.sum(abs_sizes)
    tot_fwd_bytes = np.sum(pkt_sizes[pkt_sizes > 0])
    tot_bwd_bytes = - np.sum(pkt_sizes[pkt_sizes < 0])

    # global statistics
    global_stats = gen_global_statistics(flow, direction="bi")
    global_fwd_stats = gen_global_statistics(flow, direction="fwd")
    global_bwd_stats = gen_global_statistics(flow, direction="bwd")

    # burst statistics
    burst, burst_count = gen_burst(pkt_sizes)
    fwd_burst_statistics = gen_burst_statistics(burst, burst_count, direction="fwd")
    bwd_burst_statistics = gen_burst_statistics(burst, burst_count, direction="bwd")

    # concat
    summary_stat = np.array([tot_records, tot_fwd_records, tot_bwd_records, tot_bytes, tot_fwd_bytes, tot_bwd_bytes])
    # summary_stat = np.array([0, 0, 0, 0, 0, 0])
    tot_stat = np.concatenate([summary_stat, global_stats, global_fwd_stats, global_bwd_stats,
                               fwd_burst_statistics, bwd_burst_statistics])
    tot_stat[np.isnan(tot_stat)] = 0
    return tot_stat


def action2packet(action_mode, last_pkt, last_delay, action):
    pkt = np.zeros((2,))
    sign = 1 if last_pkt > 0 else -1
    if action_mode == "direction_limited":
        pkt[0] = sign * action[0]
        pkt[1] = action[1] + last_delay
        return pkt
    elif action_mode == "direction_unlimited":
        pkt[0] = action[0]
        pkt[1] = action[1] + last_delay
        return pkt
    elif action_mode == "pkt_change":
        byte_change = np.clip(action[0], -np.abs(last_pkt), 1)
        pkt[0] = np.clip(sign * byte_change + last_pkt, -1, 1)
        pkt[1] = action[1] + last_delay
        return pkt
    elif action_mode == "addition_only":
        pkt[0] = np.clip(sign * action[0] + last_pkt, -1, 1)
        pkt[1] = action[1] + last_delay
        return pkt
    else:
        raise ValueError("undefined action mode")
