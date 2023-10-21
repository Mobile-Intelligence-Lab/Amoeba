import requests
import traceback
import subprocess
import time
import pandas as pd
from tqdm import tqdm


def crawl_single_page(domain, v2ray_listen_port):
    tcpdump_command = ['tshark', '-i', 'eno1', '-w']
    save_path = "/root/tor-pcap/tmp.pcap"
    proxies = {'http': f'socks5://localhost:{v2ray_listen_port}',
               'https': f'socks5://localhost:{v2ray_listen_port}'}
    tcpdump_command.append(save_path)
    success = False
    p = subprocess.Popen(tcpdump_command, stdout=subprocess.PIPE)
    time.sleep(1)
    try:
        requests.get("https://www." + domain, stream=False, proxies=proxies)
        success = True
    except Exception as e:
        print("error")
    finally:
        time.sleep(1)
        p.terminate()
    return save_path, success


def clean_pcap(pcap_path, save_path):
    condition = "(ip.addr==3.82.252.238)&&(tcp.port==443)"
    target_command = ["tshark", '-r', pcap_path, "-Y", condition, '-w', save_path]
    subprocess.run(target_command)


def extract_pkt_timestamp_from_pcap(pcap_path, output_file):
    # get flow id
    command = ['tshark', '-r', pcap_path, '-T', 'fields', '-e', 'tcp.stream']
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    flow_id_list = result.split('\n')[:-1]
    flow_ids = [int(i) for i in flow_id_list]

    # get timestamp
    command = ['tshark', '-r', pcap_path, '-T', 'fields', '-e', 'frame.time_epoch']
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    timestamp_list = result.split('\n')[:-1]
    timestamps = [float(i) for i in timestamp_list]

    # get record size
    command = ['tshark', '-r', pcap_path, '-T', 'fields', '-e', 'tls.record.length']
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    packet_list = result.split('\n')[:-1]
    packet_sizes = [int(i) if i != "" else 0 for i in packet_list]

    # get directions
    command = ['tshark', '-r', pcap_path, '-T', 'fields', '-e', 'ip.src']
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    ips = result.split('\n')[:-1]
    if len(ips) == 0:
        return
    src_ip = ips[0]
    directions = [1 if src_ip == ip else -1 for ip in ips]

    assert len(packet_sizes) == len(timestamps)
    assert len(packet_sizes) == len(flow_ids)
    assert len(ips) == len(packet_sizes)

    data_dict = {}
    for idx, id in enumerate(flow_ids):
        if id not in data_dict:
            data_dict[id] = {"timestamp": "", "packet_size": ""}
        size = packet_sizes[idx]
        t = timestamps[idx]
        direction = directions[idx]
        if size != 0:
            data_dict[id]["timestamp"] += str(direction * size) + ","
            data_dict[id]["packet_size"] += str(t) + ","

    with open(output_file, "a+") as f:
        for key, item in data_dict.items():
            format = item["packet_size"] + "\n" + item["timestamp"] + "\n\n"
            f.write(format)


if __name__ == "__main__":
    sites = pd.read_csv("./misc/sample_sites.csv")["site"].to_list()
    output_file = "extracted_flows.txt"
    # TODO: Provider v2ray client port
    v2ray_listen_port = None
    for idx, site in enumerate(tqdm(sites)):
        save_path, success = crawl_single_page(site, v2ray_listen_port)
        clean_path = save_path + "_clean.pcap"
        clean_pcap(save_path, clean_path)
        try:
            if success:
                extract_pkt_timestamp_from_pcap(clean_path, output_file)
        except Exception as e:
            traceback.format_exc()
