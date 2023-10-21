from requests_tor import RequestsTor
from queue import Queue
import subprocess
import time
import pandas as pd
from tqdm import tqdm
import traceback
import multiprocessing


def crawl_single_page(rt, domain, relay_ip, relay_port):
    tcpdump_command = ['tshark', '-i', 'en0', '-f', f"host {relay_ip} && tcp port {relay_port}", '-w']

    save_path = f"./tor-pcap/{domain}.pcap"
    tcpdump_command.append(save_path)
    p = subprocess.Popen(tcpdump_command, stdout=subprocess.PIPE)
    time.sleep(1)
    try:
        rt.get("https://www." + domain, stream=False, timeout=(15, 30))
    except Exception as e:
        traceback.format_exc()
    finally:
        p.terminate()
    return save_path


def crawl_pages(site_list, queue, relay_ip, relay_port):
    print("crawling method starts")
    rt = RequestsTor(tor_ports=(9050,), tor_cport=9051, autochange_id=0)
    sites = pd.read_csv(site_list)["site"].to_list()[:10]
    for idx, site in enumerate(tqdm(sites)):
        print(f"{idx}/{len(sites)}")
        save_path = crawl_single_page(rt, site, relay_ip, relay_port)
        queue.put(save_path)

        if idx % 50 == 0:
            print(f"crawled: {idx}")
            queue.put(idx)

        if idx == len(sites) - 1:
            queue.put('done')


def extract_pkt_timestamp_from_pcap(output_file, queue):
    print("extracting method starts")
    while True:
        pcap_path = queue.get()
        if isinstance(pcap_path, int) and pcap_path % 50 == 0:
            print(f"processed: {pcap_path}")
            continue
        if pcap_path == 'done':
            break
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

        # get packet size
        command = ['tshark', '-r', pcap_path, '-T', 'fields', '-e', 'tcp.len']
        result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        packet_list = result.split('\n')[:-1]
        packet_sizes = [int(i) for i in packet_list]

        # get directions
        command = ['tshark', '-r', pcap_path, '-T', 'fields', '-e', 'ip.src']
        result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        ips = result.split('\n')[:-1]
        if len(ips) == 0:
            return
        src_ip = ips[0]
        directions = [1 if src_ip == ip else -1 for ip in ips]

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
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    site_list = "./misc/sample_sites.csv"
    output_file = "extracted_flows.txt"
    # TODO: Provide relay node ip and port number
    relay_ip = None
    relay_port = None
    crawl_proc = multiprocessing.Process(target=crawl_pages, args=(site_list, queue, relay_ip, relay_port))
    extract_proc = multiprocessing.Process(target=extract_pkt_timestamp_from_pcap, args=(output_file, queue))
    crawl_proc.start()
    extract_proc.start()
    # crawl_proc.join()
    extract_proc.join()
