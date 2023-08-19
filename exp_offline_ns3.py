from config import cfg_from_yaml_file, get_agent_names, draw_rectangle, split_file_name
import time
import zmq
import json
import threading
from socket import *

from pathlib import Path
import numpy as np
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--zmqPort', type=int, default=5557)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    return parser.parse_args()

args = parse_args()

m_zmqPort =  args.zmqPort

m_root_path = Path () / args.path # Path() / "G:/dataGenerator/output/20230413_False_1_6_110_0_600_5"
m_ns3_dir = "ns3_" + split_file_name(args.config)[1]
m_cfg = cfg_from_yaml_file(m_root_path / args.config)


m_zmq_send_position = zmq.Context().socket(zmq.PUSH)
m_zmq_send_cp = zmq.Context().socket(zmq.PUSH)

m_ns3_event = threading.Event()
m_sync_lock_time = threading.Lock()

m_sync_time = m_cfg["NS3"]["SYNC_TIME"]
m_fps = m_cfg["CARLA_CONFIG"]["FPS"]
m_last_stop_time = 0.0
m_start_time = 0.0


def get_collection_time(timestamp):
    global m_sync_lock_time
    m_sync_lock_time.acquire()
    global m_start_time
    result = timestamp - m_start_time
    m_sync_lock_time.release()
    return round(result, 16)

def fun_zmq_send_position(id, location):
    global m_zmq_send_position
    msg = {'id': id, 'type': "setPosition", 'x': location["x"], 'y': location["y"],
           'z': location["z"], 'timestamp': get_collection_time(location["timestamp"])}
    m_zmq_send_position.send_json(msg)
    print("fun_zmq_send_position: {}".format(msg))


def fun_zmq_recv_result(zmq_host, agent_names, ns3_dir):
    global m_root_path
    global m_cfg
    agent_names = get_agent_names(m_cfg)
    for agent_name in agent_names:
        save_dir = Path() / m_root_path / "object" / agent_name / ns3_dir
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)
        
    context = zmq.Context()
    zmq_receiver = context.socket(zmq.PULL)
    zmq_receiver.connect(zmq_host)

    while True:
        recv_json = zmq_receiver.recv_json()
        agent = agent_names[int(recv_json["receiveMac"])] if recv_json["messageType"] == "CR" else agent_names[recv_json["id"]]
        save_dir = Path() / m_root_path / "object" / agent / ns3_dir
        with open(save_dir / "{0:06}.txt".format(int(recv_json["frame"])), "a+") as f:
            f.write(json.dumps(recv_json)+"\n")


def fun_zmq_send_lu(id, location, frame_number, agent_name):
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    
    
    lu_cast_method = m_cfg["NS3"]["LU"]["CAST_METHOD"]
    lu_wifi_mode = m_cfg["NS3"]["LU"]["WIFI_MODE"]
    lu_receiver_id = m_cfg["NS3"]["RECEIVER_ID"] if lu_cast_method == "unicast" else -1
    lu_packet_size = m_cfg["NS3"]["LU"]["PACKET_SIZE"]
    lu_time_offset = m_cfg["NS3"]["LU"]["TIME_OFFSET"]
    
    timestamp = get_collection_time(location["timestamp"]) + lu_time_offset
    
    timestamp = max(0.01, timestamp)
    
    x, y, z, rx, ry, rz = location["x"], location["y"], location["z"], location["rx"], location["ry"], location["rz"]
    msg = {'id': id, 'agentName':agent_name, 'messageType': 'LU', 'type': 'pushEvent', 'castMethod': lu_cast_method, 'receiverId': lu_receiver_id, 'mode': lu_wifi_mode, 'size': lu_packet_size, 'frame': frame_number, 'timestamp': timestamp, 'segment': -1, 'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}
    print("fun_zmq_send_lidar: {}".format(msg))
    m_zmq_send_cp.send_json(msg)
            
def fun_zmq_send_pu(id, location, frame_number, agent_name, fps=10):
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    
    pu_cast_method = m_cfg["NS3"]["PU"]["CAST_METHOD"]
    pu_wifi_mode = m_cfg["NS3"]["PU"]["WIFI_MODE"]
    pu_receiver_id = m_cfg["NS3"]["RECEIVER_ID"] if pu_cast_method == "unicast" else -1
    
    x, y, z, rx, ry, rz = location["x"], location["y"], location["z"], location["rx"], location["ry"], location["rz"]
    
    npy_packet_file_name = Path() / m_root_path  / "object" / agent_name / "packet" / "{0:06}.npy".format(frame_number)
    assert npy_packet_file_name.exists(), "npy_packet_file_name: {} not exists".format(npy_packet_file_name)
    
    packet_size_list = np.load(npy_packet_file_name, allow_pickle=True).item()["packet_size_list"]
    sector_interval_time  = 1.0 / fps / len(packet_size_list)
    for i, packet_size in enumerate(packet_size_list.tolist()):
        if packet_size <= 0:
            continue
        elif packet_size < 100:
            packet_size = 100
        msg = {'id': id, 'agentName':agent_name, 'messageType': 'PU', 'type': 'pushEvent', 'castMethod': pu_cast_method, 'receiverId': pu_receiver_id, 'mode': pu_wifi_mode, 'size': packet_size, 'frame': frame_number, 'timestamp': get_collection_time(location["timestamp"]) + sector_interval_time * i, 'segment': i, 'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}
        print("fun_zmq_send_lidar: {}".format(msg))
        m_zmq_send_cp.send_json(msg)

def fun_zmq_send_cr(id, location, frame_number, agent_name):
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    
    
    cr_cast_method = m_cfg["NS3"]["CR"]["CAST_METHOD"]
    cr_wifi_mode = m_cfg["NS3"]["CR"]["WIFI_MODE"]
    cr_receiver_id = m_cfg["NS3"]["RECEIVER_ID"] if cr_cast_method == "unicast" else -1
    cr_packet_size = m_cfg["NS3"]["CR"]["PACKET_SIZE"]
    cr_time_offset = m_cfg["NS3"]["CR"]["TIME_OFFSET"]
    
    timestamp = get_collection_time(location["timestamp"]) + cr_time_offset
    timestamp = max(0.01, timestamp)
    
    x, y, z, rx, ry, rz = location["x"], location["y"], location["z"], location["rx"], location["ry"], location["rz"]
    msg = {'id': cr_receiver_id, 'agentName':agent_name, 'messageType': 'CR', 'type': 'pushEvent', 'castMethod': cr_cast_method, 'receiverId': id, 'mode': cr_wifi_mode, 'size': cr_packet_size, 'frame': frame_number, 'timestamp': timestamp, 'segment': -1, 'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}
    print("fun_zmq_send_lidar: {}".format(msg))
    m_zmq_send_cp.send_json(msg)


    
def fun_zmq_recv_status(zmq_host):
    context = zmq.Context()
    zmq_receiver = context.socket(zmq.PULL)
    zmq_receiver.connect(zmq_host)
    while True:
        json = zmq_receiver.recv_json()
        print("fun_zmq_recv_status: {}".format(json))
        m_ns3_event.set()
  
def main():
    global m_last_stop_time
    global m_sync_time
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    global m_zmq_send_position
    global m_fps
    global m_ns3_event
    global m_start_time
    global m_ns3_dir
    global m_zmqPort
    
    # init ZMQ connection
    m_zmq_send_position.bind("tcp://127.0.0.1:{}".format(m_zmqPort))
    m_zmq_send_cp.bind("tcp://127.0.0.1:{}".format(m_zmqPort+1))
    
    
    
    agent_names = get_agent_names(m_cfg)
    assert len(agent_names) > 0, "agent_list is empty"
    
    frame_size = len(sorted((m_root_path / "object" / agent_names[0] / "label_2").glob("*.txt")))
    assert frame_size > 0, "frame is empty"
    print("frame_size: {}".format(frame_size))
    
    # listen
    thread_recv_result = threading.Thread(target=fun_zmq_recv_result, args=("tcp://127.0.0.1:{}".format(m_zmqPort+2), list(m_cfg["AGENT_CONFIG"].keys()), m_ns3_dir, ))
    thread_recv_result.daemon = True
    thread_recv_result.start()

    thread_recv_status = threading.Thread(target=fun_zmq_recv_status, args=("tcp://127.0.0.1:{}".format(m_zmqPort+3), ))
    thread_recv_status.daemon = True
    thread_recv_status.start()
    m_ns3_event.set()
    
    # init world time
    m_start_time = np.load(m_root_path / "object" / agent_names[0] / "location" / "000000.npy", allow_pickle=True).item()["timestamp"]
    m_last_stop_time = m_start_time
    print("please running NS3 now !!!")
    
    # init edge server
    nodes_edge_server = 0 if m_cfg["EDGE_SERVER_CONFIG"] is None else len(m_cfg["EDGE_SERVER_CONFIG"])
    if nodes_edge_server > 0:
        for edge_server_name, edge_server_config in m_cfg["EDGE_SERVER_CONFIG"].items():
            npy_location = {}
            if edge_server_config["DSRC_WIFI_NODE"]["ENABLE"]:
                # init the edge server fill ap position
                fill_points = draw_rectangle(edge_server_config["LOCATION"][0], edge_server_config["LOCATION"][1], edge_server_config["LOCATION"][2], edge_server_config["DSRC_WIFI_NODE"]["WIDTH"], edge_server_config["DSRC_WIFI_NODE"]["WIDTH"], edge_server_config["DSRC_WIFI_NODE"]["INTERAL"])
                print("fill_points: {}".format(len(fill_points)))
                for i, p in enumerate(fill_points):
                    npy_location = {}
                    npy_location["x"], npy_location["y"], npy_location["z"], npy_location["timestamp"] = p[0], p[1], p[2], m_start_time
                    fun_zmq_send_position(int(edge_server_config["ID"]) + (i+1), npy_location)
            # init the edge server position
            npy_location["x"], npy_location["y"], npy_location["z"], npy_location["timestamp"] = edge_server_config["LOCATION"][0], edge_server_config["LOCATION"][1], edge_server_config["LOCATION"][2], m_start_time
            fun_zmq_send_position(edge_server_config["ID"], npy_location)
    
    # upload the all position and messages
    for frame_number in range(frame_size):
        m_ns3_event.wait()
        for agent_id, agent_name in enumerate(agent_names):
            npy_location_file_name = m_root_path / "object" / agent_name / "location" / "{0:06}.npy".format(frame_number)
            npy_location = np.load(npy_location_file_name, allow_pickle=True).item()
            fun_zmq_send_position(agent_id, npy_location)
            if m_cfg["NS3"]["LU"]["ENABLE"]:
                fun_zmq_send_lu(agent_id, npy_location, frame_number, agent_name)
            if m_cfg["NS3"]["PU"]["ENABLE"]:
                fun_zmq_send_pu(agent_id, npy_location, frame_number, agent_name, m_fps)
            if m_cfg["NS3"]["CR"]["ENABLE"]:
                fun_zmq_send_cr(agent_id, npy_location, frame_number, agent_name)
            
        if npy_location["timestamp"] - m_last_stop_time > m_sync_time:
            m_ns3_event.clear()
            m_last_stop_time = npy_location["timestamp"]
            print("carla pause: {}".format(get_collection_time(npy_location["timestamp"])))
            m_zmq_send_cp.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
            m_zmq_send_position.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
    
    for i in range(500):
        m_zmq_send_cp.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
        m_zmq_send_position.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
        
    for i in range(1000):
        print("waiting stop...")
        time.sleep(1)

        

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Total time: {}".format(time.time() - start_time))