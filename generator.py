from DataSave import DataSave
from SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file, cfg_from_list, get_agent_names, draw_rectangle
from data_utils import objects_filter
import time
import argparse
import zmq
import json
from multiprocessing import Process
import threading
from socket import *
from pathlib import Path
import numpy as np
import shutil
import sys
import glob
import os
try:
    sys.path.append(glob.glob('.../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

parser = argparse.ArgumentParser(description='carla-generator')
parser.add_argument('--n', type=int, default=10000, help='number of frames will be collect')
parser.add_argument('--d', type=str, default=None, help='name of save directory, if null will be generate according to config.yaml')
parser.add_argument('--s', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
parser.add_argument('--c', type=str, default="config.yaml", help='config file path')
args = parser.parse_args()

m_cfg = cfg_from_yaml_file(args.c)

cfg_from_list(m_cfg, args.set_cfgs)

m_zmq_send_position = zmq.Context().socket(zmq.PUSH)  # use to send location data
m_zmq_send_position.bind(m_cfg["ZMQ"]["SEND_POSITION"])

m_zmq_send_cp = zmq.Context().socket(zmq.PUSH)  # use to send lidar data
m_zmq_send_cp.bind(m_cfg["ZMQ"]["SEND_CP"])

m_ns3_event = threading.Event()

m_start_time = 0.0
m_last_stop_time = 0.0
m_sync_lock_time = threading.Lock()

m_sync_time = m_cfg["NS3"]["SYNC_TIME"]
m_use_ns3 = m_cfg["NS3"]["ENABLE"]
m_send_stop_signal = m_cfg["NS3"]["SEND_STOP_SIGNAL"]

m_root_path = None

def get_collection_time(snapshot):
    global m_sync_lock_time
    m_sync_lock_time.acquire()
    global m_start_time
    result = snapshot.timestamp.elapsed_seconds - m_start_time
    m_sync_lock_time.release()
    return round(result, 16)


def fun_zmq_recv_result():
    '''
    use to receive ns3 simulator results
    '''
    global m_root_path
    global m_cfg
    agent_names = get_agent_names(m_cfg)
    for agent_name in agent_names:
        save_dir = Path() / m_root_path / "object" / agent_name / "ns3"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
    context = zmq.Context()
    zmq_receiver = context.socket(zmq.PULL)
    zmq_receiver.connect(m_cfg["ZMQ"]["RECV_RESULT"])

    while True:
        recv_json = zmq_receiver.recv_json()
        agent = agent_names[int(recv_json["receiveMac"])] if recv_json["messageType"] == "CR" else agent_names[recv_json["id"]]
        save_dir = Path() / m_root_path / "object" / agent / "ns3"
        with open(save_dir / "{0:06}.txt".format(int(recv_json["frame"])), "a+") as f:
            f.write(json.dumps(recv_json)+"\n")

def fun_zmq_send_position_by_location(id, location, snapshot):
    global m_zmq_send_position
    msg = {'id': id, 'type': "setPosition", 'x': location.x, 'y': location.y,
           'z': location.z, 'timestamp': get_collection_time(snapshot)}
    m_zmq_send_position.send_json(msg)
    # print("fun_zmq_send_position_by_location: {}".format(msg))
    
def fun_zmq_send_lu(id, transform, frame_number, snapshot, agent_name):
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    
    
    x, y, z, rx, ry, rz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if transform is not None:
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        rx = transform.rotation.roll
        ry = transform.rotation.pitch
        rz = transform.rotation.yaw
    
    lu_cast_method = m_cfg["NS3"]["LU"]["CAST_METHOD"]
    lu_wifi_mode = m_cfg["NS3"]["LU"]["WIFI_MODE"]
    lu_receiver_id = m_cfg["NS3"]["RECEIVER_ID"] if lu_cast_method == "unicast" else -1
    lu_packet_size = m_cfg["NS3"]["LU"]["PACKET_SIZE"]
    lu_time_offset = m_cfg["NS3"]["LU"]["TIME_OFFSET"]
    msg = {'id': id, 'agentName':agent_name, 'messageType': 'LU', 'type': 'pushEvent', 'castMethod': lu_cast_method, 'receiverId': lu_receiver_id, 'mode': lu_wifi_mode, 'size': lu_packet_size, 'frame': frame_number, 'timestamp': get_collection_time(snapshot) + lu_time_offset, 'segment': -1, 'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}
    m_zmq_send_cp.send_json(msg)
    
def fun_zmq_send_pu(id, transform, frame_number, snapshot, agent_name, fps=10):
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    
    
    x, y, z, rx, ry, rz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if transform is not None:
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        rx = transform.rotation.roll
        ry = transform.rotation.pitch
        rz = transform.rotation.yaw
    
    pu_cast_method = m_cfg["NS3"]["PU"]["CAST_METHOD"]
    pu_wifi_mode = m_cfg["NS3"]["PU"]["WIFI_MODE"]
    pu_receiver_id = m_cfg["NS3"]["RECEIVER_ID"] if pu_cast_method == "unicast" else -1
    npy_packet_file_name = Path() / m_root_path  / "object" / agent_name / "packet" / "{0:06}.npy".format(frame_number)
    assert npy_packet_file_name.exists(), "npy_packet_file_name: {} not exists".format(npy_packet_file_name)
    
    packet_size_list = np.load(npy_packet_file_name, allow_pickle=True).item()["packet_size_list"]
    sector_interval_time  = 1.0 / fps / len(packet_size_list)
    for i, packet_size in enumerate(packet_size_list.tolist()):
        if packet_size <= 0:
            continue
        elif packet_size < 100:
            packet_size = 100
        msg = {'id': id, 'agentName':agent_name, 'messageType': 'PU', 'type': 'pushEvent', 'castMethod': pu_cast_method, 'receiverId': pu_receiver_id, 'mode': pu_wifi_mode, 'size': packet_size, 'frame': frame_number, 'timestamp': get_collection_time(snapshot) + sector_interval_time * i, 'segment': i, 'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}
        m_zmq_send_cp.send_json(msg)

def fun_zmq_send_cr(id, transform, frame_number, snapshot, agent_name):
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    
    
    x, y, z, rx, ry, rz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if transform is not None:
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        rx = transform.rotation.roll
        ry = transform.rotation.pitch
        rz = transform.rotation.yaw
    
    cr_cast_method = m_cfg["NS3"]["CR"]["CAST_METHOD"]
    cr_wifi_mode = m_cfg["NS3"]["CR"]["WIFI_MODE"]
    cr_receiver_id = m_cfg["NS3"]["RECEIVER_ID"] if cr_cast_method == "unicast" else -1
    cr_packet_size = m_cfg["NS3"]["CR"]["PACKET_SIZE"]
    cr_time_offset = m_cfg["NS3"]["CR"]["TIME_OFFSET"]
    # EDGE SERVER TO AGENTS
    msg = {'id': cr_receiver_id, 'agentName': agent_name, 'messageType': 'CR', 'type': 'pushEvent', 'castMethod': cr_cast_method, 'receiverId': id, 'mode': cr_wifi_mode, 'size': cr_packet_size, 'frame': frame_number, 'timestamp': get_collection_time(snapshot) + cr_time_offset, 'segment': -1, 'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}
    m_zmq_send_cp.send_json(msg)

def check_update_status(snapshot):
    global m_zmq_send_position
    global m_zmq_send_cp
    global m_sync_time
    global m_last_stop_time
    global m_ns3_event
    elapsed_seconds = snapshot.timestamp.elapsed_seconds
    if elapsed_seconds - m_last_stop_time >= m_sync_time:
        m_ns3_event.clear()
        print("carla pause: {}".format(get_collection_time(snapshot)))
        m_last_stop_time = elapsed_seconds
        m_zmq_send_cp.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
        m_zmq_send_position.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause

def fun_zmq_recv_status():
    '''
    use to receive continue message
    '''
    global m_ns3_event
    context = zmq.Context()
    zmq_receiver = context.socket(zmq.PULL)
    zmq_receiver.connect("tcp://127.0.0.1:5560")
    while True:
        json = zmq_receiver.recv_json()
        # print("fun_zmq_recv_status: {}".format(json))
        m_ns3_event.set()
    
def main():
    global m_start_time
    global m_last_stop_time
    global m_ns3_event
    global m_sync_time
    global m_cfg
    global m_root_path
    global m_zmq_send_cp
    global m_zmq_send_position
    global m_use_ns3
    global m_send_stop_signal
    global args
    
    print(args)
    
    if args.d is not None:
        root_dir = args.d
    else:
        root_dir = "{}_{}_{}_{}_{}_{}_{}".format(get_date(), m_cfg["CARLA_CONFIG"]["SKIP_EMPTY"], m_cfg["SAVE_CONFIG"]["STEP"], len(get_agent_names(m_cfg)), m_cfg["CARLA_CONFIG"]["NUM_OF_VEHICLES"], m_cfg["CARLA_CONFIG"]["NUM_OF_WALKERS"], args.n)
        
    m_root_path = "{}/{}".format(m_cfg["SAVE_CONFIG"]["ROOT_PATH"], root_dir)
    
    assert m_cfg["NS3"]["PU"]["PACKET_MODE"] == 1 or (m_cfg["NS3"]["PU"]["PACKET_MODE"] == 2 and m_cfg["SAVE_CONFIG"]["IS_SAVE"]["VELODYNE_GROUND_REMOVAL"] and m_cfg["SAVE_CONFIG"]["IS_SAVE"]["VELODYNE_COMPRESSION"]), "packet mode error" 
    assert m_cfg["NS3"]["ENABLE"] == False or m_cfg["SAVE_CONFIG"]["IS_SAVE"]["PACKET"], "ns3 mode error"
    
    m_cfg["SAVE_CONFIG"]["ROOT_PATH"] = m_root_path
    print("root path: {}".format(m_root_path))
    
    limit_fov = m_cfg["SIMULATOR"]["LIMIT_FOV"]
    
    fps = m_cfg["CARLA_CONFIG"]["FPS"]
    
    model = SynchronyModel(m_cfg)
    dtsave = DataSave(m_cfg)
    
    shutil.copy("config.yaml", Path() / m_root_path / "config.yaml")
    try:
        model.set_synchrony() # initialize carla world configuration
        vehicle_spawn_points = model.get_vehicle_spawn_points() # list all of vehicle spawn points
        unused_vehicle_spawn_points = model.spawn_agent(vehicle_spawn_points) # create agent vehicles
        model.spawn_actors(unused_vehicle_spawn_points) # create poor vehicles and walkers
        model.set_actors_route() # set poor vehicles and walkers route
        model.hide_environment_objects() # hide environment objects (since we can't labeling them)
        model.sensor_listen() # listen to sensor data from agent vehicles
        interval_step = m_cfg["SAVE_CONFIG"]["STEP"]
        save_count, step = 0, 0
        
        print("initializing environment...")
        
        start_time = time.time()
        
        # skip initial frames
        for i in range(20):
            model.world.tick()

        # college how many frames
        catch_frame_number = args.n

        nodes_edge_server = 0 if m_cfg["EDGE_SERVER_CONFIG"] is None else len(m_cfg["EDGE_SERVER_CONFIG"])
        nodes = len(get_agent_names(m_cfg)) + nodes_edge_server

        print("catch_frame_number: {}".format(catch_frame_number))
        print("nodes: {}".format(nodes))
        print("m_sync_time: {}".format(m_sync_time))
        print("m_use_ns3: {}".format(m_use_ns3))
        

        if m_use_ns3:
            thread_recv_result = threading.Thread(target=fun_zmq_recv_result)
            thread_recv_result.daemon = True
            thread_recv_result.start()

            thread_recv_status = threading.Thread(target=fun_zmq_recv_status)
            thread_recv_status.daemon = True
            thread_recv_status.start()
            m_ns3_event.set()

            print("please running NS3 now !!!")

            m_start_time = model.world.get_snapshot().timestamp.elapsed_seconds
            m_last_stop_time = m_start_time
        
            # init edge server location
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
                            npy_location["x"], npy_location["y"], npy_location["z"], npy_location["timestamp"] = p[0], p[1], p[2], 0.0
                            fun_zmq_send_position_by_location(int(edge_server_config["ID"]) + (i+1), npy_location, model.world.get_snapshot())
                    # init the edge server position
                    npy_location["x"], npy_location["y"], npy_location["z"], npy_location["timestamp"] = edge_server_config["LOCATION"][0], edge_server_config["LOCATION"][1], edge_server_config["LOCATION"][2], 0.0
                    fun_zmq_send_position_by_location(edge_server_config["ID"], npy_location, model.world.get_snapshot())
            
        while True:
            start_time = time.time()
            captured_frame_no = dtsave.captured_frame_no
            if captured_frame_no >= catch_frame_number:
                print("stop generator !!!")
                break
            
            if step % interval_step == 0:
                data = model.tick()
                data = objects_filter(data, limit_fov)
                dtsave.save_training_files(data, model.agent_configs, get_collection_time(model.world.get_snapshot()))
                if m_use_ns3:
                    for i, agent in enumerate(model.actors["agents"]):
                        if m_cfg["NS3"]["LU"]["ENABLE"]:
                            fun_zmq_send_lu(i, agent.get_transform(), int(captured_frame_no), model.world.get_snapshot(), list(model.agent_names.values())[i])
                        if m_cfg["NS3"]["PU"]["ENABLE"]:
                            fun_zmq_send_pu(i, agent.get_transform(), int(captured_frame_no), model.world.get_snapshot(), list(model.agent_names.values())[i], fps)
                        if m_cfg["NS3"]["CR"]["ENABLE"]:
                            fun_zmq_send_cr(i, agent.get_transform(), int(captured_frame_no), model.world.get_snapshot(), list(model.agent_names.values())[i])
                    check_update_status(model.world.get_snapshot())
                print("step: {}, frame number: {}, consuming time: {}".format(step, captured_frame_no, time.time() - start_time))
                save_count += 1
            else:
                model.world.tick()
            step += 1
            
            if m_use_ns3:
                m_ns3_event.wait()
                for i, agent in enumerate(model.actors["agents"]):
                    fun_zmq_send_position_by_location(i, agent.get_transform().location, model.world.get_snapshot())
                    
    finally:
        # recover the carla world actors
        model.setting_recover()
        
        # sending over signal to simulator the all packets are received
        if m_use_ns3 and m_send_stop_signal:
            for i in range(200):
                m_zmq_send_cp.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
                m_zmq_send_position.send_json({'type': 'updateStatus', 'status': 0})  # 0 = pause
                print("sending pause signal...")
                
            for i in range(100):
                print("waiting stop...")
                time.sleep(1)

def get_date():
    import datetime
    return str(datetime.date.today()).replace('-', '')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Total time: {}".format(time.time() - start_time))