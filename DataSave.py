from config import config_to_trans
import copy
from export_utils import *
import shutil
from pathlib import Path

class DataSave:
    def __init__(self, cfg):
        self.cfg = cfg
        self.OBJECT_FOLDER = None
        self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num(list(self.cfg["AGENT_CONFIG"].keys()))
        self.LIDAR_HEIGHT_POS = self.cfg["CARLA_CONFIG"]["FPS"]

    def _generate_path(self, root_path):
        self.OBJECT_FOLDER = os.path.join(root_path, "object")
        self.IMAGESETS_FOLDER = os.path.join(root_path, "ImageSets")

        if not os.path.exists(self.IMAGESETS_FOLDER):
            os.makedirs(self.IMAGESETS_FOLDER)

        objects_folders = ['calib', 'image_2', 'image_label_2', 'label_2',
                           'carla_label', 'velodyne', 'location', 'ply', 'planes', 'semantic3d_xyzirgb', 'semantic3d_label', 'velodyne_fg', 'velodyne_ground_removal', 'velodyne_compression', 'packet']

        for agent_name, agent_config in self.cfg["AGENT_CONFIG"].items():
            if agent_config["ENABLE"] and "SENSOR_CONFIG" in agent_config: # has_key  and agent_config["SENSOR_CONFIG"] is not None
                for folder in objects_folders:
                    directory = os.path.join(self.OBJECT_FOLDER, agent_name, folder)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

    def _current_captured_frame_num(self, agent_names):
        label_path = os.path.join(self.OBJECT_FOLDER, agent_names[0], 'label_2/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print("{} records in the current directory".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite, (A)ppend or (C)lear the dataset? (O/A/C)".format(
                self.OBJECT_FOLDER))
        if answer.upper() == "O":
            print("Resetting frame number to 0 and overwriting existing")
            return 0
        if answer.upper() == "C":
            print("Clearing existing dataset...")
            shutil.rmtree(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
            self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
            print("Resetting frame number to 0")
            return 0
        print("Continuing recording data on frame number {}".format(num_existing_data_files))
        return num_existing_data_files

    def save_training_files(self, data, agent_configs, timestamp):
        if self.cfg["CARLA_CONFIG"]["SKIP_EMPTY"]:
            for agent, dt in data["agents_data"].items():
                if self.check_datapoint(dt["kitti_datapoints"]) == False:
                    print("skip empty frame")
                    return
        is_first_obj = True
        
        voxel_size, ransac_n, distance_threshold, num_iterations = self.cfg["SIMULATOR"]["GROUND_REMOVAL"]["VOXEL_SIZE"], self.cfg["SIMULATOR"]["GROUND_REMOVAL"]["RANSAC_N"], self.cfg["SIMULATOR"]["GROUND_REMOVAL"]["DISTANCE_THRESHOLD"], self.cfg["SIMULATOR"]["GROUND_REMOVAL"]["NUM_ITERATIONS"]
        packet_mode, packet_size, sector_size = self.cfg["NS3"]["PU"]["PACKET_MODE"], self.cfg["NS3"]["PU"]["PACKET_SIZE"], self.cfg["SIMULATOR"]["SECTOR_SIZE"]
        for agent, dt in data["agents_data"].items():
            save_dir = Path() / self.OBJECT_FOLDER / dt["agent_name"]
            lidar_fname = save_dir / "velodyne/{0:06}.bin".format(self.captured_frame_no)
            ground_removal_lidar_fname = save_dir / "velodyne_ground_removal/{0:06}.bin".format(self.captured_frame_no)
            kitti_label_fname = save_dir / "label_2/{0:06}.txt".format(self.captured_frame_no)
            carla_label_fname = save_dir / "carla_label/{0:06}.txt".format(self.captured_frame_no)
            img_fname = save_dir / "image_2/{0:06}.png".format(self.captured_frame_no)
            img_label_fname = save_dir / "image_label_2/{0:06}.png".format(self.captured_frame_no)
            calib_fname = save_dir / "calib/{0:06}.txt".format(self.captured_frame_no)
            location_fname = save_dir / "location/{0:06}.npy".format(self.captured_frame_no)
            location_txt_fname = save_dir / "location/{0:06}.txt".format(self.captured_frame_no)
            ply_fname = save_dir / "ply/{0:06}.ply".format(self.captured_frame_no)
            groundplane_fname = save_dir / "planes/{0:06}.txt".format(self.captured_frame_no)
            semantic3d_xyzirgb_fname = save_dir / "semantic3d_xyzirgb/{0:06}.txt".format(self.captured_frame_no)
            semantic3d_label_fname = save_dir / "semantic3d_label/{0:06}.labels".format(self.captured_frame_no)
            fg_lidar_fname = save_dir / "velodyne_fg/{0:06}.bin".format(self.captured_frame_no)
            fg_ply_fname = save_dir / "velodyne_fg/{0:06}.ply".format(self.captured_frame_no)

            camera_transform = config_to_trans(agent_configs[agent]["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
            lidar_transform = config_to_trans(agent_configs[agent]["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["CALIB"]:
                save_calibration_matrices([camera_transform, lidar_transform], calib_fname, dt["intrinsic"])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["IMAGE_2"]:
                save_image_data(img_fname, dt["sensor_data"][0])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["IMAGE_LABEL_2"]:
                save_image_data_label(img_label_fname, img_fname, dt["kitti_datapoints"], dt["occlude_list"], dt["bbox2d_area_list"])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["CARLA_LABEL"]:
                save_label_data_carla(carla_label_fname, dt['carla_datapoints'])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["VELODYNE"]:
                save_lidar_data(lidar_fname, dt["sensor_data"][2])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["LOCATION"]:
                dt["location"]['timestamp'] = timestamp
                save_location(location_fname, dt["location"])
                save_location_txt(location_txt_fname, dt["location"])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["PLY"]:
                save_ply_by_bin(ply_fname, dt["sensor_data"][2])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["PLANES"]:
                save_groundplanes(groundplane_fname, dt["location"]["rx"], dt["location"]["ry"], self.LIDAR_HEIGHT_POS)
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["SEMANTIC3D"]:
                save_semantic3d(semantic3d_xyzirgb_fname, semantic3d_label_fname, lidar_fname, dt["sensor_data"][3])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["LABEL_2"]:
                save_label_data_kitti(kitti_label_fname, dt["kitti_datapoints"])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["FG_LIDAR"]:
                save_fg_lidar_data(fg_lidar_fname, fg_ply_fname, dt["sensor_data"][2], dt["kitti_datapoints"])
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["VELODYNE_GROUND_REMOVAL"]:
                save_ground_removal_lidar_data(ground_removal_lidar_fname, dt["sensor_data"][2], voxel_size, ransac_n, distance_threshold, num_iterations)
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["VELODYNE_COMPRESSION"]:
                save_compression_lidar_data(ground_removal_lidar_fname if self.cfg["NS3"]["PU"]["PACKET_MODE"] == 2 else lidar_fname, save_dir / "velodyne_compression")
            if self.cfg["SAVE_CONFIG"]["IS_SAVE"]["PACKET"]:
                save_packet_data(save_dir / "velodyne_compression" / "{0:06}.bin".format(self.captured_frame_no), packet_mode, packet_size, sector_size, timestamp, save_dir / "packet")
            if is_first_obj:  
                save_ref_files(self.IMAGESETS_FOLDER, self.captured_frame_no)
                is_first_obj = False
        self.captured_frame_no += 1

    def check_datapoint(self, datapoints):
        for point in datapoints:
            if point is not None:
                kitti_data = str(point)
                kitti_datas = kitti_data.split(" ")
                if len(kitti_datas) == 15 and kitti_datas[0]!="DontCare":
                    return True
        return False