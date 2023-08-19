import numpy as np
import os
import math
import copy
from json.tool import main
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
import open3d as o3d
from matplotlib import cm
from data_utils import *
import json
import time
from pathlib import Path

"""
This file contains all the methods responsible for saving the generated data in the correct output format.
"""

def save_ref_files(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')


def save_image_data(file_name, image):
    image.save_to_disk(str(file_name))


def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def draw_bbox_2d(image_file_name, bboxs, texts):
    image = Image.open(image_file_name)
    pdraw = ImageDraw.Draw(image)
    left_text_list = [["object: {}\n".format(len(texts)), (255, 0, 0)]]

    polygon_text_pts = []
    for i, bbox in enumerate(bboxs):
        polygon_pts = []
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        polygon_text_pts.append((x1, y1))

        polygon_pts.append((x1, y1))
        polygon_pts.append((x1, y2))
        polygon_pts.append((x2, y2))
        polygon_pts.append((x2, y1))
        rgb_color = get_random_color()
        pdraw.polygon(polygon_pts, fill=None, outline=rgb_color)
        pdraw.text(polygon_pts[0], texts[i], rgb_color)
        left_text_list.append([texts[i], rgb_color])

    for i, lt in enumerate(left_text_list):
        pdraw.text((10, 10 * i), lt[0], lt[1])

    # pdraw.text((10, 10), left_text, (255, 0, 0))
    return image


def save_image_data_label(file_name, image_file_name, datapoints, occlude_list, bbox2d_area_list):
    bbox_list = []
    text_list = []
    for i, point in enumerate(datapoints):
        if point is not None:
            kitti_data = str(point)
            kitti_datas = kitti_data.split(" ")
            #print("kitti_datas: {}".format(kitti_datas))
            if len(kitti_datas) == 15:
                bbox_list.append(np.asarray(kitti_datas)[4:8])
                text_list.append("{}.{} {} {} {:.2} {:1}".format(len(text_list)+1, kitti_datas[0], kitti_datas[1], kitti_datas[2], occlude_list[i], bbox2d_area_list[i]))
    if len(bbox_list) > 0:
        image = draw_bbox_2d(str(image_file_name), bbox_list, text_list)
        image.save(str(file_name))


def save_bbox_image_data(file_name, image):
    im = Image.fromarray(image)
    im.save(str(file_name))


def save_lidar_data(file_name, point_cloud, format="bin"):
    """ Saves lidar data to given file_name, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.
        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """

    if format == "bin":
        point_cloud = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
        
        lidar_array = point_cloud.copy()
        lidar_array[:, 1] = -lidar_array[:, 1]
        lidar_array = np.array(lidar_array).astype(np.float32)
        lidar_array.tofile(file_name)


def save_fg_lidar_data(bin_file_name, ply_file_name, point_cloud, datapoints):
    calibs = {"R0_rect": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "Tr_velo_to_cam": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]} #[skewen]: fixed calib config
    point_cloud = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

    
    lidar_array = point_cloud.copy()
    lidar_array[:, 1] = -lidar_array[:, 1]
    lidar_array = np.array(lidar_array).astype(np.float32)

    points_result = np.zeros((0, 4))

    for obj in datapoints:
        if obj is not None:
            kitti_datas = str(obj).split(" ")
            if len(kitti_datas) == 15:
                h, w, l, x, y, z, rotation_y = float(kitti_datas[8]),float(kitti_datas[9]), float(kitti_datas[10]), float(kitti_datas[11]), float(kitti_datas[12]), float(kitti_datas[13]), float(kitti_datas[14])
                points_result = np.concatenate((points_result, get_hull_points(lidar_array, compute_box_3d(calibs, h, w, l, x, y, z, rotation_y))), axis=0)
    
    points_result.astype(np.float32).tofile(str(bin_file_name))


    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_result[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(get_point_color(points_result[:, 3]))
    o3d.io.write_point_cloud(str(ply_file_name), point_list)
    
def save_label_data_kitti(file_name, datapoints):
    with open(file_name, 'w') as f:
        out_str = ""  # Pedestrian
        for point in datapoints:
            kitti_data = str(point)
            if out_str != "":
                out_str = out_str + "\n"
            out_str = out_str + kitti_data
                    
        if out_str != "":
            f.write(out_str)


def save_ground_removal_lidar_data(file_name, point_cloud, voxel_size, ransac_n, distance_threshold, num_iterations):
    import open3d as o3d
    
    start_time = time.process_time()
    point_cloud = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
    
    lidar_array = point_cloud.copy()
    lidar_array[:, 1] = -lidar_array[:, 1]
    points_xyzi = np.array(lidar_array).astype(np.float32)
    
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.normals = o3d.utility.Vector3dVector(np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1))
    
    pcd_filtered = point_list.voxel_down_sample(voxel_size=voxel_size)

    all_indexs = np.arange(len(pcd_filtered.points))

    [planes, ground_indexs] = pcd_filtered.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    non_ground_indexs = list(set(all_indexs) - set(ground_indexs))
    
    pcd_non_ground = pcd_filtered.select_by_index(non_ground_indexs)
    
    result = np.concatenate((np.asarray(pcd_non_ground.points), np.asarray(pcd_non_ground.normals)[:, 0].reshape(-1, 1)), axis=1)

    result.astype(np.float32).tofile(str(file_name))
    
    bin2ply(file_name.parent / "{}.ply".format(file_name.stem), result)
    
    running_time = time.process_time() - start_time
    np.save(file_name.parent / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "voxel_size": voxel_size, "ransac_n": ransac_n, "distance_threshold": distance_threshold, "num_iterations": num_iterations, "time": time.time(), "running_time": running_time})
    
def save_compression_lidar_data(file_name, save_dir):
    import DracoPy
    start_time = time.process_time()
    
    points_xyzi = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
    colors = np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1)
    colors = (colors * 255).astype(np.uint8)
    binary = DracoPy.encode(points_xyzi[:, :3], colors = colors, preserve_order = True)
    
    buffer_bin = np.frombuffer(binary, dtype=np.uint8)
    buffer_bin.tofile(save_dir / "{}.drc".format(file_name.stem))
    
    compressed_drc = DracoPy.decode(np.fromfile(save_dir / "{}.drc".format(file_name.stem), dtype=np.uint8).tobytes())
    compressed_points_xyzi = np.concatenate((compressed_drc.points, compressed_drc.colors[:, 0].reshape(-1, 1) / 255), axis=1)
    
    compressed_points_xyzi.astype(np.float32).tofile(save_dir / "{}.bin".format(file_name.stem))
    
    bin2ply(save_dir / "{}.ply".format(file_name.stem), compressed_points_xyzi)
    
    compressed_size = Path(save_dir / "{}.drc".format(file_name.stem)).stat().st_size
    
    point_count = compressed_points_xyzi.shape[0]

    running_time = time.process_time() - start_time
    np.save(save_dir / "{}_log.npy".format(file_name.stem), {"file_name": str(file_name), "compressed_size": compressed_size, "point_count": point_count, "time": time.time(), "running_time": running_time})


def save_packet_data(file_name, packet_mode, packet_size, sector_size, timestamp, save_dir):
    packet_size_list = []
    if packet_mode == 1:
        packet_size_list = np.full((sector_size,), packet_size).astype(int)
    else:
        npy_log = np.load(file_name.parent / "{}_log.npy".format(file_name.stem), allow_pickle=True).item()
        compressed_size = npy_log["compressed_size"]
        point_count = npy_log["point_count"]
        bytes_per_point = compressed_size / point_count
        
        point_xyzis = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        packet_size_list = get_point_count_by_sector_size(point_xyzis, sector_size)
        packet_size_list = np.round(np.asarray(packet_size_list) * bytes_per_point).astype(int)
    
    np.save(save_dir / "{}.npy".format(file_name.stem), {"file_name": str(file_name), "packet_size_list": packet_size_list, "timestamp": timestamp})

    

def get_point_count_by_sector_size(points_xyzi, sector_size):
    sector_list = range(sector_size)
    sector_range = 360 / sector_size
    points_a = np.arctan(points_xyzi[:, 1] / points_xyzi[:, 0]) / math.pi * 180
    points_xyzia = np.concatenate((points_xyzi, points_a.reshape(-1, 1)), axis = 1)

    points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] > 0)), 4] = 90.0
    points_xyzia[np.where((points_xyzia[:, 0] == 0.0) & (points_xyzia[:, 1] <= 0)), 4] = -90.0
    points_xyzia[np.where(points_xyzia[:, 0] < 0), 4] += 180
    points_xyzia[np.where(points_xyzia[:, 4] < 0), 4] += 360

    result = []
    for sector in sector_list:
        result.append(points_xyzi[np.where((points_xyzia[:, 4] >= sector * sector_range) & (points_xyzia[:, 4] < (sector + 1) * sector_range))].shape[0])
    return result

def save_label_data_carla(file_name, datapoints):
    with open(file_name, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)


def save_calibration_matrices(transform, file_name, intrinsic_mat):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)

    camera_transform = transform[0]
    lidar_transform = transform[1]
    # pitch yaw rool
    b = math.radians(lidar_transform.rotation.pitch-camera_transform.rotation.pitch)
    x = math.radians(lidar_transform.rotation.yaw-camera_transform.rotation.yaw)
    a = math.radians(lidar_transform.rotation.roll-lidar_transform.rotation.roll)
    R0 = np.identity(3)

    TR = np.array([[math.cos(b) * math.cos(x), math.cos(b) * math.sin(x), -math.sin(b)],
                   [-math.cos(a) * math.sin(x) + math.sin(a) * math.sin(b) * math.cos(x),
                    math.cos(a) * math.cos(x) + math.sin(a) * math.sin(b) * math.sin(x), math.sin(a) * math.cos(b)],
                   [math.sin(a) * math.sin(x) + math.cos(a) * math.sin(b) * math.cos(x),
                    -math.sin(a) * math.cos(x) + math.cos(a) * math.sin(b) * math.sin(x), math.cos(a) * math.cos(b)]])
    TR_velodyne = np.dot(TR, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

    TR_velodyne = np.dot(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]), TR_velodyne)

    '''
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    '''
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(file_name, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)


def save_rgb_image(file_name, image):
    im = Image.fromarray(image)
    im.save(file_name)


def save_location(file_name, location):
    np.save(file_name, location)

def save_location_txt(file_name, location):
    with open(file_name, "a+") as f:
        f.write(json.dumps(np.asarray(location).tolist()))

def save_ply_by_bin(file_name, point_cloud):
    points_xyzi = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    points_xyzi = np.reshape(points_xyzi, (int(points_xyzi.shape[0] / 4), 4))
    bin2ply(file_name, points_xyzi.copy())

def bin2ply(file_name, points_xyzi):
    points_xyzi[:, 1] = -points_xyzi[:, 1]
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points_xyzi[:, :3])
    point_list.colors = o3d.utility.Vector3dVector(get_point_color(points_xyzi[:, 3]))
    o3d.io.write_point_cloud(str(file_name), point_list)

def get_point_color(points_i):
    VIRIDIS = np.array(cm.get_cmap('plasma').colors)
    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
    intensity_col = 1.0 - np.log(points_i) / np.log(np.exp(-0.004 * 100))
    return np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]


def save_groundplanes(planes_fname, roll, pitch, lidar_height):
    from math import cos, sin
    """ Saves the groundplane vector of the current frame.
        The format of the ground plane file is first three lines describing the file (number of parameters).
        The next line is the three parameters of the normal vector, and the last is the height of the normal vector,
        which is the same as the distance to the camera in meters.
    """
    # Since measurements are in degrees, convert to radians
    # print("save_groundplanes: {}, {}, {}".format(roll, pitch, lidar_height))
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    # Rotate normal vector (y) wrt. pitch and yaw
    normal_vector = [cos(pitch)*sin(roll),
                     -cos(pitch)*cos(roll),
                     sin(pitch)
                     ]
    normal_vector = map(str, normal_vector)
    with open(planes_fname, 'w') as f:
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        f.write("{} {}\n".format(" ".join(normal_vector), lidar_height))


def save_semantic3d(xyzirgb_fname, label_fname, lidar_fname, semantic_lidar_measurement):
    lidar_xyzis = np.fromfile(lidar_fname, dtype=np.float32, count=-1).reshape([-1, 4])
    txt_result, label_result = "", ""
    for i, detection in enumerate(semantic_lidar_measurement):
        #print(detection)
        if lidar_xyzis.shape[0] > i:
            intensity = lidar_xyzis[i, 3]
        else:
            if lidar_xyzis.shape[0] > 0:
                intensity = lidar_xyzis[-1, 3]
            else:
                intensity = -1
                
        semantic3d_class_id = carla2semantic3d(detection.object_tag)
        rgb = get_rgb_by_class_id(semantic3d_class_id)
        txt_result += "{} {} {} {} {} {} {} \n".format(detection.point.x, -detection.point.y, detection.point.z, intensity, rgb[0], rgb[1], rgb[2])
        label_result += "{} \n".format(semantic3d_class_id)

    with open(xyzirgb_fname, 'w') as f:
        f.write(txt_result)
    
    with open(label_fname, 'w') as f:
        f.write(label_result)


def carla2semantic3d(class_id):
    # semantic-8 is a benchmark for classification with 8 class labels, namely {1: man-made terrain, 2: natural terrain, 3: high vegetation, 4: low vegetation, 5: buildings, 6: hard scape, 7: scanning artefacts, 8: cars
    # carla semantic tag: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
    class_maps = {0: 0, 1: 5, 2: 6, 3: 0, 4: 0, 5: 7, 6: 1, 7: 0, 8: 1, 9: 3, 10: 8, 11: 6,12: 7, 13: 0, 14: 1, 15: 6, 16: 1, 17: 6, 18: 7, 19: 0, 20: 0, 21: 2, 22: 2}
    if class_id in class_maps.keys():
        return class_maps[class_id]
    else:
        return 0


def get_rgb_by_class_id(class_id):
    rgb_maps = {0: [0, 0, 0], 1: [128, 64, 128], 2: [145, 170, 100], 3: [107, 142, 35], 4: [40, 60, 150],
                5: [70, 70, 70], 6: [102, 102, 156], 7: [220, 220, 0], 8: [0, 0, 142]}
    if class_id in rgb_maps.keys():
        return rgb_maps[class_id]
    else:
        return [0, 0, 0]