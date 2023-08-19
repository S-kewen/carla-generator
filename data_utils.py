import sys
import glob
import os
import numpy as np
from numpy.linalg import inv
from config import cfg_from_yaml_file
from DataDescriptor import KittiDescriptor, CarlaDescriptor
from image_converter import depth_to_array, to_rgb_array
import math
from visual_utils import draw_3d_bounding_box
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

cfg = cfg_from_yaml_file("./config.yaml")

MAX_RENDER_DEPTH_IN_METERS = cfg["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MIN_VISIBLE_VERTICES_FOR_RENDER"]
MAX_OUT_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
MIN_BBOX_AREA_IN_PX = cfg["FILTER_CONFIG"]["MIN_BBOX_AREA_IN_PX"]
MAX_OCCLUDED = cfg["FILTER_CONFIG"]["MAX_OCCLUDED"]
MIN_OBJECT_POINT_COUNT = cfg["FILTER_CONFIG"]["MIN_OBJECT_POINT_COUNT"]

WINDOW_WIDTH = cfg["FILTER_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = cfg["FILTER_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]
RGB_IMAGE_HEIGHT = cfg["FILTER_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]
RGB_IMAGE_WIDTH = cfg["FILTER_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]

def objects_filter(data, limit_fov=True):
    # FILTER OBJECT
    environment_objects = data["environment_objects"]
    agents_data = data["agents_data"]
    actors = data["actors"]
    actors = [x for x in actors if x.type_id.find("vehicle") != -1 or x.type_id.find("walker") != -1]
    for agent, dataDict in agents_data.items():
        intrinsic = dataDict["intrinsic"]
        extrinsic = dataDict["extrinsic"]
        sensors_data = dataDict["sensor_data"]
        rgb_transform = dataDict["rgb_transform"]
        kitti_datapoints = []
        carla_datapoints = []
        occlude_list = []
        bbox2d_area_list = []
        
        rgb_image = to_rgb_array(sensors_data[0])
        image = rgb_image.copy()
        depth_data = sensors_data[1]
        lidar_data = sensors_data[2]

        data["agents_data"][agent]["visible_environment_objects"] = []
        for obj in environment_objects:
            kitti_datapoint, carla_datapoint, occlude, bbox2d_area = is_visible_by_bbox(agent, obj, image, depth_data, intrinsic, extrinsic, lidar_data, rgb_transform, limit_fov)
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_environment_objects"].append(obj)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)
                occlude_list.append(occlude)
                bbox2d_area_list.append(bbox2d_area)

        data["agents_data"][agent]["visible_actors"] = []

        for act in actors:
            kitti_datapoint, carla_datapoint, occlude, bbox2d_area = is_visible_by_bbox(agent, act, image, depth_data, intrinsic, extrinsic, lidar_data, rgb_transform, limit_fov)
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_actors"].append(act)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)
                occlude_list.append(occlude)
                bbox2d_area_list.append(bbox2d_area)

        # sort dontcare objects to the end
        kitti_datapoints = sort_data_points(kitti_datapoints)
        
        data["agents_data"][agent]["rgb_image"] = image
        data["agents_data"][agent]["kitti_datapoints"] = kitti_datapoints
        data["agents_data"][agent]["carla_datapoints"] = carla_datapoints
        data["agents_data"][agent]["occlude_list"] = occlude_list
        data["agents_data"][agent]["bbox2d_area_list"] = bbox2d_area_list
    return data

def sort_data_points(kitti_datapoints):
    result = []
    for dp in kitti_datapoints:
        if dp is not None:
            items = str(dp).split(" ")
            if len(items) == 15 and items[0] != "DontCare":
                result.append(dp)
    
    for dp in kitti_datapoints:
        if dp is not None:
            items = str(dp).split(" ")
            if len(items) == 15 and items[0] == "DontCare":
                result.append(dp)
                
    return result
        

def get_truncated(bbox, width, height) -> float: # using rgb image height, width and the bbox to cal truncated
    left, top, right, bottom = bbox
    if left == right or top == bottom:
        return 1.0
    _left = min(max(0, left), width-1)
    _top = min(max(0, top), height-1)
    _right = max(min(width-1, right), 0)
    _bottom = max(min(height-1, bottom), 0)
    result = min(1.0 - (_right - _left) / (right - left) * (_bottom - _top) / (bottom - top), 1.0)
    if result>1.0:
        result = 1.0
    if result < 0:
        result = 0.0
    return result

def _complete_homography_matrix(matrix) -> np.array:
    if len(matrix.shape) == 1:
        matrix = np.expand_dims(matrix, axis=0)
    if matrix.shape[1] == 3:
        shift = np.ones((matrix.shape[0], 1))
        return np.concatenate([matrix, shift], axis=1)
    elif matrix.shape[1] == 4:
        return matrix
    else:
        raise NotImplementedError


def world_to_sensor(cords, sensor_transform, homography=False) -> np.array:
    assert isinstance(cords, np.ndarray), type(cords)
    #assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    cords = _complete_homography_matrix(cords)
    sensor_world_matrix = sensor_transform.get_matrix()
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, np.transpose(cords))
    if homography:
        return np.transpose(sensor_cords)
    else:
        return np.transpose(sensor_cords)[:, :3]
    
    
    
    
class _baseLabel(object):
    def __init__(self, x=0, y=0, z=0, h=0, w=0, l=0, ry=0):
        self.x = x
        self.y = y
        self.z = z
        self.h = h
        self.w = w
        self.l = l
        self.ry = ry

    def get_location(self):
        return np.array([self.x, self.y, self.z])
    def get_extent(self):
        return np.array([self.h, self.w, self.l])
    def get_heading(self):
        return self.ry

    def __str__(self):
        return str(self.__dict__)
    __repr__ = __str__


class _baseLabelList(object):
    def __init__(self):
        self.labels = []
        self.__iterater = 0
    def __eq__(self, o):
        return self.labels == o.labels if type(self) == type(o) else TypeError
    def __add__(self, x):
        if isinstance(x, _baseLabel):
            self.labels.append(x)
            return self
        elif isinstance(x, _baseLabelList):
            self.labels.extend(x.labels)
            return self
        else:
            raise NotImplementedError
    def __getitem__(self, index):
        # assert isinstance(index, (int, np.int)), type(index)
        return self.labels[index]
    def __iter__(self):
        return self
    def __next__(self):
        if self.__iterater < len(self.labels):
            self.__iterater += 1
            return self.labels[self.__iterater - 1]
        else:
            raise StopIteration
    def __len__(self):
        return len(self.labels)
    def __str__(self):
        return str(self.labels)
    __repr__ = __str__


class CarlaLabel(_baseLabel):
    def __init__(self, x, y, z, h, w, l, ry):
        super(CarlaLabel, self).__init__(x, y, z, h, w, l, ry)

    def to_str(self):
        return "%.2f %.2f %.2f %.2f %.2f %.2f %.2f" %(self.x, self.y, self.z, self.h, self.w, self.l, self.ry)

    @staticmethod
    def fromlist(list):
        assert len(list) == 7, len(list)
        label = CarlaLabel(*list)
        return label
    
class CarlaLabelList(_baseLabelList):
    def __init__(self):
        super(CarlaLabelList, self).__init__()

    @staticmethod
    def fromarray(array):
        assert isinstance(array, np.ndarray), type(array)
        _labellist = CarlaLabelList()
        for row in array:
            _labellist = _labellist + CarlaLabel.fromlist(row)
        return _labellist
    
    
class _baseCustomSensor:
    import weakref
    def __init__(self, world, attached, transform, sensor_type, **params):
        self.world = world
        self.parent = attached
        self.offset = transform
        self.sensor = self.__init_sensor(transform, attached, sensor_type, **params)
        self.attributes = self.sensor.attributes
        self.id = self.sensor.id
        self.is_alive = self.sensor.is_alive
        self.semantic_tags = self.sensor.semantic_tags
        self.data = None
        self.retrive = True
        self.tick = 0

    def __init_sensor(self, transform, attached, sensor_type, **params):
        sensor_bp = self.world.get_blueprint_library().find(sensor_type)
        for key in params:
            sensor_bp.set_attribute(key, str(params[key]))
        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=attached)
        weak_self = weakref.ref(self)
        sensor.listen(lambda data: weak_self().__set_data(weak_self, data))
        return sensor

    @staticmethod
    def __set_data(weak_self, data):
        self = weak_self()
        if self.retrive:
            self.tick += 1
            self.data = data
            self.retrive = False

    def save_data(self):
        raise NotImplementedError

    def get_transform(self):
        return self.sensor.get_transform()
    def get_location(self):
        return self.sensor.get_location()
    def get_world(self):
        return self.world
    def destroy(self):
        self.sensor.stop()
        return self.sensor.destroy()
    def get_offset(self):
        return self.offset
    
def get_alpha(rgb_transform, obj) -> float:
    ego_heading = rgb_transform.rotation.yaw
    
    other_vehicles_location = []
    other_vehicles_hwl = []
    other_vehicles_heading = []
    
    
    obj_transform = obj.transform if isinstance(obj, carla.EnvironmentObject) else obj.get_transform()
    
    location = obj_transform.location
    other_vehicles_location.append(np.array([[location.x, location.y, location.z]]))
    
    extent = obj.bounding_box.extent
    other_vehicles_hwl.append(np.array([[extent.z * 2, extent.y * 2, extent.x * 2]]))
    
    heading = obj_transform.rotation.yaw
    other_vehicles_heading.append(np.array([[np.deg2rad(heading - ego_heading)]]))
        
    other_vehicles_location = np.concatenate(other_vehicles_location, axis=0)
    other_vehicles_hwl = np.concatenate(other_vehicles_hwl, axis=0)
    other_vehicles_heading = np.concatenate(other_vehicles_heading, axis=0)

    other_vehicles_location = world_to_sensor(other_vehicles_location, rgb_transform)
    labels_xyzhwlry = np.concatenate([other_vehicles_location, other_vehicles_hwl, other_vehicles_heading], axis=1)
    labels = CarlaLabelList.fromarray(labels_xyzhwlry)
    # print("labels: {}".format(labels))
    x = labels[0].y
    z = labels[0].x
    ry = labels[0].ry
    
    theta = -np.arcsin(x/np.sqrt(x**2+z**2))
    result = ry - theta
    result = max(result, -math.pi)
    result = min(result, math.pi)
    return result

def calculate_occlusion(bbox, agent, depth_map):
    """Calculate the occlusion value of a 2D bounding box.
    Iterate through each point (pixel) in the bounding box and declare it occluded only
    if the 4 surroinding points (pixels) are closer to the camera (by using the help of depth map)
    than the actual distance to the middle of the 3D bounding boxe and some margin (the extent of the object)
    """
    
    bbox_3d_mid = np.mean(np.array(bbox)[:, 2])
    min_x, min_y, max_x, max_y = calc_projected_2d_bbox(bbox)
    area = ((max_x-min_x) * (max_y-min_y))
    if area == 0:
        return 3, -1
    
    height, width, length = agent.bounding_box.extent.z, agent.bounding_box.extent.x, agent.bounding_box.extent.y

    #depth_margin should depend on the rotation of the object but this solution works fine
    depth_margin = np.max([2 * width, 2 * length])
    
    # [skewen]: old code, too slow
    #is_occluded = []
    # for x in range(int(min_x), int(max_x)):
    #     for y in range(int(min_y), int(max_y)):
    #         is_occluded.append(point_is_occluded(
    #             (y, x), bbox_3d_mid - depth_margin, depth_map))
    # occlusion = ((float(np.sum(is_occluded))) / area)
    
    
    # [skewen]: Accelerate calculations with numpy
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(max_x, WINDOW_WIDTH)
    max_y = min(max_y, WINDOW_HEIGHT)
    
    xs = np.arange(min_x, max_x)
    ys = np.arange(min_y, max_y)
    points_xys = get_all_combinations(xs, ys)
    
    if points_xys.shape[0] > 0:
        occluded_area = np_point_is_occluded(points_xys, bbox_3d_mid - depth_margin, depth_map)
    else:
        occluded_area = 0

    occlusion = occluded_area / area

    #discretize the 0–1 occlusion value into KITTI’s {0,1,2,3} labels by equally dividing the interval into 4 parts
    occlusion_level = np.digitize(occlusion, bins=[0.15, 0.50, 0.75])

    return occlusion_level, occlusion

def get_all_combinations(x, y):
    from itertools import product
    return np.array(list(product(x, y)))  

def np_point_is_occluded(points_xys, vertex_depth, depth_map):
    # result = points_xys[np.where((points_xys[:, 1] >= 0) & (points_xys[:, 1] < WINDOW_HEIGHT) & (points_xys[:, 0] >= 0) & (points_xys[:, 0] < WINDOW_WIDTH))]
    result = points_xys[np.where(depth_map[points_xys[:, 1], points_xys[:, 0]] < vertex_depth)]
    return result.shape[0]
    

def point_is_occluded(point, vertex_depth, depth_image):
    y, x = map(int, point)
    from itertools import product
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy + y, dx + x)):
            if depth_image[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    return all(is_occluded)
    
def point_in_canvas(pos):
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False

def is_visible_by_bbox(agent, obj, rgb_image, depth_data, intrinsic, extrinsic, lidar_data, rgb_transform=None, limit_fov=True):
    # [skewen]: here is going to filter the object.
    if agent.id == obj.id:
        return None, None, None, None
    
    obj_transform = obj.transform if isinstance(obj, carla.EnvironmentObject) else obj.get_transform()
    obj_bbox = obj.bounding_box
    if isinstance(obj, carla.EnvironmentObject):
        vertices_pos2d = bbox_2d_from_agent(intrinsic, extrinsic, obj_bbox, obj_transform, 0)
    else:
        vertices_pos2d = bbox_2d_from_agent(intrinsic, extrinsic, obj_bbox, obj_transform, 1)
    depth_image = depth_to_array(depth_data) # raw depth image to gray image
    
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(vertices_pos2d, depth_image, limit_fov)
    
    # DontCare -1 -1 -10 566.39 168.89 585.07 184.56 -1 -1 -1 -1000 -1000 -1000 -10
    
    if num_visible_vertices == 0: #num_in_canvas < 2 or
        # object does not appear in RGB image at all
        return None, None, None, None
    
    kitti_data = KittiDescriptor()
    carla_data = CarlaDescriptor()
    
    bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
    kitti_data.set_bbox(bbox_2d)
    
    bbox2d_area = calc_bbox2d_area(bbox_2d)
    
    if limit_fov:
        occluded_level_depth_image, occluded = calculate_occlusion(vertices_pos2d, agent, depth_image)
    else:
        occluded_level_depth_image, occluded = 0, -1
        
    obj_tp = obj_type(obj)

    ext = obj.bounding_box.extent

    midpoint = midpoint_from_agent_location(obj_transform.location, extrinsic)

    rotation_y = get_relative_rotation_y(agent.get_transform().rotation, obj_transform.rotation) % math.pi

    calibs = {"R0_rect": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "Tr_velo_to_cam": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]} #[skewen]: fixed calib config
    # print("calibs: {}".format(calibs))
    #calibs = read_calib_file(r"D:\mnt\sda\kittiGenerator\output\20221110_2_50_0_454\object\training\calib\000000.txt")

    h, w, l = 2*ext.z, 2*ext.x, 2*ext.y
    _x, _y, _z = [float(x) for x in midpoint][0:3]

    if obj_tp == "Pedestrian":
        # Since the midpoint/location of the pedestrian is in the middle of the agent, while for car it is at the bottom
        # we need to subtract the bbox extent in the height direction when adding location of pedestrian.
        _z -= h

    x, y, z = _y, -_z, _x # make coordinate transformation
    point_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
    points_xyzi = [[point[0], -point[1], point[2], point[3]] for point in point_cloud]
    points_xyzi = np.array(points_xyzi).astype(np.float32).reshape(-1, 4)

    object_point_count = in_hull_count(points_xyzi[:, :3], compute_box_3d(calibs, h, w, l, x, y, z, rotation_y))
    
    if object_point_count == 0:
        return None, None, None, None
    
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER and bbox2d_area >= MIN_BBOX_AREA_IN_PX and MAX_OCCLUDED >= occluded and object_point_count>=MIN_OBJECT_POINT_COUNT:
        #print("object_point_count={}".format(object_point_count))
        if limit_fov:
            truncated = get_truncated(bbox_2d, RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT)
        else:
            truncated = 0
            
        if num_visible_vertices >= 6:
            occluded_level_visible_vertice = 0
        elif num_visible_vertices >= 4:
            occluded_level_visible_vertice = 1
        else:
            occluded_level_visible_vertice = 2
        
        occluded_level = max(occluded_level_depth_image, occluded_level_visible_vertice)
            
        alpha = get_alpha(rgb_transform, obj)
        
        velocity = "0 0 0" if isinstance(obj, carla.EnvironmentObject) else\
            "{} {} {}".format(obj.get_velocity().x, obj.get_velocity().y, obj.get_velocity().z)
        acceleration = "0 0 0" if isinstance(obj, carla.EnvironmentObject) else \
            "{} {} {}".format(obj.get_acceleration().x, obj.get_acceleration().y, obj.get_acceleration().z)
        angular_velocity = "0 0 0" if isinstance(obj, carla.EnvironmentObject) else\
            "{} {} {}".format(obj.get_angular_velocity().x, obj.get_angular_velocity().y, obj.get_angular_velocity().z)

        draw_3d_bounding_box(rgb_image, vertices_pos2d)

        
        kitti_data.set_type(obj_tp)
        kitti_data.set_truncated(truncated)
        kitti_data.set_occlusion(occluded_level)
        kitti_data.set_alpha(alpha)
        # 1    alpha        Observation angle of object, ranging [-pi..pi]
        

        kitti_data.set_3d_object_dimensions(ext)

        kitti_data.set_3d_object_location(midpoint)
        kitti_data.set_rotation_y(rotation_y)
        # print("kitti_data({})： {}".format(len(str(kitti_data).split()), str(kitti_data).split()))
        
        carla_data.set_type(obj_tp)
        carla_data.set_velocity(velocity)
        carla_data.set_acceleration(acceleration)
        carla_data.set_angular_velocity(angular_velocity)
    # else:
    #     # dontcare
    #     if (num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER and MAX_OCCLUDED >= occluded and object_point_count>=MIN_OBJECT_POINT_COUNT and  bbox2d_area < MIN_BBOX_AREA_IN_PX ):
    #         print("bbox2d_area: {}".format(bbox2d_area))
    #     print("FILTER DontCare OBJECT: {}".format(str(obj)))
    return kitti_data, carla_data, occluded, bbox2d_area


def obj_type(obj):
    if isinstance(obj, carla.EnvironmentObject):
        if obj.type == "Vehicles":
            return "Car"
        elif obj.type == "Pedestrians":
            return "Pedestrian"
        else:
            return obj.type
    else:
        if obj.type_id.find('walker') != -1:
            return 'Pedestrian'
        if obj.type_id.find('vehicle') != -1:
            return 'Car'
        return obj.type_id


def get_relative_rotation_y(agent_rotation, obj_rotation):
    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return -degrees_to_radians(rot_agent - rot_car)


def bbox_2d_from_agent(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    bbox = vertices_from_extension(obj_bbox.extent)
    
    # env object and actor object have different function to get the bbox
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        bbox = transform_points(bbox_transform, bbox)
    else:
        box_location = carla.Location(obj_bbox.location.x-obj_transform.location.x,
                                      obj_bbox.location.y-obj_transform.location.y,
                                      obj_bbox.location.z-obj_transform.location.z)
        box_rotation = obj_bbox.rotation
        bbox_transform = carla.Transform(box_location, box_rotation)
        bbox = transform_points(bbox_transform, bbox)
    bbox = transform_points(obj_transform, bbox)
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def vertices_from_extension(ext):
    return np.array([
        [ext.x, ext.y, ext.z],  # Top left front
        [- ext.x, ext.y, ext.z],  # Top left back
        [ext.x, - ext.y, ext.z],  # Top right front
        [- ext.x, - ext.y, ext.z],  # Top right back
        [ext.x, ext.y, - ext.z],  # Bottom left front
        [- ext.x, ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])


def transform_points(transform, points):
    points = points.transpose()
    # [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]  (4,8)
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    points = np.mat(transform.get_matrix()) * points
    return points[0:3].transpose()


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    vertices_pos2d = []
    for vertex in bbox:
        pos_vector = vertex_to_world_vector(vertex)
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        vertex_depth = pos2d[2]
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def calculate_occlusion_stats(vertices_pos2d, depth_image, limit_fov=True):
    num_visible_vertices, num_vertices_outside_camera = 0, 0
    # the vertex is in the camera's view or not
    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # print("y_2d, x_2d, vertex_depth", y_2d, x_2d, vertex_depth)
        in_canvas = point_in_canvas((y_2d, x_2d))
        # the vertex is out the max depth (default is 100) or not
        if limit_fov:
            if MAX_RENDER_DEPTH_IN_METERS >= vertex_depth > 0 and in_canvas: #  and in_canvas
                is_occluded = point_is_occluded((y_2d, x_2d), vertex_depth, depth_image)
                if not is_occluded:
                    num_visible_vertices += 1
            else:
                num_vertices_outside_camera += 1
        else:
            if MAX_RENDER_DEPTH_IN_METERS >= abs(vertex_depth) > 0: #  and in_canvas
                    num_visible_vertices += 1
            else:
                num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def midpoint_from_agent_location(location, extrinsic_mat):
    midpoint_vector = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


def proj_to_camera(pos_vector, extrinsic_mat):
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def proj_to_2d(camera_pos_vector, intrinsic_mat):
    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    # normalize the 2D points
    pos2d = np.array([
        pos2d[0] / pos2d[2],
        pos2d[1] / pos2d[2],
        pos2d[2]
    ])
    return pos2d


def filter_by_distance(data_dict, dis):
    environment_objects = data_dict["environment_objects"]
    actors = data_dict["actors"]
    for agent, _ in data_dict["agents_data"].items():
        data_dict["environment_objects"] = [obj for obj in environment_objects if
                                            distance_between_locations_new(obj.transform.location, agent.get_location())
                                            < dis]
        data_dict["actors"] = [act for act in actors if
                               distance_between_locations_new(act.get_location(), agent.get_location()) < dis]


def distance_between_locations(location1, location2):
    return math.sqrt(pow(location1.x-location2.x, 2)+pow(location1.y-location2.y, 2))

def distance_between_locations_new(location1, location2):
    return location1.distance(location2)

def calc_projected_2d_bbox(vertices_pos2d):
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_x, min_y, max_x, max_y]


def degrees_to_radians(degrees):
    return degrees * math.pi / 180

def calc_bbox2d_area(bbox_2d):
    """ Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple 
    """
    xmin, ymin, xmax, ymax = bbox_2d
    return (ymax - ymin) * (xmax - xmin)

def in_hull_count(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    results = hull.find_simplex(p) >= 0
    return np.sum(results!=0)

def get_hull_points(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    results = hull.find_simplex(p[:, :3]) >= 0
    return p[np.where(results!=0)]

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(calibs, h, w, l, x, y, z, ry):
    ry = roty(ry)
    x_corners = [l / 2, l / 2, -l / 2, -
                    l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2,
                    w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(ry, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    ro = calibs["R0_rect"]
    ro = np.reshape(ro, [3, 3])
    c2v = inverse_rigid_trans(np.reshape(calibs["Tr_velo_to_cam"], [3, 4]))
    return project_rect_to_velo(np.transpose(corners_3d), ro, c2v)

def project_rect_to_velo(pts_3d_rect, ro, c2v):
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, ro)
    return project_ref_to_velo(pts_3d_ref, c2v)

def project_rect_to_ref(pts_3d_rect, ro):
    """ Input and Output are nx3 points """
    return np.transpose(np.dot(np.linalg.inv(ro), np.transpose(pts_3d_rect)))

def project_ref_to_velo(pts_3d_ref, c2v):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(c2v))

def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def read_calib_file(filepath):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data