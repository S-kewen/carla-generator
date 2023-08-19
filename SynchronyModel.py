import queue
import sys
import random
import glob
import os
import numpy as np
from random import choice
from config import config_to_trans
from data_utils import camera_intrinsic, filter_by_distance, distance_between_locations_new
from random import choice
try:
    sys.path.append(glob.glob('.../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class SynchronyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager()
        self.init_settings = None
        self.frame = None
        self.actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
        self.data = {"sensor_data": {}, "environment_data": None}
        self.agent_names = {}
        self.agent_configs = {}
        
    def set_synchrony(self):
        self.init_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1/self.cfg["FILTER_CONFIG"]["LIDAR"]["ATTRIBUTE"]["rotation_frequency"]
        self.world.apply_settings(settings)

    def setting_recover(self):
        for agent in self.actors["agents"]:
            if agent in self.actors["sensors"]:
                for sensor in self.actors["sensors"][agent]:
                    sensor.destroy()
            agent.destroy()
        batch = []
        for actor_id in self.actors["non_agents"]:
            batch.append(carla.command.DestroyActor(actor_id))
        for walker_id in self.actors["walkers"]:
            batch.append(carla.command.DestroyActor(walker_id))
        self.client.apply_batch_sync(batch)
        self.world.apply_settings(self.init_settings)

    def get_vehicle_spawn_points(self):
        vehicle_spawn_points = self.filter_spawn_points_by_distance(self.world.get_map().get_spawn_points(), self.get_agent_transform_list(), self.cfg["CARLA_CONFIG"]["VEHICLE_SPAWN_POINT_DISTANCE"])
        random.shuffle(vehicle_spawn_points)
        return vehicle_spawn_points

    def spawn_actors(self, vehicle_spawn_points):
        num_of_vehicles = self.cfg["CARLA_CONFIG"]["NUM_OF_VEHICLES"]
        num_of_walkers = self.cfg["CARLA_CONFIG"]["NUM_OF_WALKERS"]

        # create vehicle actors without sensor
        number_of_vehicle_spawn_points = len(vehicle_spawn_points)
        print("number_of_vehicle_spawn_points: ", number_of_vehicle_spawn_points)
        
        assert number_of_vehicle_spawn_points>=num_of_vehicles, "requested {} vehicles, but could only find {} spawn points".format(num_of_vehicles, number_of_vehicle_spawn_points)
        
        batch = []
        for n, transform in enumerate(vehicle_spawn_points):
            if n >= num_of_vehicles:
                break
            blueprints = self.world.get_blueprint_library().filter(choice(self.cfg["BLUEPRINTS"][self.cfg["CARLA_CONFIG"]["VEHICLE"]["BLUEPRINTS"]]))
            blueprints = sorted(blueprints, key=lambda bp: bp.id)
            
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform))

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                assert False, "batch sync error"
            else:
                self.actors["non_agents"].append(response.actor_id)

        # Generate pedestrian actors
        blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        spawn_points = []
        for i in range(num_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                continue
            else:
                self.actors["walkers"].append(response.actor_id)
        print("spawn {} vehicles and {} walkers".format(len(self.actors["non_agents"]),
                                                        len(self.actors["walkers"])))
        self.world.tick()

    def set_actors_route(self):
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_synchronous_mode(True)
        vehicle_actors = self.world.get_actors(self.actors["non_agents"])
        for vehicle in vehicle_actors:
            vehicle.set_autopilot(True, self.traffic_manager.get_port())

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        batch = []
        for i in range(len(self.actors["walkers"])):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(),
                                                  self.actors["walkers"][i]))
        controllers_id = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                controllers_id.append(response.actor_id)
        self.world.set_pedestrians_cross_factor(0.2)

        for con_id in controllers_id:
            # start walker
            self.world.get_actor(con_id).start()
            # set walk to random point
            destination = self.world.get_random_location_from_navigation()
            self.world.get_actor(con_id).go_to_location(destination)
            # max speed
            self.world.get_actor(con_id).set_max_speed(10)
            
    def get_spawn_point_by_distance(self, spawn_points, location, distance):
        for i, spawn_point in enumerate(spawn_points):
            if distance_between_locations_new(spawn_point.location, carla.Location(location[0], location[1], location[2])) <= distance:
                return spawn_points.pop(i)
        return None
    
    
    def get_spawn_point_count_by_distance(self, spawn_points, location, distance):
        result = 0
        for i, spawn_point in enumerate(spawn_points):
            if distance_between_locations_new(spawn_point.location, carla.Location(location[0], location[1], location[2])) <= distance:
                result += 1
        return result
            

    def spawn_agent(self, vehicle_spawn_points):
        # create LiDAR-vehicle
        for agent_name, agent_config in self.cfg["AGENT_CONFIG"].items():
            # generate vehicle
            if agent_config["ENABLE"]:
                vehicle_bp = random.choice(self.world.get_blueprint_library().filter(agent_config["BLUEPRINT"]))
                if isinstance(agent_config["TRANSFORM"], str) and agent_config["TRANSFORM"].lower() == "random":
                    assert len(vehicle_spawn_points) > 0, "No spawn points available, total number of spawn points: {}".format(len(vehicle_spawn_points))
                    transform = vehicle_spawn_points.pop()
                elif isinstance(agent_config["TRANSFORM"], dict): # specify location or specify area randomly spawn point
                    if "random" in agent_config["TRANSFORM"]:
                        # count = self.get_spawn_point_count_by_distance(vehicle_spawn_points, agent_config["TRANSFORM"]["random"], agent_config["TRANSFORM"]["distance"])
                        transform = self.get_spawn_point_by_distance(vehicle_spawn_points, agent_config["TRANSFORM"]["random"], agent_config["TRANSFORM"]["distance"])
                        assert transform is not None, "No spawn points available"
                    else:
                        trans_cfg = agent_config["TRANSFORM"]
                        transform = config_to_trans(trans_cfg)
                        
                agent = self.world.spawn_actor(vehicle_bp, transform)
                if agent_config["AUTOPILOT"]:
                    agent.set_autopilot(True, self.traffic_manager.get_port())
                self.actors["agents"].append(agent)
                self.agent_names[agent] = agent_name
                self.agent_configs[agent] = agent_config
                
                
                if "SENSOR_CONFIG" in agent_config:
                    # install sensor
                    self.actors["sensors"][agent] = []
                    for sensor_name, sensor_config in agent_config["SENSOR_CONFIG"].items():
                        sensor_bp = self.world.get_blueprint_library().find(sensor_config["BLUEPRINT"])
                        for attr, val in sensor_config["ATTRIBUTE"].items():
                            sensor_bp.set_attribute(attr, str(val))
                        trans_cfg = sensor_config["TRANSFORM"]
                        transform = carla.Transform(carla.Location(trans_cfg["location"][0], trans_cfg["location"][1], trans_cfg["location"][2]), carla.Rotation(trans_cfg["rotation"][0], trans_cfg["rotation"][1], trans_cfg["rotation"][2]))
                        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=agent)
                        self.actors["sensors"][agent].append(sensor)
        self.world.tick()
        return vehicle_spawn_points

    def sensor_listen(self):
        for agent, sensors in self.actors["sensors"].items():
            self.data["sensor_data"][agent] = []
            for sensor in sensors:
                q = queue.Queue()
                self.data["sensor_data"][agent].append(q)
                sensor.listen(q.put)

    def tick(self):
        ret = {"environment_objects": [], "actors": [], "agents_data": {}}
        self.frame = self.world.tick()

        if self.cfg["FILTER_CONFIG"]["CATCH_ENVIRONMENT_OBJECT"]:
            #ret["environment_objects"] = self.world.get_environment_objects(carla.CityObjectLabel.Any) # Vehicles Pedestrians Any
            environment_object_vehicles = self.world.get_environment_objects(carla.CityObjectLabel.Vehicles)
            environment_object_pedestrians = self.world.get_environment_objects(carla.CityObjectLabel.Pedestrians)
            ret["environment_objects"] = environment_object_vehicles + environment_object_pedestrians
            
        if self.cfg["FILTER_CONFIG"]["CATCH_ACTOR"]:
            ret["actors"] = self.world.get_actors()
            
        image_width = self.cfg["FILTER_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        image_height = self.cfg["FILTER_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]
        for agent, dataQue in self.data["sensor_data"].items():
            data = [self._retrieve_data(q) for q in dataQue]
            assert all(x.frame == self.frame for x in data)
            ret["agents_data"][agent] = {}
            ret["agents_data"][agent]["sensor_data"] = data
            ret["agents_data"][agent]["intrinsic"] = camera_intrinsic(image_width, image_height)
            ret["agents_data"][agent]["extrinsic"] = np.mat(self.actors["sensors"][agent][0].get_transform().get_matrix())
            ret["agents_data"][agent]["rgb_transform"] = self.actors["sensors"][agent][0].get_transform()
            ret["agents_data"][agent]["location"] = self.get_vehicle_location(agent)
            ret["agents_data"][agent]["agent_name"] = self.get_agent_name(agent)
        filter_by_distance(ret, self.cfg["FILTER_CONFIG"]["PRELIMINARY_FILTER_DISTANCE"])
        return ret

    def _retrieve_data(self, q):
        while True:
            data = q.get()
            if data.frame == self.frame:
                return data
    
    def get_vehicle_location(self, node):
        x, y, z, rx, ry, rz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if node is not None:
            transform = node.get_transform()
            x = transform.location.x
            y = transform.location.y
            z = transform.location.z
            rx = transform.rotation.roll
            ry = transform.rotation.pitch
            rz = transform.rotation.yaw
        else:
            print("warning !!! get_vehicle_location's node is None !! ")
        result = {'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz, 'timestamp': self.frame}
        # print(result)
        return result
    
    def get_agent_name(self, agent):
        return self.agent_names[agent]
    
    def hide_environment_objects(self):
        environment_objects = self.world.get_environment_objects(carla.CityObjectLabel.Vehicles) #Vehicles Pedestrians Any
        environment_object_ids = []
        for obj in environment_objects:
            # print("hide_environment_objects: {}".format(str(obj)))
            environment_object_ids.append(obj.id)

        self.world.enable_environment_objects(environment_object_ids, False) 
        print("cleared the all of vehicles in environment: {}".format(len(environment_objects)))
        
    
    def get_agent_transform_list(self):
        result = []
        for agent_name, agent_config in self.cfg["AGENT_CONFIG"].items():
            if agent_config["ENABLE"] and isinstance(agent_config["TRANSFORM"], dict) and "location" in agent_config["TRANSFORM"]:
                result.append(config_to_trans(agent_config["TRANSFORM"]))
        return result

    def filter_spawn_points_by_distance(self, spawn_points, agent_transform_list, distance):
        result = []
        for spawn_point in spawn_points:
            for agent_transform in agent_transform_list:
                if distance > distance_between_locations_new(spawn_point.location, agent_transform.location):
                    break
            else:
                result.append(spawn_point)
        return result