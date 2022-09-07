import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle


class CustomHighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded with respect to the 
    parameterization of the environment.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "offroad_terminal": False,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,                 # The reward received when colliding with a vehicle.
            "lane_rewards": [0, 0.033, 0.067, 0.1],  # Reward for each lane, from left to right
            "desired_speed": 25,                      
            "desired_speed_range": [20, 30],
            "speed_reward": 0.4,                     # The reward received when driving exactly at the desired speed, linearly
                                                     # mapped to zero for other speeds according to config["desired_speed_range"].
            "lane_change_reward": -0.1,              # The reward received at each lane change action (proxy for lateral jerk).
            "acceleration_change_reward": -0.1,      # The reward received at each acceleration change (proxy for jerk).
            "minimum_follow_distance": 0.15,         # Any distance lower than this will be penalized. At this distance, the
                                                     # additional reward is 0. It decreases linearly to
                                                     # config["close_follow_reward"] as the distance decreases to 0.
            "close_follow_reward": -1.0
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.prev_action = 1

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
           
        reward = 0.
        # add collision reward
        reward += self.config["collision_reward"] * self.vehicle.crashed
        # add lane reward
        reward += self.config["lane_rewards"][lane]
        # add speed reward
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["desired_speed_range"], [0, 1])
        scaled_desired_speed = utils.lmap(self.config["desired_speed"], self.config["desired_speed_range"], [0, 1])
        if 0 < scaled_speed < 1:
            if scaled_desired_speed > scaled_speed:
                reward += scaled_speed * self.config["speed_reward"] / scaled_desired_speed
            else:
                reward += (1 - scaled_speed) * self.config["speed_reward"] / (1 - scaled_desired_speed)
        # add lane change reward
        if action in [0, 2]:
            reward += self.config["lane_change_reward"]
        # add acceleration change reward
        if (action in [0,1,2] and self.prev_action not in [0,1,2]) or (action in [3,4] and action != self.prev_action):
            reward += self.config["acceleration_change_reward"]
        # add close follow reward -- this is hacky
        obs = self.observation_type.observe()
        obs = obs[1:,:3]
        follow_distance = np.inf
        for i in range(len(obs)):
            if np.abs(obs[i,2]) < 1e-2 and obs[i,1] > 0. and np.isclose(obs[i,0], 1):
                follow_distance = obs[i,1]
                break
        if follow_distance < self.config["minimum_follow_distance"]:
            reward += (self.config["minimum_follow_distance"] - follow_distance) * self.config["close_follow_reward"] / \
                self.config["minimum_follow_distance"]
            
        reward = utils.lmap(reward,
                          [self.config["collision_reward"]*0 + self.config["lane_change_reward"] + \
                            self.config["acceleration_change_reward"] + self.config["close_follow_reward"] + \
                            np.min(self.config["lane_rewards"]),
                           self.config["speed_reward"] + np.max(self.config["lane_rewards"])],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='customhighway-v0',
    entry_point='custom_highway_env:CustomHighwayEnv',
)
