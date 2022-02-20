#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import traceback
import xml.etree.ElementTree as ET
import numpy.random as random

import py_trees

import carla

from haco.DIDrive_core.utils.planner import RoadOption
from haco.DIDrive_core.utils.simulator_utils.carla_utils import convert_waypoint_to_transform

# # pylint: disable=line-too-long
from haco.DIDrive_core.simulators.srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from haco.DIDrive_core.simulators.carla_data_provider import CarlaDataProvider
from haco.DIDrive_core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from haco.DIDrive_core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from haco.DIDrive_core.simulators.srunner.tools.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from haco.DIDrive_core.simulators.srunner.tools.route_manipulation import interpolate_trajectory, downsample_route
from haco.DIDrive_core.simulators.srunner.tools.py_trees_port import oneshot_behavior

from haco.DIDrive_core.simulators.srunner.scenarios.control_loss_new import ControlLossNew
from haco.DIDrive_core.simulators.srunner.scenarios.control_loss import ControlLoss
from haco.DIDrive_core.simulators.srunner.scenarios.follow_leading_vehicle_new import FollowLeadingVehicleNew
from haco.DIDrive_core.simulators.srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from haco.DIDrive_core.simulators.srunner.scenarios.change_lane import ChangeLane
from haco.DIDrive_core.simulators.srunner.scenarios.cut_in import CutIn
from haco.DIDrive_core.simulators.srunner.scenarios.opposite_direction import OppositeDirection
from haco.DIDrive_core.simulators.srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from haco.DIDrive_core.simulators.srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from haco.DIDrive_core.simulators.srunner.scenarios.signalized_junction_straight import SignalizedJunctionStraight
from haco.DIDrive_core.simulators.srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from haco.DIDrive_core.simulators.srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from haco.DIDrive_core.simulators.srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from haco.DIDrive_core.simulators.srunner.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, \
                                                                        NoSignalJunctionCrossingRoute
from haco.DIDrive_core.simulators.srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection

from haco.DIDrive_core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import \
    (CollisionTest,
     InRouteTest,
     RouteCompletionTest,
     OutsideRouteLanesTest,
     RunningRedLightTest,
     ActorSpeedAboveThresholdTest)

SECONDS_GIVEN_PER_METERS = 0.5

SCENARIO_CLASS_DICT = {
    "ControlLossNew": ControlLossNew,
    "ControlLoss": ControlLoss,
    "FollowLeadingVehicleNew": FollowLeadingVehicleNew,
    "FollowLeadingVehicle": FollowLeadingVehicle,
    "ChangeLane": ChangeLane,
    "CutIn": CutIn,
    "OppositeDirection": OppositeDirection,
    "ManeuverOppositeDirection": ManeuverOppositeDirection,
    "SignalizedJunctionLeftTurn": SignalizedJunctionLeftTurn,
    "SignalizedJunctionRightTurn": SignalizedJunctionRightTurn,
    "SignalizedJunctionStraight": SignalizedJunctionStraight,
    "DynamicObjectCrossing": DynamicObjectCrossing,
    'VehicleTurningRoute': VehicleTurningRoute,
    'NoSignalJunctionCrossingRoute': NoSignalJunctionCrossingRoute,
    'OtherLeadingVehicle': OtherLeadingVehicle
}

NUMBER_CLASS_DICT = {
    "Scenario1": 'ControlLoss',
    "Scenario2": 'FollowLeadingVehicle',
    "Scenario3": 'DynamicObjectCrossing',
    "Scenario4": 'VehicleTurningRoute',
    "Scenario5": 'OtherLeadingVehicle',
    "Scenario6": 'ManeuverOppositeDirection',
    "Scenario7": 'SignalizedJunctionStraight',
    "Scenario8": 'SignalizedJunctionLeftTurn',
    "Scenario9": 'SignalizedJunctionRightTurn',
    "Scenario10": 'NoSignalJunctionCrossingRoute'
}


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(
        location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']), z=float(actor_dict['z'])),
        rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw']))
    )


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """

    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, debug_mode=False, criteria_enable=True, resolution=1.0):
        """
        Setup all relevant parameters and create scenarios along route
        """

        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None
        self._resolution = resolution

        self._update_route(world, config, debug_mode)

        ego_vehicle = self._update_ego_vehicle()

        self.list_scenarios = self._build_scenario_instances(
            world,
            ego_vehicle,
            self.sampled_scenarios_definitions,
            scenarios_per_tick=5,
            timeout=self.timeout,
            debug_mode=debug_mode
        )

        super(RouteScenario, self).__init__(
            name=config.name,
            ego_vehicles=[ego_vehicle],
            config=config,
            world=world,
            debug_mode=False,
            terminate_on_failure=False,
            criteria_enable=criteria_enable
        )

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """

        # Transform the scenario file into a dictionary
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

        # prepare route's trajectory (interpolate and add the GPS route)
        gps_route, route = interpolate_trajectory(world, config.trajectory, hop_resolution=self._resolution)
        self.route = convert_waypoint_to_transform(route.copy())
        ds_ids = downsample_route(self.route, 1)
        global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]
        CarlaDataProvider.set_hero_vehicle_route(global_plan_world_coord)

        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(
            config.town, self.route, world_annotations
        )
        print('[SCENARIO] provided scenarios: ', len(world_annotations[config.town]))
        print('[SCENARIO] find scenarios in route: ', len(potential_scenarios_definitions))

        #CarlaDataProvider.set_hero_vehicle_route(convert_transform_to_location(self.route))

        #config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout()
        self.route_timeout = self.timeout

        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', elevate_transform, rolename='hero')

        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1].value == RoadOption.LEFT.value:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1].value == RoadOption.RIGHT.value:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1].value == RoadOption.CHANGELANELEFT.value:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1].value == RoadOption.CHANGELANERIGHT.value:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1].value == RoadOption.STRAIGHT.value:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(
            waypoints[0][0].location + carla.Location(z=vertical_shift),
            size=0.2,
            color=carla.Color(0, 0, 255),
            life_time=persistency
        )
        world.debug.draw_point(
            waypoints[-1][0].location + carla.Location(z=vertical_shift),
            size=0.2,
            color=carla.Color(255, 0, 0),
            life_time=persistency
        )

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rng = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rng.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _validate_type(self, definition):
        """
        Suit for scenario type from scenario runner
        """
        if 'Scenario' in definition['name']:
            definition['name'] = NUMBER_CLASS_DICT[definition['name']]

    def _build_scenario_instances(
        self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, debug_mode=False
    ):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []

        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(
                    scenario['trigger_position']['x'], scenario['trigger_position']['y'],
                    scenario['trigger_position']['z']
                ) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(
                    loc,
                    str(scenario['name']),
                    draw_shadow=False,
                    color=carla.Color(0, 0, 255),
                    life_time=100000,
                    persistent_lines=True
                )

        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number

            self._validate_type(definition)

            scenario_class = SCENARIO_CLASS_DICT[definition['name']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position

            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [
                ActorConfigurationData('vehicle.lincoln.mkz2017', ego_vehicle.get_transform(), 'hero')
            ]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name

            try:
                scenario_instance = scenario_class(
                    world, [ego_vehicle], scenario_configuration, criteria_enable=False, timeout=timeout
                )
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                scenario_number += 1
            except Exception as e:  # pylint: disable=broad-except
                if debug_mode:
                    traceback.print_exc()
                print("[WARNING] Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            for actor in get_actors_from_list(list_of_antagonist_actors['front']):
                actor.direction = 'front'
                list_of_actors.append(actor)

        if 'left' in list_of_antagonist_actors:
            for actor in get_actors_from_list(list_of_antagonist_actors['left']):
                actor.direction = 'left'
                list_of_actors.append(actor)

        if 'right' in list_of_antagonist_actors:
            for actor in get_actors_from_list(list_of_antagonist_actors['right']):
                actor.direction = 'right'
                list_of_actors.append(actor)

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """

        amount = config.n_vehicles
        new_actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*',
            amount,
            carla.Transform(),
            autopilot=True,
            random_location=True,
            rolename='background',
            disable_two_wheels=config.disable_two_wheels
        )

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """

        scenario_trigger_distance = 10  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        subbehavior = py_trees.composites.Parallel(
            name="Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        scenario_behaviors = []
        blackboard_list = []

        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name
                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name, scenario.config.trigger_points[0].location])
                else:
                    name = "{} - {}".format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(name, behaviour=scenario.scenario.behavior, name=name)
                    scenario_behaviors.append(oneshot_idiom)

        # Add behavior that manages the scenarios trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance, repeat_scenarios=False
        )

        subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)

        return behavior

    def _create_test_criteria(self):
        """
        """

        criteria = []

        route = convert_transform_to_location(self.route)

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)

        route_criterion = InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30, terminate_on_failure=False)

        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)

        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)

        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])

        # stop_criterion = RunningStopTest(self.ego_vehicles[0])

        blocked_criterion = ActorSpeedAboveThresholdTest(
            self.ego_vehicles[0], speed_threshold=0.1, below_threshold_max_time=90.0, terminate_on_failure=True
        )

        criteria.append(completion_criterion)
        criteria.append(collision_criterion)
        criteria.append(route_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(red_light_criterion)
        # criteria.append(stop_criterion)
        criteria.append(blocked_criterion)

        for scenario in self.list_scenarios:
            if scenario.scenario.test_criteria is not None:
                criteria += scenario.scenario.test_criteria

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
