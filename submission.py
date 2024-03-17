from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


def credit_from_package(package):
    return 2 * manhattan_distance(package.position, package.destination)


def moves_to_dropoff_held_package(robot):
    return manhattan_distance(robot.position, robot.package.destination) + 1


def moves_to_package_then_to_dropoff(robot, package):
    return manhattan_distance(robot.position, package.position) + 1 + manhattan_distance(package.position, package.destination)


def calculate_min_moves_to_pickup_package(robot, package):
    if robot.package:
        return manhattan_distance(robot.position, robot.package.destination) + 1 + \
            manhattan_distance(robot.package.destination, package.position)
    else:
        return manhattan_distance(robot.position, package.position) + 1


def calculate_min_moves_to_charging_station(env, robot):
    return min(manhattan_distance(robot.position, x.position) for x in env.charge_stations) + 1


def expected_credit_gain(env, robot_id):
    first_robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    if first_robot.package is not None:
        return credit_from_package(first_robot.package) / moves_to_dropoff_held_package(first_robot)
    else:
        potential_credits = [0]
        for package in env.packages:
            if not package.on_board:
                continue
            package_multiplier = 1 if moves_to_package_then_to_dropoff(first_robot, package) > moves_to_package_then_to_dropoff(other_robot, package) else 0.95
            potential_credits.append(package_multiplier * credit_from_package(package) / moves_to_package_then_to_dropoff(first_robot, package))
        return max(potential_credits)


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    first_robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    credit_diff = first_robot.credit - other_robot.credit
    credit_diff_factor = 100 / env.num_steps
    expected_credit_gain_factor = 5 / 20

    credit_heuristic = credit_diff * credit_diff_factor
    expected_credit_heuristic = expected_credit_gain(env, robot_id) * expected_credit_gain_factor

    with open('data.csv', 'a+') as f:
        f.write(f"{credit_heuristic}, {expected_credit_heuristic}\n")

    return credit_heuristic + expected_credit_heuristic


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
