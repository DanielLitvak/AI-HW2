import time
import math

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
    credit_diff_factor = 100 / (env.num_steps+1)
    expected_credit_gain_factor = 5 / 20

    credit_heuristic = credit_diff * credit_diff_factor
    expected_credit_heuristic = expected_credit_gain(env, robot_id) * expected_credit_gain_factor

    return credit_heuristic + expected_credit_heuristic

def autistic_heuristic(env: WarehouseEnv, robot_id: int):
    first_robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    credit_diff = first_robot.credit - other_robot.credit
    credit_diff_factor = 100 / (min(env.num_steps, first_robot.battery + other_robot.battery) + 1)
    expected_credit_gain_factor = 5 / 20
    enemy = (robot_id+1)%2

    credit_heuristic = credit_diff * credit_diff_factor
    expected_credit_heuristic = expected_credit_gain(env, robot_id) * expected_credit_gain_factor
    enemy_expected_credit = expected_credit_gain(env, enemy) * expected_credit_gain_factor

    return credit_heuristic + expected_credit_heuristic - enemy_expected_credit


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):

    def rb_minimax(self, env: WarehouseEnv, agent_id, turn, d):
        other_agent = (agent_id+1)%2
        if env.done():
            return (env.get_robot(agent_id).credit - env.get_robot(other_agent).credit) * 2**32
        if d == 0:
            return smart_heuristic(env, agent_id)
        if turn == agent_id:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            cur_max = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                cur_max = max(cur_max, self.rb_minimax(child, agent_id, (turn+1)%2, d-1))
            return cur_max
        else:
            operators = env.get_legal_operators(other_agent)
            children = [env.clone() for _ in operators]
            cur_min = math.inf
            for child, op in zip(children, operators):
                child.apply_operator(other_agent, op)
                cur_min = min(cur_min, self.rb_minimax(child, agent_id, (turn+1)%2, d-1))
            return cur_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children_heuristics = []
        total_operators = 7
        depth_to_scan = 0
        remaining_time = time_limit
        while (time_limit - remaining_time) < (time_limit / total_operators):
            start = time.time()
            children_heuristics = []
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                children_heuristics += [self.rb_minimax(child, agent_id, ((agent_id+1)%2), depth_to_scan)]
            depth_to_scan += 1
            remaining_time = remaining_time - (time.time() - start)
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        #print(remaining_time)
        return operators[index_selected]

class AgentAlphaBeta(Agent):
    def rb_alphabeta(self, env: WarehouseEnv, agent_id, turn, d, alpha, beta):
        other_agent = (agent_id + 1) % 2
        if env.done():
            return (env.get_robot(agent_id).credit - env.get_robot(other_agent).credit) * 2 ** 32
        if d == 0:
            return smart_heuristic(env, agent_id)
        if turn == agent_id:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            cur_max = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                cur_max = max(cur_max, self.rb_alphabeta(child, agent_id, (turn + 1) % 2, d - 1, alpha, beta))
                alpha = max(alpha, cur_max)
                if cur_max >= beta:
                    return math.inf
            return cur_max
        else:
            operators = env.get_legal_operators(other_agent)
            children = [env.clone() for _ in operators]
            cur_min = math.inf
            for child, op in zip(children, operators):
                child.apply_operator(other_agent, op)
                cur_min = min(cur_min, self.rb_alphabeta(child, agent_id, (turn + 1) % 2, d - 1, alpha, beta))
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -math.inf
            return cur_min

    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children_heuristics = []
        total_operators = 7
        depth_to_scan = 0
        remaining_time = time_limit
        while (time_limit - remaining_time) < (time_limit / total_operators):
            start = time.time()
            children_heuristics = []
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                children_heuristics += [self.rb_alphabeta(child, agent_id, ((agent_id+1)%2), depth_to_scan, alpha=-math.inf, beta=math.inf)]
            depth_to_scan += 1
            remaining_time = remaining_time - (time.time() - start)
        print(depth_to_scan)
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        #print(remaining_time)
        return operators[index_selected]


class AgentExpectimax(Agent):

    STEPS_WITH_DOUBLE_PROBABILITY = ['move east', 'pick up']

    def rb_expectimax(self, env: WarehouseEnv, agent_id, turn, d):
        other_agent = (agent_id+1)%2
        if env.done():
            return (env.get_robot(agent_id).credit - env.get_robot(other_agent).credit) * 2**32
        if d == 0:
            return smart_heuristic(env, agent_id)

        if turn == agent_id:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            cur_max = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                cur_max = max(cur_max, self.rb_expectimax(child, agent_id, (turn+1)%2, d-1))
            return cur_max
        else:
            operators = env.get_legal_operators(other_agent)
            denominator = len(operators) + len([x for x in self.STEPS_WITH_DOUBLE_PROBABILITY if x in operators])
            operator_probabilities = {}
            for op in operators:
                operator_probabilities[op] = (1 + (1 if op in self.STEPS_WITH_DOUBLE_PROBABILITY else 0)) / denominator
            children = [env.clone() for _ in operators]
            res = 0
            for child, op in zip(children, operators):
                child.apply_operator(other_agent, op)
                res += (operator_probabilities[op] * self.rb_expectimax(child, agent_id, (turn+1)%2, d-1))
            return res

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children_heuristics = []
        total_operators = 7
        depth_to_scan = 0
        remaining_time = time_limit
        while (time_limit - remaining_time) < (time_limit / total_operators):
            start = time.time()
            children_heuristics = []
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                children_heuristics += [self.rb_expectimax(child, agent_id, ((agent_id+1)%2), depth_to_scan)]
            depth_to_scan += 1
            remaining_time = remaining_time - (time.time() - start)
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        #print(remaining_time)
        return operators[index_selected]


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
