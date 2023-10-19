from itertools import product
from copy import deepcopy


def get_all_states(map_size: tuple, attacker_num: int, defender_num: int):
    all_pos = list()
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            all_pos.append((i, j))

    agent_pos = list()
    for i in range(attacker_num):
        agent_pos.append(all_pos)
    for i in range(defender_num):
        agent_pos.append(all_pos)

    states = list(product(*agent_pos))
    return states


def get_capture_states(states: list, attacker_num: int, defender_num: int):
    policy = dict()
    for state in states:
        flag = False
        for i in range(attacker_num):
            for j in range(attacker_num, attacker_num + defender_num):
                if state[i] == state[j]:
                    flag = True
                    break
            if flag:
                break
        if flag:
            policy[state] = {"next": state, "cost": 0}
    return policy


def get_transitions(states: list, policy: dict, attacker_num: int,
                    defender_num: int):
    move = True
    while move:
        move = False
        for state in states:
            if state in policy:
                continue
            


if __name__ == "__main__":
    states = get_all_states((3, 3), 3, 1)
    policy = get_capture_states(states, 3, 1)
    print(policy)
