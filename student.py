"""
Bot for PAC-MAN IIA 2018
"""

__author__ = "Catarina Silva, Daniel Carvalho"
__email__ = "c.alexandracorreia@ua.pt, danielmatoscarvalho@ua.pt"
__version__ = "3.0"

import random
import sys
import json
import asyncio
import websockets
import os
import logging
import math
from mapa import Map
from tree_search import *

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger('Client')
logger.setLevel(logging.INFO)


# Converts any array of scores/distances into the range of [a,b]
# By default it scales into the range [0,1]
def scaling(scores, a=0.0, b=1.0):
    minimum = min(scores)
    maximum = max(scores)
    d = (maximum - minimum)
    if d > 0:
        return [(b - a) * ((s - minimum) / d) + a for s in scores]
    else:
        return [(b - a) * (s - minimum)  + a for s in scores]


# Converts and array of distances into an array of scores.
# The difference id that the lower the distance the better.
# On the other hand, the higher the score the better.
def distance_2_score(distances):
    distances = scaling(distances)
    return [1.0 - i for i in distances]


# Combines a variable list of scores
# It is used to simulate a voting system
def combine_scores(l, *args):
    scores = []
    for i in range(0, l):
        score = 0.0
        for a in args:
            score += a[i]
        scores.append(score/len(a))
    return scores


# Computes the Minkowski distance.
# Its a general form to compute distances:
# P = 1 gives the Manhattan distance
# P = 2 gives the Euclidean distance
def distance(a, b, p=1.0):
    if p == 1:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    if p == 2:
        math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return (abs(a[0]-b[0])**p + abs(a[1]-b[1])**p)**(1/p)


# Return the closer element from the list l to the element e
def find_closer(e, l):
    ldistances = [distance(x, e) for x in l]
    return l[ldistances.index(min(ldistances))]


# PAC-MAN Search Domain
# Relies on the mapa.py code to generate the valid actions.
# The cost function avoids ghosts based on distance,
# the proximity of a ghost increases the cost of the final solution.
# There is a simple cache (dic) that stores the cost of each state,
# this is used to improve performance (dynamic programming).
class PacPath(SearchDomain):
    def __init__(self, Map, lghosts, lboosts, ghost_penalization=0, boost_penalization=0):
        self.Map = Map
        self.lghosts = lghosts
        self.lboosts = lboosts
        self.ghost_penalization = ghost_penalization
        self.boost_penalization = boost_penalization
        self.cache = {}

    def actions(self, state):
        rv = []
        for d in ['w', 's', 'a', 'd']:
            pos = self.Map.calc_pos(state[0], d)
            if pos != state[0] and pos not in self.lghosts:
                rv.append(d)
        return rv

    def result(self, state, action):
        return (self.Map.calc_pos(state[0], action[0]), action)

    def cost(self, state, action):
        if len(self.lghosts) == 0 and len(self.lboosts) == 0:
            return 1.0
        else:
            new_state = self.result(state, action)
            if new_state in self.cache:
                cost = self.cache[new_state]
            else:
                if len(self.lghosts) == 0:
                    boost_distances = [distance(b, new_state[0]) for b in self.lboosts]
                    cost = 1.0 + self.boost_penalization/(1.0 + min(boost_distances))
                elif len(self.lboosts) == 0:
                    ghost_distances = [distance(g, new_state[0]) for g in self.lghosts]
                    cost = 1.0 + self.ghost_penalization/(1.0 + min(ghost_distances))
                else:
                    ghost_distances = [distance(g, new_state[0]) for g in self.lghosts]
                    boost_distances = [distance(b, new_state[0]) for b in self.lboosts]
                    cost = 1.0 + self.ghost_penalization/(1.0 + min(ghost_distances)) + self.boost_penalization/(1.0 + min(boost_distances))
                self.cache[new_state] = cost
            return cost

    def heuristic(self, state, goal_state):
        return distance(state[0], goal_state[0])

    def satisfies(self, state, goal):
        if state[0] == goal[0]:
            return True
        return False

    def equivalent(self, state1, state2):
        return self.satisfies(state1, state2)


# Select a target
# Based on a simple heuristic
def select_target(ppos, lenergy, lghosts, lzombies, lboosts, ptarget, ghost_spawn, ghost_chase_distance, zombie_speed):
    # Return a zombie if it is close enough
    if len(lzombies) > 0:
        distance_zombies = [distance(ppos, x[0], p=1) for x in lzombies]
        valid_zombies = [x[0] for x in zip(lzombies, distance_zombies) if (x[0][1] > x[1] + x[0][1] * zombie_speed) and distance(x[0][0], ghost_spawn) > 4]
        if len(valid_zombies) > 0:
            logger.debug("Target = Zombie")
            return [x[0] for x in valid_zombies]

    # Return a boots if a ghost is tailing
    if len(lghosts) > 0 and len(lboosts) > 0:
        if distance(find_closer(ppos, lghosts), ppos) < ghost_chase_distance and len(lboosts) > 0:
            logger.debug("Target = Boost(Chased)")
            return lboosts

    # Improve performance on a clean map
    if len(lghosts) == 0 and len(lzombies) == 0:
        logger.debug("Target = Energy & Boost")
        targets = lenergy + lboosts
        distance_target = [distance(ppos, x) for x in targets]
        distance_ptarget = [distance(ptarget, x) for x in targets]
        scores = combine_scores(len(targets), distance_target, distance_ptarget)
        targets = [x for _,x in sorted(zip(scores, targets))]
        return targets

    # Only the energy need and heuristic to be targeted
    # The idea is to target the energy with the following conditions:
    # 1. The closer energy
    # 2. In direction of the pacman (smooth movement)
    # 3. Farther away from ghosts
    if len(lenergy) > 0:
        if len(lenergy) == 1:
            logger.debug("Target = Energy")
            return lenergy
        distance_energies = distance_2_score([distance(ppos, x) for x in lenergy])
        distance_ptarget = distance_2_score([distance(ptarget, x) for x in lenergy])
        if len(lghosts) > 0:
            score_ghosts = scaling([min([distance(g, t) for g in lghosts]) for t in lenergy], a=0, b=3)
            scores = combine_scores(len(lenergy), distance_energies, distance_ptarget, score_ghosts)
        else:
            scores = combine_scores(len(lenergy), distance_energies, distance_ptarget)
        targets = [x for _,x in sorted(zip(scores, lenergy), reverse=True)]
        logger.debug("Target = Energy")
        return targets

    # Eat the last boost
    if len(lboosts) > 0:
        logger.debug("Target = Boost")
        return lboosts
    return None

def reverse_directions(d):
    rd = {'w':'s', 'a':'d', 's': 'w', 'd':'a'}
    return rd[d]

async def agent_loop(server_address = "localhost:8000", agent_name="student"):
    async with websockets.connect("ws://{}/player".format(server_address)) as websocket:
        # Receive information about static game properties 
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        msg = await websocket.recv()
        game_properties = json.loads(msg) 

        _map = Map(game_properties['map'])

        area = _map.size[0] * _map.size[1]
        GHOST_PENALIZATION = area
        BOOST_PENALIZATION = max(_map.size)
        GHOST_CHASE_DISTANCE = 3
        N_GOALS = 4
        ZOMBIE_SPEED = .5

        #initialize agent properties 
        key = None
        cur_x, cur_y = None, None
        ptarget = None
        while True: 
            r = await websocket.recv()
            state = json.loads(r)
            ppos = tuple(state['pacman'])
            lenergy = [tuple(x) for x in state['energy']]
            lghosts = [tuple(x[0]) for x in state['ghosts'] if not x[1]]
            lzombies = [(tuple(x[0]), x[2]) for x in state['ghosts'] if x[1]]
            lboosts = [tuple(x) for x in state['boost']]
            if ptarget is None:
                ptarget = ppos

            if not state['lives'] or (len(lenergy) == 0 and len(lboosts) == 0):
                logger.info("GAME OVER")
                return

            logger.debug("Lives = %d", state['lives'])

            # Create new search problem
            targets = select_target(ppos, lenergy, lghosts, lzombies, lboosts, ptarget, tuple(_map.ghost_spawn), GHOST_CHASE_DISTANCE, ZOMBIE_SPEED)
            
            # Create multiple goals
            goals = []
            for i in range(min(len(targets), N_GOALS)):
                goals += [(targets[i], None)]

            # Search all goals at the same time
            if len(lghosts) == 0 and len(lzombies) == 0:
                st = SearchTree(SearchProblem(PacPath(_map, lghosts, lboosts), (ppos, None), goals))
            else:
                st = SearchTree(SearchProblem(PacPath(_map, lghosts, lboosts, GHOST_PENALIZATION, BOOST_PENALIZATION), (ppos, None), goals))
            path = st.search()

            # When trapped by ghosts take a valid step
            if path is None:
                logger.debug("No Path found...")
                if direction is None:
                    for d in ['w', 's', 'a', 'd']:
                        npos = _map.calc_pos(ppos, d)
                        if npos != ppos:
                            direction = d
                            target = npos
                            cost = 1
            else:
                target = path[1][-1][0]
                direction = path[1][1][1]
                cost = path[0]
                
            logger.debug("Target = %s Direction = %s (Cost = %f)", target, direction, cost)
            #send new key
            await websocket.send(json.dumps({"cmd": "key", "key": direction}))
            ptarget = target

loop = asyncio.get_event_loop()
SERVER = os.environ.get('SERVER', 'localhost')
PORT = os.environ.get('PORT', '8000')
NAME = os.environ.get('NAME', '76399_64003')
loop.run_until_complete(agent_loop("{}:{}".format(SERVER,PORT), NAME))
