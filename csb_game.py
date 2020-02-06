import math
import random
import sys
import numpy as np
import base64

LEFT_ACC = 0
LEFT_MOVE = 1
STRAIGHT_ACC = 2
STRAIGHT_MOVE = 3
RIGHT_ACC = 4
RIGHT_MOVE = 5

MAX_SPEED = 650
MAX_DISTANCE = math.sqrt(16000 * 16000 + 9000 * 9000)

maps = [ [ (12460,1350),(10540,5980),(3580,5180),(13580,7600)  ],[(3600,5280),(13840,5080),(10680,2280),(8700,7460),(7200,2160)],[(4560,2180),(7350,4940),(3320,7230),(14580,7700),(10560,5060),(13100,2320)],[(5010,5260),(11480,6080),(9100,1840)],[(14660,1410),(3450,7220),(9420,7240),(5970,4240)],[(3640,4420),(8000,7900),(13300,5540),(9560,1400)],[(4100,7420),(13500,2340),(12940,7220),(5640,2580)],[(14520,7780),(6320,4290),(7800,860),(7660,5970),(3140,7540),(9520,4380)],[(10040,5970),(13920,1940),(8020,3260),(2670,7020)],[(7500,6940),(6000,5360),(11300,2820)],[(4060,4660),(13040,1900),(6560,7840),(7480,1360),(12700,7100)],[(3020,5190),(6280,7760),(14100,7760),(13880,1220),(10240,4920),(6100,2200)],[(10323,3366),(11203,5425),(7259,6656),(5425,2838)] ];

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def Rotate(a, angle): # Have to add self since this will become a method
    r = math.radians(angle)
    cs = math.cos(r)
    sn = math.sin(r)

    px = a[0] * cs - a[1] * sn
    a[1] = a[0] * sn + a[1] * cs
    a[0] = px

def D2(a, b):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])

def collision2(p1, p2, v1, v2, isCp):
    if v1[0] == v2[0] and v1[1] == v2[1]:
        return 1.0;
    if isCp:
        sr2 = 357604
    else:
        sr2 = 640000
    dp = p1 - p2
    dv = v1 - v2
    a = dv[0]**2 + dv[1]**2;
    if a < 0.00001:
        return 1.0;
    b = -2.0 * (dp[0] * dv[0] + dp[1] * dv[1]);
    delta = b * b - 4.0 * a * (dp[0] * dp[0] + dp[1] * dp[1] - sr2);
    if (delta < 0.0):
        return 1.0;
    t = (b - math.sqrt(delta)) * (1.0 / (2.0 * a));
    if (t <= 0.0 or t > 1.0):
        return 1.0;
    return t;

class Collision:
    def __init__(self, pod1, pod2 = None ):
        self.pod1 = pod1
        self.pod2 = pod2
        if self.pod2 == None:
            self.t = collision2(pod1.pos, map[pod1.cp % len(map)], pod1.v, [0, 0], True)
        else:
            self.t = collision2(pod1.pos, pod2.pos, pod1.v, pod2.v, False)

    def Bounce(self):
        m1 = 1
        m2 = 1
        mcoeff = (m1 + m2) / (m1 * m2);
        n = self.pod1.pos - self.pod2.pos
        dst2 = n[0]**2 + n[1]**2
        dv = self.pod1.v - self.pod2.v
        prod = (n[0]*dv[0] + n[1]*dv[1]) / (dst2 * mcoeff);
        f = n * prod
        m1_inv = 1.0 / m1;
        m2_inv = 1.0 / m2;
        self.pod1.v -= f * m1_inv
        self.pod2.v += f * m2_inv
        impulse = math.sqrt(f[0]**2 + f[1]**2);
        if impulse < 120.:
            df = 120.0 / impulse;
            f *= df;
        self.pod1.v -= f * m1_inv
        self.pod2.v += f * m2_inv

class Pod:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([random.randint(0, 16000), random.randint(0, 9000)], dtype=np.float)
        self.angle = random.randint(0, 359)
        self.v = np.array([random.randint(0, MAX_SPEED), 0], dtype=np.float)
        Rotate(self.v, random.randint(0, 359))
        self.v[0] = int(self.v[0])
        self.v[1] = int(self.v[1])
        self.time = 100
        self.cp = 0

    def setInput(self, x, y, vx, vy, angle, next_check_point_id):
        self.pos = np.array([x, y], dtype=np.float)
        self.v = np.array([vx, vy], dtype=np.float)
        self.angle = angle
        self.cp = next_check_point_id

    def apply(self, a):
        if a == LEFT_ACC or a == LEFT_MOVE:
            self.angle = (self.angle - 18 + 360) % 360
        if a == RIGHT_ACC or a == RIGHT_MOVE:
            self.angle = (self.angle + 18 + 360) % 360
        if a % 2 == 0:
            ra = math.radians(self.angle)
            thrust = 100.0
            self.v[0] = self.v[0] + math.cos(ra) * thrust
            self.v[1] = self.v[1] + math.sin(ra) * thrust

    def end(self):
        self.v[0] = math.trunc(self.v[0] * 0.85)
        self.v[1] = math.trunc(self.v[1] * 0.85)
        self.pos[0] = round(self.pos[0])
        self.pos[1] = round(self.pos[1])
        self.time-=1

    def move(self, t):
        self.pos[0] = round(self.pos[0] + self.v[0] * t)
        self.pos[1] = round(self.pos[1] + self.v[1] * t)

    def encode(self):
        nextcp = self.pos - map[self.cp % len(map) ]
        nextnextcp = self.pos - map[(self.cp + 1) % len(map)]
        if self.cp == len(map) * 3 - 1:
            nextnextcp = nextcp
        inputs = []
        vx = np.array([1.0, 0.0])
        vy = np.array([0.0, 1.0])
        Rotate(vx, self.angle)
        Rotate(vy, self.angle)
        inputs.append(np.dot(vx, self.v) / MAX_SPEED)
        inputs.append(np.dot(vy, self.v) / MAX_SPEED)
        inputs.append(np.dot(vx, nextcp) / MAX_DISTANCE)
        inputs.append(np.dot(vy, nextcp) / MAX_DISTANCE)
        inputs.append(np.dot(vx, nextnextcp) / MAX_DISTANCE)
        inputs.append(np.dot(vy, nextnextcp) / MAX_DISTANCE)
        return inputs


    def encodeBlocker(self, runner):
        inputs = []

        vx = np.array([1.0, 0.0])
        vy = np.array([0.0, 1.0])
        Rotate(vx, self.angle)
        Rotate(vy, self.angle)

        runnerDir = np.array([1.0, 0.0])
        Rotate(runnerDir, runner.angle)

        runnerP = (self.pos - runner.pos) / 6

        inputs.append(np.dot(vx, self.v) / MAX_SPEED)
        inputs.append(np.dot(vy, self.v) / MAX_SPEED)

        inputs.append(np.dot(vx, runnerP) / MAX_DISTANCE)
        inputs.append(np.dot(vy, runnerP) / MAX_DISTANCE)

        inputs.append(np.dot(vx, runnerDir))
        inputs.append(np.dot(vy, runnerDir))

        return inputs



class CSB_Game:
    ACTION_SPACE = 36
    STATE_SIZE = 12

    def __init__(self):
        self.pods = [Pod(), Pod()]
        self.init_board()
        self.players = [0, 1]  # player1 and player2


    def init_board(self):
        global map
        self.mapid = random.randint(0, len(maps)-1)
        map = np.array(maps[self.mapid], dtype= np.float)
        self.map = map
        for pod in self.pods:
            pod.reset()
        self.current_player = 0

    def current_state(self):
        if self.current_player == 0:
            ec = self.pods[0].encode() + self.pods[1].encodeBlocker( self.pods[0])
        else:
            ec = self.pods[1].encode() + self.pods[0].encodeBlocker( self.pods[1])


        return np.array(ec)


    def do_move(self, a):
        #print(a)

        self.pods[self.current_player].apply(a)
        if self.current_player == 1:
            self.play()

        self.current_player = 1 - self.current_player


    def get_current_player(self):
        return self.current_player

    def has_a_winner(self):
        if self.current_player == 1:
            return False, -1

        if self.pods[0].time < 0 and self.pods[1].time < 0:

            d1 = D2(self.pods[0].pos, map[self.pods[0].cp])
            d2 = D2(self.pods[1].pos, map[self.pods[1].cp])

            winner = 1
            if d1 < d2:
                winner = 0
            return True, winner

        for i in range(2):
            if self.pods[i].time < 0:
                return True, 1 - i

        if self.pods[0].cp == self.pods[1].cp:
            return False, -1

        for i in range(2):
            if self.pods[i].cp == len(map):
                return True, i

        return False, -1

    def game_end(self):
        #print(self.has_a_winner())
        return self.has_a_winner()

    def availables(self):
        return [0, 1, 2, 3, 4, 5]

    def play(self):
        t = 0.0

        while t < 1.0:

            col = Collision(self.pods[0], self.pods[1])

            cpPods = [Collision(self.pods[i]) for i in range(2)]


            for coll in cpPods:
                if coll.t < col.t:
                    col = coll
            if col.t < 1.0 - t:
                for pod in self.pods:
                    pod.move(col.t)
                if col.pod2 == None:
                    col.pod1.cp += 1
                    col.pod1.time = 100
                    print(".")
                else:
                    col.Bounce()
                t += col.t
            else:
               for pod in self.pods:
                    pod.move(col.t)
               break

        for pod in self.pods:
            pod.end()



class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        #self.Draw = Draw

    def graphic(self, board):
        pass


    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):

        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
