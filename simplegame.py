import random
import cv2
import numpy as np

size = 10

class Player:
    def __init__(self, food):
        self.pos = [food[0], food[1]]
        while self.pos[0] == food[0] and self.pos[1] == food[1]:
            self.pos = [random.randint(0, size - 1), random.randint(0, size - 1)]

class SimpleGame:

    def __init__(self):
        self.init_board()

    def init_board(self):
        self.round = 0
        self.food = [random.randint(0, size - 1), random.randint(0, size - 1)]
        self.current_player = 0
        self.players = [Player(self.food), Player(self.food)]

    def current_state(self):
        ret = []
        for j in range(2):
            p = self.players[j - self.current_player].pos
            ret.append(p[0])
            ret.append(p[1])
        ret.append(self.food[0])
        ret.append(self.food[1])


        return np.array(ret) / size


    def do_move(self, action):
        self.round+=1
        player = self.players[self.current_player].pos
        if action == 0:
            player[0] += 1
        elif action == 1:
            player[0] -= 1
        elif action == 2:
            player[1] += 1
        elif action == 3:
            player[1] -= 1
        else:
            raise Exception()



    def has_a_winner(self):
        done = False
        reward = -1

        for i in range(2):
            player = self.players[i]

            if player.pos[0] == self.food[0] and player.pos[1] == self.food[1]:
                reward = i
                done = True


            for j in range(2):
                if player.pos[j] < 0  or player.pos[j] > size:
                    reward = 1 - i
                    done = True

        if self.round > 50:
            done = True

        self.current_player = 1 - self.current_player
        return done, reward

    def game_end(self):
        # print(self.has_a_winner())
        return self.has_a_winner()

    def get_current_player(self):
        return self.current_player


    def draw(self):
        im = np.zeros((self.size, self.size, 3))
        ySize = int(self.size / 2)

        if self.playerX > -1 and self.playerX < self.size and self.playerY > -1 and self.playerY < self.size:
            im[self.playerY, self.playerX] = [0,127,0]
        im[ySize, self.foodX] = [0,0,127]
        im = cv2.resize(im, (300, 300))
        return im






class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        #self.Draw = Draw()

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

        #if is_shown:

            #self.graphic(self.board, player1.player, player2.player)
        while True:

            print(",")
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
            #self.Draw.Draw(self.board)
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)

            #print(move, move_probs)
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