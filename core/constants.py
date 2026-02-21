"""Shared constants for the multi-agent RL system."""

# Direction head (5 outputs)
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_STAY = 0, 1, 2, 3, 4

# Action type head (5 outputs)
ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE = 0, 1, 2, 3, 4

# Direction deltas: UP, DOWN, LEFT, RIGHT, STAY (row, col)
DIR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
