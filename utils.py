import multiprocessing, threading
import numpy as np
import tensorflow as tf
import re #regular expressions
import math
from enum import Enum

#uses object as the value because it uses less memory than auto()
class Tile(Enum):
	AIR = object()
	SPAWN_P1 = object()
	SPAWN_P2 = object()
	SPAWN_P3 = object()
	SPAWN_P4 = object()
	MINE_P1 = object()
	MINE_P2 = object()
	MINE_P3 = object()
	MINE_P4 = object()
	MINE_N = object() 
	P1 = object()
	P2 = object()
	P3 = object()
	P4 = object()
	TAVERN = object()
	WALL = object()
	
	#hides the value because the only important component is the name
	def __repr__(self):
		return "<{}.{}>".format(self.__class__.__name__, self.name)
	
	@staticmethod
	def mine(id: str):
		if id=="1":
			return Tile.MINE_P1
		if id=="2":
			return Tile.MINE_P2
		if id=="3":
			return Tile.MINE_P3
		if id=="4":
			return Tile.MINE_P4
		if id=="-":
			return Tile.MINE_N
		raise ValueError("Invalid mine identifier")

	def player(id: str):
		if id=="1":
			return Tile.P1
		if id=="2":
			return Tile.P2
		if id=="3":
			return Tile.P3
		if id=="4":
			return Tile.P4
		raise ValueError("Invalid player identifier")

class Pos:
	#mutable positions
	def __init__(self, x: int, y: int):
		self.x = x
		self.y = y
	
	def north(self):
		return clone(self).add
	
	def clone(self):
		return Pos(self.x, self.y)
	
	def add(self, pos):
		self.x += pos.x
		self.y += pos.y
		return self
	
	def subtract(self, pos):
		self.x -= pos.x
		self.y -= pos.y
		return self

	def multiply(self, scalar: int):
		self.x *= scalar
		self.y *= scalar
		return self
	
	def distance(self, pos) -> int:
		return math.sqrt( (self.x - pos.x)^2 + (self.y - pos.y)^2 )
		
	def neighbors(self) -> list:
		return map(lambda x: self.clone().add(x), [Pos.get_pos_vector(x) for x in range(4)])
	
	@staticmethod
	def get_pos_vector(direction: int):
		if direction == 0: #Action.NORTH.value
			return Pos(0,1)
		elif direction == 1: #Action.EAST.value
			return Pos(1,0)
		elif direction == 2: #Action.SOUTH.value
			return Pos(0,-1)
		elif direction == 3: #Action.WEST.value
			return Pos(-1,0)
	
	def vectorize(self, size: int) -> int:
		return self.x*size + self.y
	
	def mirrorX(self, size: int):
		self.x = size - 1 - pos.x
		return self
	
	def mirrorY(self, size: int):
		self.y = size - 1 - pos.y
		return self

#use Action.value to get num
class Action(Enum):
	NORTH = 0
	EAST = 1
	SOUTH = 2
	WEST = 3

class GameState:
	
	def __init__(self, state: dict):
		self.state_dict = state
		self.size = state['board']['size'] #int
		self.tile_string = state['board']['tiles']
		self.finished = state['board']['finished'] #special treatment
		self.player_id = state['hero']['id']
		self.players = [Player(x['id'], Pos(x['pos']['x'], x['pos']['y']), x['life'], x['gold'], 
								Pos(x['spawnPos']['x'], x['spawnPos']['y']), x['crashed']) for x
								in state['heroes']] #note that state['heroes'] is already ordered w.r.t the player ids
		self.spawn_positions = [x.spawn_pos for x in self.players]
		self.turn = state['turn']
		self.max_turn = state['maxTurns']
		self.__read_board()
	
	def get_player(self, id: int):
		return self.players[id]
	
	def get_player_id(self) -> int:
		return self.player_id
	
	#reads the board in a way different from parse_board by 
	def __read_board(self):
		self.walls = np.zeros((self.size,self.size), dtype = np.uint8)
		self.taverns = []
		self.neutral_mines = []
		self.p1_mines = []
		self.p2_mines = []
		self.p3_mines = []
		self.p4_mines = []
		for x in range(self.size):
			for y in range(self.size):
				chunk = chunk(self.tile_string, x, y)
				tile = parse_tile(chunk)
				if tile == Tile.WALL:
					self.walls[x][y] = 1
				elif tile == Tile.TAVERN:
					self.taverns.append(Pos(x,y))
				elif tile == Tile.MINE_N:
					self.neutral_mines.append(Pos(x,y))
				elif tile == Tile.MINE_P1:
					self.p1_mines.append(Pos(x,y))
				elif tile == Tile.MINE_P2:
					self.p2_mines.append(Pos(x,y))
				elif tile == Tile.MINE_P3:
					self.p3_mines.append(Pos(x,y))
				elif tile == Tile.MINE_P4:
					self.p4_mines.append(Pos(x,y))
		self.read = True
	
	def get_mines(id: int) -> list:
		if id==1:
			return self.p1_mines
		elif id==2:
			return self.p2_mines
		elif id==3:
			return self.p3_mines
		elif id==4:
			return self.p4_mines
		return []
	
	# returns a rank 3 tensor of dim = input_size * input_size * 23
	# layer 1: corresponds to obstacle map -- binary
	# layer 2: corresponds to tavern positions -- binary
	# layer 3-7: corresponds to the mine map -- binary
	# layer 8-11: corresponds to player positions -- binary
	# layer 12-15: corresponds to player spawn positions -- binary
	# layer 16-19: corresponds to player HP (uniform) -- discrete
	# layer 20-23: corresponds to player gold (uniform)-- discrete
	# NOTE: layers 3, 8, 12, 16, and 20 are modified such that they correspond to the current player
	def tensorize(self, input_size: int) -> np.ndarray:
		tensor = np.zeros((input_size, input_size, 23), dtype = np.uint16)
		tensor[:,:,0] = self.walls
		for tav in self.taverns:
			tensor[tav.x,tav.y,1] = 1
		for m in neutral_mines:
			tensor[m.x, m.y, 2] = 1
		
		cur_player = self.get_player_id - 1
		for mine_index in range(4):
			mines = get_mines(cur_player + 1)
			for mine in mines:
				tensor[mine.x,mine.y,mine_index + 2] = 1
			cur_player = (cur_player + 1)%4
		
		for player_index in range(4):
			player = get_player(cur_player) 
			pos = player.pos
			spawn_pos = player.spawn_pos
			hp = player.life
			gold = player.gold
			
			tensor[pos.x,pos.y,player_index + 7] = 1 if not player.crashed else 0
			tensor[spawn_pos.x,spawn_pos.y,player_index + 11] = 1 if not player.crashed else 0
			tensor[:,:,player_index + 15] = player.hp_mat(input_size)
			tensor[:,:,player_index + 19] = player.gold_mat(input_size)
			
			cur_player = (cur_player + 1)%4
		
		return tensor
	
	
class Player:
	def __init__(self, id: int, pos, life: int, gold: int, spawn_pos, crashed: bool):
		self.player_id = id
		self.pos = pos
		self.life = life
		self.gold = gold
		self.spawn_pos = spawn_pos
		self.crashed = crashed #special treatment
	
	#returns an uniform matrix representing the hp of the player
	def hp_mat(size: int) -> np.ndarray:
		return np.full((size,size), self.life if not self.crashed else 0, dtype=np.uint16)
	
	#NOTE SIZE HYPERPARAM CORRESPONDS TO THE SIZE OF THE N X N PLACEHOLDER
	def gold_mat(size: int) -> np.ndarray:
		return np.full((size,size), self.gold if not self.crashed else 0, dtype=np.uint16)
	
	def spawn_pos_mat(size: int) -> np.ndarray:
		result = np.zeros((size,size), dtype=np.uint16)
		if self.crashed:
			return result
		result[self.spawn_pos.x][self.spawn_pos.y] = 1
		return result
	
	def pos_mat(size: int) -> np.ndarray:
		result = np.zeros((size,size), dtype=np.uint16)
		if self.crashed:
			return result
		result[self.pos.x][self.pos.y] = 1
		return result

def parse_tile(str: str):
	if str == "##":
		return Tile.WALL
	if str == "  ":
		return Tile.AIR
	if str=="[]":
		return Tile.TAVERN
	match = re.match('\$([-0-9])', str)
	if match:
		return Tile.mine(match.group(1)) #gets the number associated with the $x token
	match = re.match('\@([-0-9])', str)
	if match:
		return Tile.player(match.group(1)) #gets the number associated with the @x token

# returns a rank 3 tensor of dim = size * size * 11 consisting of binary elements
# layer 1: corresponds to obstacle map
# layer 2: corresponds to tavern positions
# layer 3-7: corresponds to the mine map
# layer 8-11: corresponds to player positions
def parse_board(board: str, size: int) -> np.ndarray:
	tensor = np.zeros((size, size, 11), dtype = np.uint8)
	if len(board) is not size^2:
		raise ValueError("Input board size incorrect")
	
	for x in range(size):
		for y in range(size):
			chunk = chunk(board, x, y)
			tile = parse_tile(chunk)
			if tile == Tile.WALL:
				tensor[x][y][0] = 1
			elif tile == Tile.TAVERN:
				tensor[x][y][1] = 1
			elif tile == Tile.MINE_N:
				tensor[x][y][2] = 1
			elif tile == Tile.MINE_P1:
				tensor[x][y][3] = 1
			elif tile == Tile.MINE_P2:
				tensor[x][y][4] = 1
			elif tile == Tile.MINE_P3:
				tensor[x][y][5] = 1
			elif tile == Tile.MINE_P4:
				tensor[x][y][6] = 1
			elif tile == Tile.P1:
				tensor[x][y][7] = 1
			elif tile == Tile.P2:
				tensor[x][y][8] = 1
			elif tile == Tile.P3:
				tensor[x][y][9] = 1
			elif tile == Tile.P4:
				tensor[x][y][10] = 1
	
	return tensor

def chunk(board: str, x: int, y: int) -> str:
	return board[(size*x + 2*y):(size*x + 2*y + 2)]