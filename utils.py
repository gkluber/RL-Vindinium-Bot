import multiprocessing, threading
import numpy as np
import tensorflow as tf
import re #regular expressions
import math
import sets as s

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
	def mine(id: str) -> Tile:
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

	def player(id: str) -> Tile:
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
	
	def __init__(self, x: int, y: int):
		self.x = x
		self.y = y
	
	def north(self) -> Pos:
		return clone(self).add
	
	def clone(self) -> Pos:
		return Pos(self.x, self.y)
	
	def add(self, pos: Pos) -> Pos:
		self.x += pos.x
		self.y += pos.y
		return self
	
	def subtract(self, pos: Pos) -> Pos:
		self.x -= pos.x
		self.y -= pos.y
		return self

	def multiply(self, scalar: int) -> Pos:
		self.x *= scalar
		self.y *= scalar
		return self
	
	def distance(self, pos: Pos) -> int:
		return math.sqrt( (self.x - pos.x)^2 + (self.y - pos.y)^2 )
		
	def neighbors(self) -> s.Set:
		north = Pos(0,1)
		east = Pos(1,0)
		south = Pos(0,-1)
		west = Pos(-1,0)
		dirs = [north, east, south, west]
		return map(lambda x: self.clone().add(x), dirs)
	
	def vectorize(self, size: int) -> int:
		return self.x*size + self.y
	
	def mirrorX(self, size: int):
		self.x = size - 1 - pos.x
		return self
	
	def mirrorY(self, size: int):
		self.y = size - 1 - pos.y
		return self

#constructs a rank 3 tensor out of the state dim = 28 * 28 * 16
def state_json_to_tensor(state: dict, spawns: dict) -> np.ndarray:
	


	
def parseTile(str: str) -> Tile:
	if str == "##":
		return Tile.WALL
	str == "  ":
		return Tile.AIR
	str=="[]":
		return Tile.TAVERN
	match = re.match('\$([-0-9])', str)
	if match:
		return Tile.mine(match.group(1))
	match = re.match('\@([-0-9])', str)
	if match:
		return Tile.player(match.group(1))

#returns a 3 rank tensor of dim = size * size * 16 consisting of binary elements
def parseBoard(board: str, size: int) -> np.ndarray:
	board = np.zeros((size, size, 16), dtype = np.uint8)
	