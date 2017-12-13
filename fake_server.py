import random
import numpy as np
from sets import Set
from queue import Queue

#allows for fast self-play on randomly generated maps
class FakeServer:
	def __init__(self):
		pass
		
	def generateMap(self) -> Map:
		pass
	
	
	def register(self, bot: Callable[[ndarray],None]) -> Player:
		pass
	
	def move(self, p: Player) -> None:
		pass
		

class Config:
	def __init__(self, map: Map, turns = 1200: int):
		self.map_config = map
		self.turns = turns
	
	def 
	
class Generator:
	def __init__(self):
		pass
	
	def random_map(self):
		size = 10 + 2 * random.randrange(10)
		wall_percent = 10 + random.randrange(32)
		mine_percent = 3 + random.randrange(7)
	
	def attempt(self, size: int, wall_percent: int, mine_percent: int, attempts = 1: int):
		draft = generate_board(
	
	#outputs unrolled map
	def generate_board(config: Config) -> np.ndarray:
		
	
	#returns 1/4 of the board
	def sector(config: Config) -> np.ndarray:
		length = config.size ^ 2
		sector = np.zeros(length)
		for x in range(length):
			rand = random.randrange(100)
			sector[x] = Tile.MINE_N if rand < config.mine_percent else (Tile.WALL if rand < (config.mine_percent + config.wall_percent) else Tile.AIR)
		
		return sector
	
	def combine_sectors(sector: np.ndarray) -> np.ndarray:
		mirror = sector[::-1]
		first = np.append(sector,mirror)
		second = first[::-1]
		return np.append(first,second)
		
	def generate_spawn_pos(config: Config, board: np.ndarray, attempts = 1: int):
		trial = Pos(random.randrange(config.size / 2 - 2), random.randrange(config.size / 2 - 2))
		if validate_spawn_pos(
	
	def validate_spawn_pos(board: np.ndarray, pos: Pos) -> bool:
		trav = Traverse(board, pos).traverse()
		return pos.clone().mirrorX().vectorize() in trav and pos.clone().mirrorY().vectorize() in trav

class Config:
	def __init__(self, size: int, wall_percent: int, mine_percent: int):
		self.size = size
		self.wall_percent = wall_percent
		self.mine_percent = mine_percent
	
	@staticmethod
	def rand_config() -> Config:
		size = 10 + 2 * random.randrange(10)
		wall_percent = 10 + random.randrange(32)
		mine_percent = 3 + random.randrange(7)
		return Config(size, wall_percent, mine_percent)

class Player(Enum):
	P1 = object()
	P2 = object()
	P3 = object()
	P4 = object()
	
	def __repr__(self):
		return "<{}.{}>".format(self.__class__.__name__, self.name)

class Traverse: 
	def __init__(self, board: np.ndarray, pos: Pos):
		self.board = board
		self.pos = pos
		
	def traverse(self):
		return self.__traverse([self.pos], Set(), [])
	
	#Breadth first search
	def __traverse(self, toVisit: list, visited: Set, accumulator: list) -> list:
		if len(toVisit)==0:
			return accumulator
		else:
			next = toVisit[0]
			succ = walkable_from(next) - visited - set(toVisit) #set difference
			__traverse(toVisit[-1] + succ, visited.add(next), accumulator + next)
	
	def walkable_from(next: Pos):
		result = set()
		for x in next.neighbors:
			vec = x.vectorize()
			if board[vec] == Tile.AIR:
				result.add(vec)
		return result
		