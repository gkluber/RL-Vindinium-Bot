#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import requests
import re

class Client:
	
	def __init__(self, bot, timeout, server_url, key, mode, number_of_turns, depth):
		self.bot = bot
		self.server_url = server_url
		self.key = key 
		self.mode = mode
		self.timeout = timeout
		self.number_of_turns = number_of_turns
		self.session = None
		#self.open = False
		self.stateBuffer = []
		self.depth = 1
	
	def get_new_game_state(self, session):
		"""Get a JSON from the server containing the current state of the game"""

		if(self.mode=='training'):
			params = { 'key': self.key, 'turns': self.number_of_turns}
			api_endpoint = '/api/training'
		elif(self.mode=='arena'):
			params = { 'key': self.key}
			api_endpoint = '/api/arena'

		#Wait for 10 minutes
		r = session.post(self.server_url + api_endpoint, params, timeout=10*60)

		if(r.status_code == 200):
			return r.json()
		else:
			print("Error when creating the game")
			print(r.text)

	#returns the next state of the board
	def move(self, session, url, action):
		"""Send a move to the server
		
		Moves can be one of: 'Stay', 'North', 'South', 'East', 'West' 
		"""

		try:
			r = session.post(url, {'dir': action.name.title()}, timeout=TIMEOUT)

			if(r.status_code == 200):
				return r.json()
			else:
				print("Error HTTP %d\n%s\n" % (r.status_code, r.text))
				return {'game': {'finished': True}}
		except requests.exceptions.RequestException as e:
			print(e)
			return {'game': {'finished': True}}


	def is_finished(state):
		return state['game']['finished']
		
	def hasNext(self):
		if len(self.stateBuffer) >= self.depth:
			return True
		else:
			return False
	
	#wraps the next state in a Game object
	def next(self):
		if not hasNext():
			return None
		else:
			temp = self.stateBuffer[self.depth-1]
			del self.stateBuffer[self.depth-1]
			#will return the most recent state first in the array
			return [Game(x) for x in self.stateBuffer[:(depth-1)]].append(temp)
	
	def start(self):
		"""Starts a game with all the required parameters"""

		# Create a requests session that will be used throughout the game
		self.session = requests.session()

		if(mode=='arena'):
			print(u'Connected and waiting for other players to join…')
		# Get the initial state
		state = get_new_game_state(session) #in JSON format
		'''if state is not None:
			self.open = True
		else
			return False'''
		print("Playing at: " + state['viewUrl'])

		while not is_finished(state):
			# Some nice output ;)
			sys.stdout.write('.')
			sys.stdout.flush()

			# Choose a move
			bot.callback(state)
			direction = bot.move(state)

			# Send the move and receive the updated game state
			url = state['playUrl']
			state = move(session, url, direction)
		
		close()
		
		return True
	
		# Clean up the session
	def stop(self):
		if session is not None:
			self.session.close()
