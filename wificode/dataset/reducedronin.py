import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

class ReducedRonin(object):
	def __init__(self, output_path):
		self.output_path = output_path
		
	def load_original_traj(self, name):
		mag, theta = [], []
		start_xy = [0,0]
		with open(name, 'r') as f:
			for i, line in enumerate(f):
				if i==0:
					continue
				line = line.strip()
				tokens = [float(token) for token in line.split('\t') if token.strip() != '']
				if len(tokens)>2:
					mag.append(tokens[0])
					theta.append(tokens[1]+tokens[2])
				elif len(tokens)==2:
					start_xy = tokens[0:2]
				elif len(tokens)==1:
					self.align_error = tokens[0]
		mag = np.asarray(mag)
		theta = np.asarray(theta)
		self.start_xy = np.asarray(start_xy)
		return mag, theta


	def read_reduced_ronin(self, name):
		self.mag, self.theta = self.load_original_traj(name)
		self.name = name
		data = np.load(self.output_path + "rssi_reduced.npz")
		self.rssi = data["rssi"]
		self.rssi[np.isnan(self.rssi)] = -102

		self.rssi = self.rssi+102
		self.rssi = np.maximum(0, self.rssi)
		self.nrssi = normalize(self.rssi)

		self.haswifi = data["haswifi"]
		self.ronin_traj = self.update_traj(0.) # you need to update this

		# print(self.mag.shape, self.theta.shape)
		# print(self.rssi.shape, self.haswifi.shape)

	def update_traj(self, bias):
		updated_x = np.cos(self.theta+bias)*self.mag
		updated_y = np.sin(self.theta+bias)*self.mag
		tmp = np.stack([updated_x, updated_y],-1)
		tmp = np.cumsum(tmp, axis=0)
		upt_traj = np.zeros((tmp.shape[0]+1,2))
		upt_traj[1:,:]= tmp
		upt_traj = upt_traj+self.start_xy
		return upt_traj