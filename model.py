import argparse
import scipy as sc
from scipy.stats import rv_continuous
import scipy.stats as st
import math
import pandas as pd
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--time_step', '-t', help='Time step', default=1, type=int
	)
	parser.add_argument(
		'--field_size', '-s', help='Field size', default=1, type=int
	)
	parser.add_argument(
		'--part_num', '-n', help='Partition number', default=12, type=int
	)
	parser.add_argument(
		'--bottom_left', '-l', help='Field bottom left angle position', 
		default=(0,0), type=int
	)
	parser.add_argument(
		'-a', help='Additional term in potential formula', 
		default=0, type=float
	)
	parser.add_argument(
		'--mesure_num', '-m', help='Number of mesuarements', 
		default=10, type=int
	)
	parser.add_argument(
		'--filename', '-f', help='Filename of distribution density table',
		type=str
	)
	args = parser.parse_args()
	return vars(args)

class Particle:
	def __init__(self, field, x=0, y=0):
		self.x = x
		self.y = y
		self.field = field
		self.track = [(x, y)]

	def step(self):
		self.x, self.y = self.field.distr()
		self.track.append((self.x, self.y))

class System_dist(rv_continuous):

	def __init__(self, filename, **args):
		rv_continuous.__init__(self, **args)
		df = pd.pandas.read_csv(filename)
		self.table = df.values
		self.counter = 0
		self.gist = np.linspace(0, 1, self.table.shape[1] + 1)
		self.max_time = self.table.shape[0]

	def next(self):
		if (self.counter < self.max_time - 1):
			self.counter += 1
		else:
			raise IndexError('Generated sequence is over: {}'.format(self.max_time))

	def get_bin(self, x):
		for i in range(self.table.shape[1]):
			if (self.gist[i] <= x < self.gist[i + 1]):
				return i

	def _pdf(self, x):
		return self.table[self.counter][self.get_bin(x)]

	def _cdf(self, x):
		j = self.get_bin(x)
		n = self.table.shape[1]
		if (j == n - 1):
			return 1
		else:
			line = self.table[self.counter]
			return (n * x - j) * line[j + 1] + line[:j + 1].sum()

class Field:
	def __init__(self, field_size, bottom_left, part_num, a, filename):
		self.a = a
		self.time_stamp = 0
		self.size = field_size
		self.particles = [
			Particle(self, bottom_left[0], bottom_left[1]) 
			for _ in range(part_num)
		]
		self.distribution = System_dist(filename, a=0, b=1)
		self.x_min = bottom_left[0]
		self.x_max = bottom_left[0] + self.size
		self.y_min = bottom_left[1]
		self.y_max = bottom_left[1] + self.size

	def distr(self):
		coord = self.distribution.rvs(size=2)
		assert self.x_min <= coord[0] <= self.x_max
		assert self.y_min <= coord[1] <= self.y_max
		return coord[0], coord[1]

	def step(self, n):
		for _ in range(n):
			self.distribution.next()
			self.time_stamp += 1
			for particle in self.particles:
				particle.step()

	def potential(self, p1, p2):
		return 1/((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + self.a ** 2)

	def sir(self):
		potentials = [
			self.potential(self.particles[0], self.particles[i]) 
			for i in range(1, len(self.particles))
		]
		return max(potentials) / (sum(potentials) - max(potentials))

	def time(self):
		return self.time_stamp


def main(time_step, mesure_num, **args):
	field = Field(**args)
	print("TIME\tSIR")
	for i in range(mesure_num):
		field.step(time_step)
		print("{}\t{}".format(field.time(), field.sir()))

if __name__ == '__main__':
	args = parse_args()
	main(**args)
