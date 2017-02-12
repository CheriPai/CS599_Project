from __future__ import print_function
from scipy import spatial
import numpy as np


class CelebClass(object):
    """Stores celeb index/name and similarity or difference result"""
    def __init__(self, name, value):
        super(CelebClass, self).__init__()
        self.name = name
        self.value = value
        

def calculate_cosine_sim(name, array_of_vectors, query, rankings):
    cos_sim_results = []
    for row in range(array_of_vectors.shape[0]):
        cos_sim_results.append(1 - spatial.distance.cosine(array_of_vectors[[row], :], query))
    rankings.update_cos_sim_ranking(name, max(cos_sim_results))
