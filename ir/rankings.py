from __future__ import print_function
import json
import simcalc
import numpy as np
import pandas as pd


class Rankings(object):
    """Stores ranking lists for Cosine Similary and Euclidean Distance"""
    def __init__(self, num_to_rank):
        super(Rankings, self).__init__()
        self.cos_sim_ranking = []
        self.euclid_dist_ranking = []
        self.num_to_rank = num_to_rank

    def update_cos_sim_ranking(self, name, cos_sim_value):
        if len(self.cos_sim_ranking) < self.num_to_rank:
            self.cos_sim_ranking.append(simcalc.CelebClass(name, cos_sim_value))
            self.cos_sim_ranking.sort(key=lambda c: c.value, reverse=True)
        else:    
            if cos_sim_value > self.cos_sim_ranking[self.num_to_rank - 1].value:
                self.cos_sim_ranking.pop(self.num_to_rank - 1)
                self.cos_sim_ranking.append(simcalc.CelebClass(name, cos_sim_value))
                self.cos_sim_ranking.sort(key=lambda c: c.value, reverse=True)
            else:
                return

    def update_euclid_dist_ranking(self, name, euclid_dist_value):
        if len(self.euclid_dist_ranking) < self.num_to_rank:
            self.euclid_dist_ranking.append(simcalc.CelebClass(name, euclid_dist_value))
            self.euclid_dist_ranking.sort(key=lambda c: c.value)
        else:    
            if euclid_dist_value < self.euclid_dist_ranking[self.num_to_rank - 1].value:
                self.euclid_dist_ranking.pop(self.num_to_rank - 1)
                self.euclid_dist_ranking.append(simcalc.CelebClass(name, euclid_dist_value))
                self.euclid_dist_ranking.sort(key=lambda c: c.value)
            else:
                return


def main(query):
    """
    Calculates cosine similarity and euclidean distance.

    Args:
     query (1 dimension numpy array)
    Returns:
     Rankings object: results in lists
    """
    ranking = Rankings(3)

    data = pd.read_pickle('celeb_vectors.pk')

    for item in data.iteritems():
        simcalc.calculate_cosine_sim_euclid_dist(item[0], np.array(item[1]), query, ranking)

    return ranking


def json_to_pickle():
    with open('celeb_vectors.json') as data_file:
        data = json.load(data_file)
        data_file.close()

    df = pd.Series(data)

    pd.to_pickle(df, 'celeb_vectors.pk')

