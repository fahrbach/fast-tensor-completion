import numpy as np
import tensorly as tl


class TensorDataManager:
    def __init__(self):
        self.input_filename = None
        self.tensor = None
        self.output_path = None


    def generate_random_normal(self, shape, seed=1234):
        np.random.seed(seed)
        self.tensor = np.random.normal(0, 1, size=shape)
        self.output_path = 'output/random-normal_'
        self.output_path += 'shape-' + '-'.join([str(x) for x in shape]) + '_'
        self.output_path += 'seed-' + str(seed) + '/'
        return self.tensor


    def generate_random_cp(self, shape, rank, seed=1234):
        self.tensor = tl.random.random_cp(shape, rank, full=True, random_state=seed) 
        self.output_path = 'output/random-cp_'
        self.output_path += 'shape-' + '-'.join([str(x) for x in shape]) + '_'
        self.output_path += 'rank-' + str(rank) + '_'
        self.output_path += 'seed-' + str(seed) + '/'
        return self.tensor


    def generate_random_tucker(self, shape, rank, seed=1234):
        self.tensor = tl.random.random_tucker(shape, rank, full=True, random_state=seed)
        self.output_path = 'output/random-tucker_'
        self.output_path += 'shape-' + '-'.join([str(x) for x in shape]) + '_'
        self.output_path += 'rank-' + '-'.join([str(x) for x in rank]) + '_'
        self.output_path += 'seed-' + str(seed) + '/'
        return self.tensor

