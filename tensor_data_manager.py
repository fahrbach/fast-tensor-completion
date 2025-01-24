import numpy as np
import scipy.io as sio
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


    def load_cardiac_mri(self):
        self.input_filename = 'data/cardiac-mri/sol_yxzt_pat1.mat'
        self.tensor = sio.loadmat(self.input_filename)['sol_yxzt'].astype(float)
        self.output_path = 'output/cardiac-mri/'
        assert self.tensor.shape == (256, 256, 14, 20)
        assert self.tensor.size == 18_350_080
        return self.tensor


    def load_hyperspectral(self):
        """
        https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1140.zip
        """
        self.input_filename = 'data/hyperspectral/nogueiro_1140.mat'
        self.tensor = sio.loadmat(self.input_filename)['hsi'].astype(float)
        self.output_path = 'output/hyperspectral/'
        assert self.tensor.shape == (1024, 1344, 33)
        assert self.tensor.size == 45_416_448
        return self.tensor

    def load_traffic(self):
        """
        Paper: "Traffic forecasting in complex urban networks: Leveraging big data and machine learning"
        Source: https://github.com/florinsch/BigTrafficData
        """
        self.input_filename = 'data/traffic/VolumeData_tensor.mat'
        self.tensor = sio.loadmat(self.input_filename)['data'].astype(float)
        self.output_path = 'output/traffic/'
        assert self.tensor.shape == (1084, 2033, 96)
        assert self.tensor.size == 211_562_112
        return self.tensor

