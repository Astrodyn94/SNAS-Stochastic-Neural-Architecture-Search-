class BaseOptions():
    def initialize(self):
        self.dataroot = '../' # path to the dir of the dataset
        self.name = 'Default' # Name of the experiment

class TrainOptions(BaseOptions):
    def __init__(self):

        BaseOptions.initialize(self)
        self.batch_size = 16
        self.learning_rate = 0.025
        self.learning_rate_min = 0.001
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.report_freq = 50
        self.epochs = 150
        self.init_channels = 16
        self.layers = 16
        self.model_path = 'saved_models'
        self.cutout = True # use cutout
        self.cutout_length = 16 # cutout length
        self.arch_learning_rate = 3e-4
        self.arch_weight_decay = 1e-3
        self.initial_temp = 2.5 # innitial softmax temperature
        self.anneal_rate = 0.00003 # annealation rate of softmax temperature
        self.train_portion = 0.5
