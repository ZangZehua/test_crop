from train_CMC import main as train_cmc
from LinearProbing import main as linear_probing
from config import TrainCMCConfig, LinearProbingConfig, SAVE_PATH_BASE


TrainCMCConfig.crop_params = [10, 10, 204, 204]
train_cmc()
TrainCMCConfig.crop_params = [20, 20, 184, 184]
train_cmc()
TrainCMCConfig.crop_params = [30, 30, 164, 164]
train_cmc()
TrainCMCConfig.crop_params = [40, 40, 144, 144]
train_cmc()
TrainCMCConfig.crop_params = [50, 50, 124, 124]
train_cmc()
TrainCMCConfig.crop_params = [60, 60, 104, 104]
train_cmc()

LinearProbingConfig.model_path_base = SAVE_PATH_BASE + "models_pt/"

linear_probing()




