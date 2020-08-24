import torch
import os

from basecamp.grain.grain import Grain

from models.bidate_model import BiDateNet
# from seldon_core.user_model import SeldonComponent


class PredictModel:
    def __init__(self):
        self.grain_exp = Grain()
        self.args = self.grain_exp.parse_args_from_json('metadata.json')
        self.model = BiDateNet(n_channels=len(self.args.band_ids), n_classes=1)
        self.model.load_state_dict(torch.load(os.environ.get('WEIGHTS_PATH'),
                                   map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, X, features_names, **kwargs):
        inp = torch.Tensor(X)
        with torch.no_grad():
            return self.model(inp).numpy()
