import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import joblib
import post_utils
from torch.utils.data import Dataset

import logging
logging.basicConfig(level=logging.INFO)

SEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.metrics import CE


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model.eval()
        self.ece = CE(task="multiclass", num_classes=2) #ECELoss().to(device)
        self.scaler = TemperatureScaler(self.model, lr=lr)

    
    def forward(self, input):
        return self.scaler(input)

    
    # This function probably should live outside of this class, but whatever
    def set_temperature(self, val_loader, test_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        val_loader (DataLoader): validation set loader
        """
        self.to(device)
        val_loader.dataset.set_return_uid(False)
        test_loader.dataset.set_return_uid(False)

        # First: collect all the logits and labels for the validation set
        # Calculate NLL and ECE before temperature scaling
        for phase, data_loader in zip(["val", "test"], [val_loader, test_loader]):
            with torch.no_grad():
                for input, label in tqdm(data_loader, total=len(data_loader)):
                    input = input.to(device)
                    logits = self.model(input)
                    self.ece.update(logits.softmax(-1), label)
            
            cal = self.ece.compute()
            print(f"{phase} ECE before scaling - {cal:.4f}%.")
        
        self.scaler = self.scaler.fit(calibration_set=val_loader.dataset)
        self.ece.reset()
        
        # Iterate on the test dataloader
        for phase, data_loader in zip(["val", "test"], [val_loader, test_loader]):
            with torch.no_grad():
                for input, label in tqdm(data_loader, total=len(data_loader)):
                    logits = self.scaler(input)
                    self.ece.update(logits.softmax(-1), label)
        
            cal = self.ece.compute()
            print(f"{phase} ECE after scaling - {cal:.2f}%.")


class ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        self.n_bins = n_bins

    def calc_bins(self, preds, labels):
        bins = np.linspace(0.1, 1, self.n_bins)
        binned = np.digitize(preds, bins)
        
        bin_accs = np.zeros(self.n_bins)
        bin_confs = np.zeros(self.n_bins)
        bin_sizes = np.zeros(self.n_bins)

        for bin in range(self.n_bins):
            bin_sizes[bin] = len(preds[binned == bin])
            if bin_sizes[bin] > 0:
              bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
              bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]
        
        return bins, binned, bin_accs, bin_confs, bin_sizes

    def calculate_ece(self, preds, labels):
        ECE, MCE = 0, 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calc_bins(preds, labels)
        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)
        return ECE, MCE

    def draw_reliability_graph(self, preds, labels):
        ECE, MCE = self.calculate_ece(preds, labels)
        bins, _, bin_accs, _, _ = self.calc_bins(preds, labels)

        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(-0.05, 1)
        ax.set_xticks([x*0.1 for x in range(0, 11)])
        ax.set_ylim(0, 1)
        ax.set_yticks([x*0.1 for x in range(0, 11)])

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
    
        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')
    
        # Error bars
        bins = torch.linspace(0, 1, self.n_bins + 1)
        width = 1.0 / self.n_bins
        bin_centers = np.linspace(0, 1.0, self.n_bins+1) - width / 2
        #bin_centers = np.append(bin_centers, 1 + width) - 0.1
        bin_accs = np.insert(bin_accs, 0, 0)
        plt.bar(bin_centers, bins, width=width, alpha=0.3, edgecolor='black', color='r', hatch='\\')
    
        # Draw bars and identity line
        plt.bar(bin_centers, bin_accs, width=width, alpha=1, edgecolor='black', color='b')
        plt.plot([- width/2,1],[0,1], '--', color='gray', linewidth=2)
    
        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')
    
        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        plt.legend(handles=[ECE_patch, MCE_patch])

    
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        preds = softmaxes[:, 1].detach().numpy()
        ECE, MCE = self.calculate_ece(preds, labels)
        return ECE


def isotonic_regressor(iso_code, config, output=None, n_bins=10):
    calibration = "isoreg"

    cwd = os.path.dirname(os.getcwd())
    exp_dir = os.path.join(
        cwd, config["exp_dir"], config["project"], f"{iso_code}_{config['config_name']}"
    )
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}_{calibration}.pkl")

    if os.path.exists(model_file):
        regressor = joblib.load(model_file)
        logging.info(f"Model loaded from {model_file}")
        return regressor
    
    if not output:
        output = post_utils.get_results(iso_code, config, phase="val")
        
    prob_true, prob_pred = calibration_curve(
        output["y_true"], output["y_probs"], n_bins=n_bins
    )
    regressor = IsotonicRegression(out_of_bounds="clip")
    regressor.fit(prob_pred, prob_true)
    
    joblib.dump(regressor, model_file) 
    logging.info(f"Model saved to {model_file}")
    return regressor