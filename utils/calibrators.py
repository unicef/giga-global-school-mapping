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

import logging
logging.basicConfig(level=logging.INFO)

SEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.temperature = nn.Parameter(torch.ones(1).to(device))
        self.nll_criterion = nn.CrossEntropyLoss().to(device)
        self.ece_criterion = ECELoss().to(device)

    
    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return torch.div(logits, temperature)

    
    def eval_temp(self, data):
        for phase in ["val", "test"]:
            after_temperature_nll = self.nll_criterion(
                self.temperature_scale(data[phase]["logits"]), data[phase]["labels"]
            ).item()
            after_temperature_ece = self.ece_criterion(
                self.temperature_scale(data[phase]["logits"]), data[phase]["labels"]
            ).item()
            print(f'After temperature ({phase}) - NLL: %.4f, ECE: %.4f' % (
                after_temperature_nll, after_temperature_ece
            ))


    def get_logits(self, val_loader, test_loader):
        self.to(device)

        data = dict()
        for phase, data_loader in zip(["val", "test"], [val_loader, test_loader]):
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label, _ in tqdm(data_loader, total=len(data_loader)):
                    input = input.to(device)
                    logits = self.model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
                logits = torch.cat(logits_list).to(device)
                labels = torch.cat(labels_list).to(device)
            data[phase] = {
                "logits": logits,
                "labels": labels
            }
            before_temperature_nll = self.nll_criterion(logits, labels).item()
            before_temperature_ece = self.ece_criterion(logits, labels).item()
            print(f'Before temperature ({phase}) - NLL: %.4f, ECE: %.4f' % (
                before_temperature_nll, before_temperature_ece
            ))
            
        return data

    
    # This function probably should live outside of this class, but whatever
    def set_temperature(self, val_loader, test_loader, lr=0.01, max_iter=1000):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        val_loader (DataLoader): validation set loader
        """
        self.to(device)

        # First: collect all the logits and labels for the validation set
        # Calculate NLL and ECE before temperature scaling
        data = self.get_logits(val_loader, test_loader)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS(
            [self.temperature], 
            lr=lr, 
            max_iter=max_iter, 
            line_search_fn='strong_wolfe'
        )

        def eval():
            optimizer.zero_grad()
            loss = self.nll_criterion(
                self.temperature_scale(data["val"]["logits"]), 
                data["val"]["labels"]
            )
            loss.backward()
            return loss
            
        optimizer.step(eval)
        print('Optimal temperature: %.3f' % self.temperature.item())

        self.eval_temp(data)
        return self


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
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
    
        fig = plt.figure(figsize=(4, 6))
        ax = fig.gca()
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')
    
        bins = torch.linspace(0, 1, self.n_bins + 1)
        width = 1.0 / self.n_bins
        bin_centers = np.linspace(0, 1.0 - width, self.n_bins) + width / 2
        bin_centers = np.append(bin_centers, 1 + + width / 2) - 0.1
        bin_accs = np.insert(bin_accs, 0, 0)
    
        # Error bars
        plt.bar(
            bin_centers, 
            bins,
            width=width, 
            alpha=0.3, 
            edgecolor='black', 
            color='r', 
            hatch='\\', 
            linewidth=0.5
        )
        plt.bar(
            bin_centers, 
            bin_accs, 
            width=1/self.n_bins, 
            alpha=1, 
            edgecolor='black', 
            color='b', 
            linewidth=0.5
        )
        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)
        plt.gca().set_aspect('equal', adjustable='box')
    
        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.4f}'.format(ECE))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.4f}'.format(MCE))
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