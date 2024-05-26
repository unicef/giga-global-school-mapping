from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F

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
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, val_loader, test_loader, lr=0.01, max_iter=100):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        val_loader (DataLoader): validation set loader
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
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

            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = nll_criterion(logits, labels).item()
            before_temperature_ece = ece_criterion(logits, labels).item()
            print(f'Before temperature ({phase}) - NLL: %.3f, ECE: %.3f' % (
                before_temperature_nll, before_temperature_ece
            ))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(
                self.temperature_scale(data["val"]["logits"]), 
                data["val"]["labels"]
            )
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        print('Optimal temperature: %.3f' % self.temperature.item())
        for phase in ["val", "test"]:
            after_temperature_nll = nll_criterion(
                self.temperature_scale(data[phase]["logits"]), data[phase]["labels"]
            ).item()
            after_temperature_ece = ece_criterion(
                self.temperature_scale(data[phase]["logits"]), data[phase]["labels"]
            ).item()
            print(f'After temperature ({phase}) - NLL: %.3f, ECE: %.3f' % (
                after_temperature_nll, after_temperature_ece
            ))

        return self


class _ECELoss(nn.Module):
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
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece