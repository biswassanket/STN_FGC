import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision import datasets, transforms

def get_events(path=None, modelname=None, key=None):
    """
    Return tensorboard events for a key as a pandas dataframe.

    Args:
      path: Path with tensorboard log directory.

      modelname: String to identify the model.

      key: Scalar value to retrieve from the logs.
    """

    event_acc = EventAccumulator(path)
    event_acc.Reload()
    tempdf = pd.DataFrame(event_acc.Scalars(key))
    tempdf['type'] = modelname

    return tempdf


def plot_metric(paths, names, key=None, fontsize=14, dot=False, logscale=False):
    """
    Plot a metric from tensorboard logs.

    Args:
      paths: List of tensorboard log directories.

      names: List with model names. Must be in the same order as `path`.

      fontsize: Font size for plotting.

      dot: If True, will dot discrete values on epoch-based plots.

      logscale: If True, the y-axis will be plotted on a log-scale.
    """

    with plt.style.context({'font.size': fontsize}):

        # Get the step events for the metrics as a data frame
        dfs = []
        for p, n in zip(paths, names):
          tmpdf = get_events(p, n, key)
          dfs.append(tmpdf)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(7,5))
        for m, n in zip(dfs, names):
          ax.plot(m['step'], m['value'], label=n)

          if dot:
              ax.scatter(m['step'], m['value'])

        if logscale:
          plt.yscale('log')

        plt.xlabel('Step')
        plt.ylabel(key)
        plt.legend()

        return fig, ax


def convert_image_np(inp):
    """
    Convert a Tensor to numpy image.
    This function is from Pytorch's spatial transformer tutorial.
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

    Args:
      inp: Torch tensor.
    """

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp


def compare_stns(model1=None, model2=None, model1name=None, model2name=None,
                 dataloader=None, device=None, figsize=(15,12)):
    """
    Evaluate one batch of MNIST data with two different models,
    and plot an image comparison grid of the original data vs. the STN
    transformation of two models. Models must have a `stn` forward function.

    Args:
      model1: First model (nn.Module).

      model2: Second model (nn.Module).

      model1name: String to identify the first model.

      model2name: String to identify the second model.

      dataloader: Dataloader from which the first batch of data will
        be evaluated.

      device: Device on which to run the computations.

      figsize: Tuple (w, h) for the plot size.
    """

    with torch.no_grad():
        data = next(iter(dataloader))[0].to(device)

        # Original data
        original_grid = convert_image_np(torchvision.utils.make_grid(
            data.cpu()))

        # Transformed input for the first model
        trans_baseline_stn_tensor = model1.stn(data).cpu()
        baseline_stn_grid = convert_image_np(
            torchvision.utils.make_grid(trans_baseline_stn_tensor))

        # Transformed input for the second model
        coordconv_stn_tensor = model2.stn(data).cpu()
        coordconv_stn_grid = convert_image_np(
            torchvision.utils.make_grid(coordconv_stn_tensor))

        # Plot the results side-by-side
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(original_grid)
        ax[0].set_title('Original input')

        ax[1].imshow(baseline_stn_grid)
        ax[1].set_title(model1name)

        ax[2].imshow(coordconv_stn_grid)
        ax[2].set_title(model2name)


def plot_wrong_preds(model1=None, model2=None, model1name=None, model2name=None,
                     dataloader=None, device=None, k=30, figsize=(15,12)):
    """
    Examine the k most incorrectly predicted samples according to model2,
    and plot the original data and STN transformations obtained by
    model1 and model2. Models must have a `stn` forward function.

    Args:
      model1: First model (nn.Module).

      model2: Second model (nn.Module).

      model1name: String to identify the first model.

      model2name: String to identify the second model.

      dataloader: Dataloader with samples to evaluate.

      device: Device on which to run the computations.

      k: Number of top-k wrong predictions to plot.

      figsize: Tuple (w, h) for the plot size.
    """

    model1_logits = []
    model2_logits = []
    targets = []

    # Evaluate all the samples from the dataloader
    # and store logits and predictions
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            data, target = batch
            data = data.to(device)

            # Forward pass
            m1 = model1(data).cpu()
            m2 = model2(data).cpu()

            # Store logits
            model1_logits.append(m1)
            model2_logits.append(m2)

            # Store predictions
            targets.append(target.cpu())
    model1_logits = np.vstack(model1_logits)
    model2_logits = np.vstack(model2_logits)
    targets = np.concatenate(targets)

    # Find the top-k most incorrect predictions according to model2
    x_logits = model2_logits
    preds = x_logits.argmax(axis=1)
    pred_df = pd.DataFrame.from_records(zip(preds, targets, preds==targets,
                                            np.exp(x_logits[np.arange(
                                                x_logits.shape[0]), preds])),
                                        columns=['predictions', 'target',
                                                 'correct', 'pred_prob'])
    incorrects = pred_df[~pred_df.correct]
    incorrects = incorrects.sort_values('pred_prob', ascending=False)
    worst_predictions = incorrects.index[0:k].values

    # Get the tensors for the incorrect predictions
    imgs = []
    val_tmp = datasets.MNIST(root='.', train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))]))

    for incorrect_idx in worst_predictions:
        x, y = val_tmp[incorrect_idx]
        imgs.append(x)
    imgs = torch.cat(imgs).unsqueeze(1)

    # Plot
    original_grid = convert_image_np(torchvision.utils.make_grid(imgs))
    model1_grid = convert_image_np(torchvision.utils.make_grid(
        model1.stn(imgs.to(device)).cpu()))
    model2_grid = convert_image_np(torchvision.utils.make_grid(
        model2.stn(imgs.to(device)).cpu()))

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(original_grid)
    ax[0].set_title('Original input')

    ax[1].imshow(model1_grid)
    ax[1].set_title(model1name)

    ax[2].imshow(model2_grid)
    ax[2].set_title(model2name)

