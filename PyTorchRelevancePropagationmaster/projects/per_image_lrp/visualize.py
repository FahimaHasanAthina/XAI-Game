"""Script with plot method for visualization of relevance scores.
"""
import argparse
import matplotlib.pyplot as plt
import torch


def plot_relevance_scores(
    x: torch.tensor, r: torch.tensor, name: str, outdir: str
) -> None:
    """Plots results from layer-wise relevance propagation next to original image.

    Method currently accepts only a batch size of one.

    Args:
        x: Original image.
        r: Relevance scores for original image.
        name: Image name.
        config: Argparse namespace object.

    """
    output_dir = outdir

    max_fig_size = 20

    _, _, img_height, img_width = x.shape
    max_dim = max(img_height, img_width)
    fig_height, fig_width = (
        max_fig_size * img_height / max_dim,
        max_fig_size * img_width / max_dim,
    )

    #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height))

    # x = x[0].squeeze().permute(1, 2, 0).detach().cpu()
    # x_min = x.min()
    # x_max = x.max()
    # x = (x - x_min) / (x_max - x_min)
    # axes[0].imshow(x)
    # axes[0].set_axis_off()

    plt.figure()

    r_min = r.min()
    r_max = r.max()
    r = (r - r_min) / (r_max - r_min)
    plt.imshow(r, cmap="afmhot")
    plt.axis('off')

    plt.savefig(f"{output_dir}/{name}.png", bbox_inches="tight")
    plt.close()
