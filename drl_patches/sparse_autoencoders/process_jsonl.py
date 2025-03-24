import argparse
import json

import numpy as np
import plotly.graph_objects as go
import torch
from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.schemas import PlotType
from drl_patches.sparse_autoencoders.utils import imshow, line
from transformer_lens import HookedTransformer


def plot_with_confidence(
    logit_diff, labels, confidence_intervals, title="Confidence Visualization"
):
    """Visualize logit differences with confidence intervals."""
    fig = go.Figure()

    # Main line plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(logit_diff))),
            y=logit_diff,
            mode="lines+markers",
            name="Logit Difference",
            hovertext=labels,
        )
    )

    # Add confidence interval (shaded region)
    lower_bound = [ld - ci for ld, ci in zip(logit_diff, confidence_intervals)]
    upper_bound = [ld + ci for ld, ci in zip(logit_diff, confidence_intervals)]

    fig.add_trace(
        go.Scatter(
            x=list(range(len(logit_diff))) + list(range(len(logit_diff)))[::-1],
            y=lower_bound + upper_bound[::-1],
            fill="toself",
            fillcolor="rgba(0,100,200,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Logit Difference",
        template="plotly_white",
    )
    fig.update_yaxes(range=[0, 100])

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the layers of a model through mechanistic interpretability"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON lines file"
    )

    args = parser.parse_args()

    n_lines = 0
    avgs = {}
    all_lines = []  # Collect all lines for variance calculation

    # Read and process input
    with open(args.input, "r") as f:
        for line_number, fline in enumerate(f, 1):
            try:
                line_content = json.loads(fline)
                all_lines.append(line_content)

                if n_lines == 0:
                    avgs["values"] = line_content["values"]
                    avgs["labels"] = line_content["labels"]
                    avgs["model"] = line_content["model"]
                    avgs["plot_type"] = line_content["plot_type"]
                else:
                    # for i, logit_diff in enumerate(line_content["logit_diff"]):
                    # avgs["logit_diff"][i] += logit_diff
                    avgs["values"] = np.sum(
                        [avgs["values"], line_content["values"]], axis=0
                    )

                n_lines += 1
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON on line {line_number}: {fline.strip()}"
                )

    if n_lines == 0:
        raise ValueError("No valid data found in the input file.")

    # Average the logit differences
    for i, logit_diff in enumerate(avgs["values"]):
        avgs["values"][i] /= n_lines

    with torch.no_grad():
        print("Disabled automatic differentiation")

    if avgs["plot_type"] == PlotType.ATTENTION.value:
        imshow(
            avgs["values"],
            labels={"x": "Head", "y": "Layer"},
            title="Logit Difference From Each Head",
        )
    elif avgs["plot_type"] == PlotType.SAE_FEATURE_IMPORTANCE.value:
        fig = line(
            avgs["values"],
            title="Average Feature Activation Difference",
            labels={"index": "Feature", "value": "Activation"},
        )
    else:
        # Calculate confidence intervals
        conf_intervals = [
            np.std([line_content["values"][i] for line_content in all_lines])
            / np.sqrt(n_lines)
            for i in range(len(avgs["values"]))
        ]
        # Visualize with confidence intervals
        plot_with_confidence(
            logit_diff=avgs["values"],
            labels=avgs["labels"],
            confidence_intervals=conf_intervals,
            title="Average Logit Difference with Confidence Intervals",
        )

    logger.info("Processing complete. Results saved to output.")
