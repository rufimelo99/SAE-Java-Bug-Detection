import argparse
import json

import numpy as np
import plotly.graph_objects as go
import torch
from transformer_lens import HookedTransformer

from drl_patches.logger import logger


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

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the layers of a model through mechanistic interpretability"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON lines file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output JSON file"
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
                    avgs["logit_diff"] = line_content["logit_diff"]
                    avgs["labels"] = line_content["labels"]
                    avgs["model"] = line_content["model"]
                else:
                    for i, logit_diff in enumerate(line_content["logit_diff"]):
                        avgs["logit_diff"][i] += logit_diff

                n_lines += 1
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON on line {line_number}: {fline.strip()}"
                )

    if n_lines == 0:
        raise ValueError("No valid data found in the input file.")

    # Average the logit differences
    for i, logit_diff in enumerate(avgs["logit_diff"]):
        avgs["logit_diff"][i] /= n_lines

    # Calculate confidence intervals
    conf_intervals = [
        np.std([line_content["logit_diff"][i] for line_content in all_lines])
        / np.sqrt(n_lines)
        for i in range(len(avgs["logit_diff"]))
    ]

    # Load model
    model = HookedTransformer.from_pretrained(
        avgs["model"],
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

    with torch.no_grad():
        print("Disabled automatic differentiation")

    # Visualize with confidence intervals
    plot_with_confidence(
        logit_diff=avgs["logit_diff"],
        labels=avgs["labels"],
        confidence_intervals=conf_intervals,
        title="Average Logit Difference with Confidence Intervals",
    )

    # Save output
    with open(args.output, "w") as f:
        json.dump(avgs, f)

    logger.info("Processing complete. Results saved to output.")
