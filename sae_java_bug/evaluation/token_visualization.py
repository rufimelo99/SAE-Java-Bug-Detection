"""
Token-level visualization of feature violations in code snippets.

Generates interactive HTML visualizations showing which tokens activated
the features that violated invariants.
"""

import base64
from typing import Dict, List, Set, Tuple

import numpy as np


def decode_b64(s: str) -> str:
    """Decode base64-encoded strings."""
    try:
        return base64.b64decode(s).decode("utf-8")
    except Exception:
        return s


def simple_tokenize(code: str) -> List[str]:
    """Simple tokenization for visualization purposes."""
    # Split by common delimiters but preserve them
    import re
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return tokens


def generate_token_html_visualization(
    vuln_id: str,
    secure_code: str,
    vulnerable_code: str,
    secure_features: List[float],
    vulnerable_features: List[float],
    invariants: List[Tuple[int, int, float]],
    violated_indices: List[Tuple[int, int, float, float]],
    cwe: str,
    secure_score: float,
    vulnerable_score: float,
    delta: float,
) -> str:
    """
    Generate HTML visualization for a case study with token highlighting.
    
    Args:
        vuln_id: Vulnerability identifier
        secure_code: Secure code snippet
        vulnerable_code: Vulnerable code snippet
        secure_features: Feature activations for secure code
        vulnerable_features: Feature activations for vulnerable code
        invariants: List of (i, j, P_ij) invariants
        violated_indices: List of (i, j, P_ij, weight) violated invariants
        cwe: CWE classification
        secure_score: Violation score for secure code
        vulnerable_score: Violation score for vulnerable code
        delta: Difference in scores
    
    Returns:
        HTML string
    """
    
    # Get the top violated features
    violated_features_set = set(i for i, j, _, _ in violated_indices)
    violated_features_sorted = sorted(
        violated_features_set,
        key=lambda f: max((w for i, j, p, w in violated_indices if i == f), default=0),
        reverse=True
    )[:10]  # Top 10 violated features
    
    # Create color palette
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#ABEBC6"
    ]
    feature_colors = {
        feature: colors[idx % len(colors)]
        for idx, feature in enumerate(violated_features_sorted)
    }
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                border-bottom: 2px solid #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
            }}
            .header h1 {{
                margin: 0 0 5px 0;
                color: #333;
            }}
            .metadata {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .metadata-item {{
                border-left: 3px solid #007bff;
                padding-left: 10px;
            }}
            .metadata-label {{
                font-weight: bold;
                color: #555;
                font-size: 0.85em;
                text-transform: uppercase;
            }}
            .metadata-value {{
                color: #333;
                font-size: 1.1em;
            }}
            .score-positive {{
                color: #d9534f;
                font-weight: bold;
            }}
            .code-section {{
                margin-bottom: 30px;
            }}
            .code-section h3 {{
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
                margin-bottom: 10px;
            }}
            .code-area {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 0.95em;
                line-height: 1.6;
                overflow-x: auto;
            }}
            .token {{
                display: inline-block;
                padding: 2px 4px;
                margin: 1px 1px;
                border-radius: 3px;
                white-space: nowrap;
                position: relative;
                cursor: help;
                border: 1px solid transparent;
            }}
            .token:hover {{
                border: 1px solid #333;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);
            }}
            .token-default {{
                background-color: #ffffff;
                color: #333;
            }}
            .violations-section {{
                margin-top: 20px;
            }}
            .violations-list {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
            }}
            .violation-item {{
                background-color: #f9f9f9;
                border-left: 4px solid;
                padding: 12px;
                border-radius: 4px;
            }}
            .violation-title {{
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }}
            .violation-detail {{
                font-size: 0.9em;
                color: #666;
                margin: 3px 0;
            }}
            .legend {{
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 4px;
            }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
            }}
            .legend-items {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 1px solid #ddd;
            }}
            .tooltip {{
                position: absolute;
                background-color: #333;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 0.85em;
                white-space: nowrap;
                z-index: 1000;
                pointer-events: none;
                display: none;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
            }}
            .token:hover .tooltip {{
                display: block;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
            }}
            .stat-box.positive {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}
            .stat-label {{
                font-size: 0.85em;
                opacity: 0.9;
                margin-bottom: 5px;
            }}
            .stat-value {{
                font-size: 1.8em;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Feature Violation Case Study: {vuln_id}</h1>
                <p>Token-level visualization of activated features that violated invariants</p>
            </div>
            
            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">CWE</div>
                    <div class="metadata-value">{cwe}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Secure Score</div>
                    <div class="metadata-value">{secure_score:.4f}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Vulnerable Score</div>
                    <div class="metadata-value">{vulnerable_score:.4f}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Delta (V - S)</div>
                    <div class="metadata-value score-positive">{delta:+.4f}</div>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Invariants Extracted</div>
                    <div class="stat-value">{len(invariants)}</div>
                </div>
                <div class="stat-box positive">
                    <div class="stat-label">Violated in Vulnerable</div>
                    <div class="stat-value">{len(violated_indices)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Top Violated Features</div>
                    <div class="stat-value">{len(violated_features_sorted)}</div>
                </div>
            </div>
            
            <div class="code-section">
                <h3>Secure Code</h3>
                <div class="code-area">
                    {_tokenize_and_highlight_html(secure_code, [], feature_colors)}
                </div>
            </div>
            
            <div class="code-section">
                <h3>Vulnerable Code</h3>
                <div class="code-area">
                    {_tokenize_and_highlight_html(vulnerable_code, list(violated_features_set), feature_colors)}
                </div>
            </div>
            
            <div class="violations-section">
                <h3>Top Violated Invariants</h3>
                <div class="violations-list">
    """
    
    # Add violated invariants
    for i, j, pij, w in sorted(violated_indices, key=lambda x: x[3], reverse=True)[:10]:
        color = feature_colors.get(i, "#cccccc")
        html += f"""
                    <div class="violation-item" style="border-color: {color};">
                        <div class="violation-title">
                            Feature {i} → Feature {j}
                        </div>
                        <div class="violation-detail">
                            <strong>P(j|i):</strong> {pij:.4f}
                        </div>
                        <div class="violation-detail">
                            <strong>Weight:</strong> {w:.2f}
                        </div>
                    </div>
        """
    
    html += """
                </div>
            </div>
            
            <div class="legend">
                <div class="legend-title">Feature Legend</div>
                <div class="legend-items">
    """
    
    # Add legend
    for feature in violated_features_sorted:
        color = feature_colors[feature]
        html += f"""
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {color};"></div>
                        <span>Feature {feature}</span>
                    </div>
        """
    
    html += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def _tokenize_and_highlight_html(
    code: str,
    active_features: List[int],
    feature_colors: Dict[int, str],
) -> str:
    """
    Tokenize code and wrap tokens in HTML with highlighting for active features.
    """
    tokens = simple_tokenize(code)
    html_tokens = []
    
    for token in tokens:
        # For now, just wrap in generic token div
        html_tokens.append(f'<span class="token token-default">{token}</span>')
    
    return "".join(html_tokens)


def generate_case_study_report(
    metadata_list: List[Dict],
    secure_features_list: List[List[float]],
    vulnerable_features_list: List[List[float]],
    secure_scores: np.ndarray,
    vulnerable_scores: np.ndarray,
    deltas: np.ndarray,
    invariants: List[Tuple[int, int, float]],
    violated_indices_per_sample: List[List[Tuple[int, int, float, float]]],
    output_dir: str = "sae_java_bug/artifacts/visualizations/",
    num_cases: int = 15,
) -> List[str]:
    """
    Generate HTML reports for top case studies and return list of file paths.
    
    Args:
        metadata_list: List of metadata dicts
        secure_features_list: List of secure feature activations
        vulnerable_features_list: List of vulnerable feature activations
        secure_scores: Array of secure violation scores
        vulnerable_scores: Array of vulnerable violation scores
        deltas: Array of score differences
        invariants: List of extracted invariants
        violated_indices_per_sample: Per-sample violated invariants
        output_dir: Directory to save HTML files
        num_cases: Number of top cases to generate
    
    Returns:
        List of generated HTML file paths
    """
    import os
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find positive delta cases
    positive_indices = np.where(deltas > 0)[0]
    sorted_by_delta = positive_indices[np.argsort(deltas[positive_indices])[::-1]]
    
    generated_files = []
    seen_cwes = set()
    case_num = 0
    
    for idx in sorted_by_delta:
        m = metadata_list[idx]
        cwe = m["cwe"]
        
        # Prefer diverse CWEs first
        if case_num < num_cases // 2 and cwe in seen_cwes:
            continue
        seen_cwes.add(cwe)
        case_num += 1
        
        if case_num > num_cases:
            break
        
        # Decode code
        secure_code = decode_b64(m["secure_code"])
        vulnerable_code = decode_b64(m["vulnerable_code"])
        
        # Get violations
        violated = violated_indices_per_sample[idx]
        
        # Generate HTML
        html = generate_token_html_visualization(
            vuln_id=m["vuln_id"],
            secure_code=secure_code,
            vulnerable_code=vulnerable_code,
            secure_features=secure_features_list[idx],
            vulnerable_features=vulnerable_features_list[idx],
            invariants=invariants,
            violated_indices=violated,
            cwe=cwe,
            secure_score=float(secure_scores[idx]),
            vulnerable_score=float(vulnerable_scores[idx]),
            delta=float(deltas[idx]),
        )
        
        # Save to file
        filename = f"case_study_{case_num:02d}_{m['vuln_id'][:30]}.html"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(html)
        
        generated_files.append(filepath)
    
    return generated_files
