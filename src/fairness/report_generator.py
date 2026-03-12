"""Fairness Report Generator — Creates HTML audit reports."""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FairnessReportGenerator:
    def generate_html(self, fairness_report, model_metrics, output_path):
        overall_status = "PASS" if fairness_report.passes_all_thresholds() else "FAIL"
        status_color = "#22c55e" if overall_status == "PASS" else "#ef4444"

        metric_rows = ""
        for m in fairness_report.metrics:
            status_icon = "✓" if m.passed else "✗"
            row_color = "#f0fdf4" if m.passed else "#fef2f2"
            metric_rows += f'<tr style="background-color: {row_color}"><td>{status_icon} {m.name}</td><td>{m.protected_attribute}</td><td>{m.group_a} vs {m.group_b}</td><td><strong>{m.value:.4f}</strong></td><td>{m.threshold}</td><td>{m.severity}</td></tr>'

        group_stats_html = ""
        for attr, groups in fairness_report.group_statistics.items():
            group_stats_html += f"<h3>{attr.replace('_', ' ').title()}</h3><table><tr><th>Group</th><th>N</th><th>Actual Rate</th><th>Predicted Rate</th><th>Mean Prob</th></tr>"
            for gn, stats in groups.items():
                group_stats_html += f"<tr><td>{gn}</td><td>{stats['n_samples']}</td><td>{stats['positive_rate']:.3f}</td><td>{stats['predicted_positive_rate']:.3f}</td><td>{stats['mean_probability']:.3f}</td></tr>"
            group_stats_html += "</table>"

        html = f"""<!DOCTYPE html>
<html><head><title>Fairness Audit Report</title>
<style>body{{font-family:'Segoe UI',system-ui,sans-serif;max-width:900px;margin:40px auto;padding:20px;color:#1a1a2e}}h1{{border-bottom:3px solid #1a1a2e;padding-bottom:10px}}table{{border-collapse:collapse;width:100%;margin:15px 0}}th,td{{border:1px solid #ddd;padding:10px;text-align:left}}th{{background:#1a1a2e;color:white}}.status{{display:inline-block;padding:8px 20px;border-radius:6px;color:white;font-weight:bold;font-size:18px}}</style></head>
<body><h1>Fairness Audit Report</h1>
<p><strong>System:</strong> XAI Credit Scoring</p><p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>EU AI Act:</strong> High-Risk (Annex III, Section 5b)</p>
<h2>Status</h2><div class="status" style="background:{status_color}">{overall_status}</div>
<h2>Model Performance</h2><table><tr><th>Metric</th><th>Value</th></tr>{"".join(f'<tr><td>{k}</td><td>{v:.4f}</td></tr>' for k,v in model_metrics.items())}</table>
<h2>Fairness Metrics</h2><table><tr><th>Metric</th><th>Attribute</th><th>Groups</th><th>Value</th><th>Threshold</th><th>Severity</th></tr>{metric_rows}</table>
<h2>Group Statistics</h2>{group_stats_html}
</body></html>"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"Fairness report saved to {output_path}")
