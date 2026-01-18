"""
HTML templates and generation methods for the evaluation report.

This module contains all the HTML generation logic, templates, and styling.
"""

import html
from typing import Dict, List, Optional, Any
from datetime import datetime


class ReportHTMLGenerator:
    """Generates HTML sections for the evaluation report."""

    def __init__(self, generator):
        """
        Initialize with reference to the main generator.

        Args:
            generator: EvaluationReportGenerator instance with all the data
        """
        self.gen = generator

    def generate_html(self) -> str:
        """Generate the complete HTML report."""
        overall_stats = self._compute_overall_stats()
        compliance_status, compliance_summary, compliance_issues = self._get_compliance_status(overall_stats)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hallucination Evaluation Report</title>
    {self._generate_styles()}
</head>
<body>
    <div class="document">
        {self._generate_cover_page(compliance_status)}
        {self._generate_executive_summary(overall_stats, compliance_status, compliance_summary, compliance_issues)}
        {self._generate_dataset_section()}
        {self._generate_devtest_split_section()}
        {self._generate_sample_analysis_section()}
        {self._generate_appendix()}
        {self._generate_footer()}
    </div>
    {self._generate_scripts()}
</body>
</html>"""

    def _compute_overall_stats(self) -> Dict[str, Any]:
        """Compute overall statistics across all metrics."""
        import numpy as np

        if not self.gen.metric_results:
            return {}

        results = list(self.gen.metric_results.values())

        return {
            'avg_accuracy': np.mean([r.accuracy for r in results]),
            'avg_precision': np.mean([r.precision for r in results]),
            'avg_recall': np.mean([r.recall for r in results]),
            'avg_f1': np.mean([r.f1 for r in results]),
            'total_support': sum(r.support for r in results),
            'total_correct': sum(r.correct_count for r in results),
            'total_incorrect': sum(r.incorrect_count for r in results),
            'total_null': sum(r.null_count for r in results),
            'total_errors': sum(r.error_count for r in results),
            'metrics_count': len(results),
            'min_accuracy': min(r.accuracy for r in results),
            'max_accuracy': max(r.accuracy for r in results),
        }

    def _get_compliance_status(self, stats: Dict) -> tuple:
        """Determine overall compliance status."""
        if not stats:
            return ('UNKNOWN', 'No metrics available', [])

        issues = []
        if stats['avg_accuracy'] < self.gen.thresholds['accuracy']:
            issues.append(f"Average accuracy ({stats['avg_accuracy']:.1%}) below threshold ({self.gen.thresholds['accuracy']:.1%})")
        if stats['avg_precision'] < self.gen.thresholds['precision']:
            issues.append(f"Average precision ({stats['avg_precision']:.1%}) below threshold ({self.gen.thresholds['precision']:.1%})")
        if stats['avg_recall'] < self.gen.thresholds['recall']:
            issues.append(f"Average recall ({stats['avg_recall']:.1%}) below threshold ({self.gen.thresholds['recall']:.1%})")

        if not issues:
            return ('PASS', 'All metrics meet or exceed defined thresholds', issues)
        elif len(issues) <= 2:
            return ('CONDITIONAL', 'Some metrics require attention', issues)
        else:
            return ('REQUIRES REVIEW', 'Multiple metrics below threshold', issues)

    def _generate_styles(self) -> str:
        """Generate link to external CSS file or embed styles."""
        import os

        if self.gen.embed_css:
            css_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.css')
            try:
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                return f"<style>\n{css_content}\n</style>"
            except FileNotFoundError:
                return self._get_fallback_styles()
        else:
            return '<link rel="stylesheet" href="evaluation_report.css">'

    def _get_fallback_styles(self) -> str:
        """Return minimal fallback styles if CSS file is not found."""
        return """<style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .document { max-width: 900px; margin: 0 auto; }
        .cover-page { background: #1e3a8a; color: white; padding: 50px; margin-bottom: 20px; }
        .cover-title { font-size: 28pt; font-weight: 600; margin-bottom: 8px; }
        .cover-subtitle { font-size: 12pt; color: rgba(255,255,255,0.7); margin-bottom: 40px; }
        .cover-meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px 40px; max-width: 700px; margin: 0 auto 40px; }
        .meta-item { display: flex; flex-direction: column; gap: 4px; }
        .meta-label { font-size: 9pt; color: rgba(255,255,255,0.5); text-transform: uppercase; }
        .meta-value { font-size: 11pt; font-weight: 500; }
        .cover-status { display: inline-block; padding: 12px 32px; font-weight: 600; border-radius: 4px; }
        .cover-status.pass { background: #059669; }
        .cover-status.conditional { background: #d97706; }
        .cover-status.review { background: #dc2626; }
        section { margin-bottom: 30px; padding: 20px; border: 1px solid #e5e7eb; }
        .data-table { width: 100%; border-collapse: collapse; }
        .data-table th, .data-table td { padding: 10px; border: 1px solid #e5e7eb; text-align: left; }
        .data-table th { background: #1e3a8a; color: white; }
    </style>"""

    def _generate_cover_page(self, compliance_status: str) -> str:
        """Generate a professional cover page with essential metadata."""
        status_class = 'pass' if compliance_status == 'PASS' else ('conditional' if compliance_status == 'CONDITIONAL' else 'review')

        # Add DRAFT to classification if not already present
        classification = self.gen.report_metadata.classification
        if 'DRAFT' not in classification.upper():
            classification = f"{classification} - DRAFT"

        # Get additional dataset info
        additional_info = self.gen.dataset_info.additional_info
        model_name = additional_info.get('Model Under Test', 'AI Assistant')
        eval_framework = additional_info.get('Evaluation Framework', 'Evaluation Pipeline')
        dev_samples = additional_info.get('Dev Samples', 'N/A')
        test_samples = additional_info.get('Test Samples', 'N/A')

        return f"""
        <div class="cover-page">
            <div class="cover-classification">{html.escape(classification)}</div>

            <div class="cover-content">
                <h1 class="cover-title">{html.escape(self.gen.dataset_info.name)}</h1>
                <p class="cover-subtitle">Model Evaluation Report</p>

                <div class="cover-meta-grid">
                    <div class="meta-item">
                        <span class="meta-label">Run Id</span>
                        <span class="meta-value">{html.escape(self.gen.report_metadata.document_id)}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Generated</span>
                        <span class="meta-value">{self.gen.generated_at.strftime('%B %d, %Y')}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Model Under Test</span>
                        <span class="meta-value">{html.escape(str(model_name))}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Total Samples</span>
                        <span class="meta-value">{self.gen.dataset_info.size:,}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Dev / Test Split</span>
                        <span class="meta-value">{dev_samples:,} / {test_samples:,}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Prepared By</span>
                        <span class="meta-value">{html.escape(self.gen.report_metadata.prepared_by)}</span>
                    </div>
                </div>

                <div class="cover-status {status_class}">
                    Evaluation Status: {compliance_status}
                </div>
            </div>
        </div>"""

    def _generate_table_of_contents(self) -> str:
        """Generate table of contents."""
        return """
        <section class="toc page-break">
            <h2 class="section-title">Table of Contents</h2>
            <ul class="toc-list">
                <li class="toc-item">
                    <span class="toc-number">1</span>
                    <span class="toc-text">Executive Summary</span>
                </li>
                <li class="toc-item">
                    <span class="toc-number">2</span>
                    <span class="toc-text">Methodology</span>
                </li>
                <li class="toc-item">
                    <span class="toc-number">3</span>
                    <span class="toc-text">Dataset Information</span>
                </li>
                <li class="toc-item">
                    <span class="toc-number">4</span>
                    <span class="toc-text">Evaluation Results</span>
                </li>
                <li class="toc-item">
                    <span class="toc-number">5</span>
                    <span class="toc-text">Detailed Metric Analysis</span>
                </li>
                <li class="toc-item">
                    <span class="toc-number">6</span>
                    <span class="toc-text">Sample Analysis</span>
                </li>
                <li class="toc-item">
                    <span class="toc-number">A</span>
                    <span class="toc-text">Appendix: Definitions & Methodology</span>
                </li>
            </ul>
        </section>"""

    def _generate_executive_summary(self, stats: Dict, status: str, summary: str, issues: List[str]) -> str:
        """Generate executive summary section."""
        if not stats:
            return ""

        issues_html = ""
        if issues:
            issues_items = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
            issues_html = f"""
                    <div class="issues-section">
                        <h5>Items Requiring Attention</h5>
                        <ul>{issues_items}</ul>
                    </div>"""

        findings_html = ""
        for metric_name, result in self.gen.metric_results.items():
            indicator_class = "success" if result.accuracy >= self.gen.thresholds['accuracy'] else ("warning" if result.accuracy >= self.gen.thresholds['accuracy'] * 0.9 else "danger")
            findings_html += f"""
                    <div class="finding-item">
                        <span class="finding-indicator {indicator_class}"></span>
                        <span><strong>{html.escape(result.display_name)}:</strong> {result.accuracy:.1%} accuracy ({result.correct_count}/{result.support} samples)</span>
                    </div>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 1</div>
            <h2 class="section-title">Executive Summary</h2>

            <div class="exec-summary-grid">
                <div class="exec-summary-text">
                    <p>This report presents the hallucination evaluation results for the <strong>AI Assistant</strong>,
                    assessing model performance across faithfulness and custom hallucination metrics using a dataset of
                    <strong>{stats['total_support']:,} samples</strong>.</p>

                    <p><strong>Overall Assessment: {status}</strong> - {summary}</p>

                    <p>The evaluation methodology measures how well the model's responses are grounded in the provided
                    context, identifying supported and unsupported facts in each response.</p>

                    {issues_html}
                </div>

                <div class="key-findings">
                    <h4>Key Findings</h4>
                    {findings_html}
                </div>
            </div>
        </section>"""

    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        return """
        <section>
            <div class="section-number">Section 2</div>
            <h2 class="section-title">Methodology</h2>

            <p>The evaluation process follows a standardized methodology designed to ensure comprehensive
            and reproducible assessment of model performance. This section outlines the key components
            of the evaluation framework.</p>

            <h3 class="subsection-title">2.1 Evaluation Framework</h3>
            <p>The evaluation employs a multi-metric assessment approach, measuring model performance
            across several dimensions to provide a holistic view of model behavior and reliability.</p>

            <ol class="methodology-list">
                <li><strong>Data Preparation:</strong> Input data is preprocessed and validated to ensure
                consistency with model training specifications.</li>
                <li><strong>Metric Computation:</strong> Standard classification metrics are computed
                including accuracy, precision, recall, F1-score, and ROC-AUC where applicable.</li>
                <li><strong>Threshold Comparison:</strong> Results are compared against predefined
                acceptance thresholds to determine compliance status.</li>
                <li><strong>Error Analysis:</strong> Misclassifications are analyzed to identify
                patterns and potential areas for improvement.</li>
                <li><strong>Sample Review:</strong> Representative examples of correct and incorrect
                predictions are extracted for qualitative review.</li>
            </ol>

            <h3 class="subsection-title">2.2 Metric Definitions</h3>
            <p>The following metrics are used to assess model performance:</p>
            <ul style="margin: 16px 0; padding-left: 24px;">
                <li><strong>Accuracy:</strong> Proportion of correct predictions among total predictions</li>
                <li><strong>Precision:</strong> Proportion of true positives among positive predictions</li>
                <li><strong>Recall:</strong> Proportion of true positives among actual positives</li>
                <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
                <li><strong>ROC-AUC:</strong> Area under the receiver operating characteristic curve</li>
            </ul>
        </section>"""

    def _generate_dataset_section(self) -> str:
        """Generate dataset information section."""
        additional_cells = ""
        for key, value in self.gen.dataset_info.additional_info.items():
            additional_cells += f"""
                <div class="info-cell">
                    <label>{html.escape(key.replace('_', ' '))}</label>
                    <value>{html.escape(str(value))}</value>
                </div>"""

        return f"""
        <section>
            <div class="section-number">Section 2</div>
            <h2 class="section-title">Dataset Information</h2>

            <p>This section provides details about the evaluation dataset structure, including the input columns
            and key characteristics relevant to the hallucination evaluation process.</p>

            <div class="info-grid">
                <div class="info-cell">
                    <label>Dataset Name</label>
                    <value>{html.escape(self.gen.dataset_info.name)}</value>
                </div>
                <div class="info-cell">
                    <label>Version</label>
                    <value>{html.escape(self.gen.dataset_info.version)}</value>
                </div>
                <div class="info-cell">
                    <label>Total Samples</label>
                    <value>{self.gen.dataset_info.size:,}</value>
                </div>
                <div class="info-cell">
                    <label>Metrics Evaluated</label>
                    <value>{len(self.gen.metrics)}</value>
                </div>
                {additional_cells}
            </div>
        </section>"""

    def _generate_devtest_split_section(self) -> str:
        """Generate Dev/Test Split Analysis section with per-metric tables."""
        # Check if we have split data in results_df
        if not hasattr(self.gen, 'results_df') or self.gen.results_df is None:
            return ""

        if 'split' not in self.gen.results_df.columns:
            return ""

        results_df = self.gen.results_df

        # Get unique metrics
        metrics = results_df['metric'].unique()

        # Build a table for each metric
        metric_tables_html = ""
        for metric in metrics:
            metric_data = results_df[results_df['metric'] == metric]
            dev_row = metric_data[metric_data['split'] == 'dev'].iloc[0] if len(metric_data[metric_data['split'] == 'dev']) > 0 else None
            test_row = metric_data[metric_data['split'] == 'test'].iloc[0] if len(metric_data[metric_data['split'] == 'test']) > 0 else None

            display_name = dev_row['display_name'] if dev_row is not None else (test_row['display_name'] if test_row is not None else metric.replace('_', ' ').title())

            # Build rows for DEV and TEST
            rows_html = ""
            for split_name, row in [('DEV', dev_row), ('TEST', test_row)]:
                if row is not None:
                    support = int(row['total_count'])
                    pass_count = int(row['correct_count'])
                    fail_count = int(row['incorrect_count'])
                    pass_rate = row['accuracy']
                    mean_score = f"{row['mean']:.3f}" if row['mean'] is not None else "N/A"

                    status_class = "pass" if pass_rate >= self.gen.thresholds['accuracy'] else "fail"
                    status_text = "PASS" if pass_rate >= self.gen.thresholds['accuracy'] else "FAIL"

                    rows_html += f"""
                    <tr>
                        <td><strong>{split_name}</strong></td>
                        <td>{support:,}</td>
                        <td>{pass_count:,}</td>
                        <td>{fail_count:,}</td>
                        <td>{pass_rate:.1%}</td>
                        <td>{pass_count}/{support}</td>
                        <td>{mean_score}</td>
                        <td><span class="status-badge {status_class}">{status_text}</span></td>
                    </tr>"""

            metric_tables_html += f"""
            <div class="metric-comparison-block">
                <h4 class="metric-comparison-title">{html.escape(display_name)}</h4>
                <table class="data-table compact-table">
                    <thead>
                        <tr>
                            <th>Split</th>
                            <th>Support</th>
                            <th>Pass</th>
                            <th>Fail</th>
                            <th>Pass Rate</th>
                            <th>Pass Ratio</th>
                            <th>Mean Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>"""

        # Calculate overall summaries
        dev_df = results_df[results_df['split'] == 'dev']
        test_df = results_df[results_df['split'] == 'test']

        dev_total = int(dev_df['total_count'].sum()) if len(dev_df) > 0 else 0
        dev_correct = int(dev_df['correct_count'].sum()) if len(dev_df) > 0 else 0
        dev_overall = dev_correct / dev_total if dev_total > 0 else 0

        test_total = int(test_df['total_count'].sum()) if len(test_df) > 0 else 0
        test_correct = int(test_df['correct_count'].sum()) if len(test_df) > 0 else 0
        test_overall = test_correct / test_total if test_total > 0 else 0

        return f"""
        <section class="page-break">
            <div class="section-number">Section 3</div>
            <h2 class="section-title">Dev/Test Split Comparison</h2>

            <p>This section presents model performance across development and test splits for each metric.</p>

            <div class="split-summary-inline">
                <span><strong>DEV:</strong> {dev_total:,} samples, {dev_overall:.1%} overall pass rate</span>
                <span class="summary-divider">|</span>
                <span><strong>TEST:</strong> {test_total:,} samples, {test_overall:.1%} overall pass rate</span>
            </div>

            {metric_tables_html}
        </section>"""

    def _generate_results_section(self, stats: Dict) -> str:
        """Generate results section with overview."""
        if not stats:
            return ""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 4</div>
            <h2 class="section-title">Evaluation Results</h2>

            <p>This section presents the aggregated evaluation results across all metrics.
            Detailed breakdowns by individual metric are provided in Section 5.</p>

            <h3 class="subsection-title">4.1 Performance Summary</h3>

            <div class="info-grid">
                <div class="info-cell">
                    <label>Total Samples Evaluated</label>
                    <value>{stats['total_support']:,}</value>
                </div>
                <div class="info-cell">
                    <label>Total Errors</label>
                    <value class="text-danger">{stats['total_errors']:,}</value>
                </div>
                <div class="info-cell">
                    <label>Overall Error Rate</label>
                    <value class="text-danger">{stats['total_errors']/stats['total_support']:.2%}</value>
                </div>
                <div class="info-cell">
                    <label>Minimum Accuracy</label>
                    <value>{stats['min_accuracy']:.1%}</value>
                </div>
                <div class="info-cell">
                    <label>Maximum Accuracy</label>
                    <value>{stats['max_accuracy']:.1%}</value>
                </div>
                <div class="info-cell">
                    <label>Metrics Count</label>
                    <value>{stats['metrics_count']}</value>
                </div>
            </div>

            <div class="threshold-reference">
                <h4>Acceptance Thresholds</h4>
                <div class="threshold-grid">
                    <div class="threshold-item">
                        <div class="value">{self.gen.thresholds['accuracy']:.0%}</div>
                        <div class="label">Accuracy</div>
                    </div>
                    <div class="threshold-item">
                        <div class="value">{self.gen.thresholds['precision']:.0%}</div>
                        <div class="label">Precision</div>
                    </div>
                    <div class="threshold-item">
                        <div class="value">{self.gen.thresholds['recall']:.0%}</div>
                        <div class="label">Recall</div>
                    </div>
                    <div class="threshold-item">
                        <div class="value">{self.gen.thresholds['f1']:.0%}</div>
                        <div class="label">F1 Score</div>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_detailed_metrics_section(self) -> str:
        """Generate detailed metrics breakdown."""
        if not self.gen.metric_results:
            return ""

        def fmt_stat(val: Optional[float]) -> str:
            return f"{val:.3f}" if val is not None else "N/A"

        def fmt_pct(val: Optional[float]) -> str:
            return f"{val:.1%}" if val is not None else "N/A"

        rows_html = ""
        for metric_name, result in self.gen.metric_results.items():
            status_class = "pass" if result.accuracy >= self.gen.thresholds['accuracy'] else ("warning" if result.accuracy >= self.gen.thresholds['accuracy'] * 0.9 else "fail")
            status_text = "PASS" if result.accuracy >= self.gen.thresholds['accuracy'] else ("REVIEW" if result.accuracy >= self.gen.thresholds['accuracy'] * 0.9 else "FAIL")

            rows_html += f"""
                <tr>
                    <td><strong>{html.escape(result.display_name)}</strong></td>
                    <td>{result.support:,}</td>
                    <td class="text-success">{result.correct_count:,}</td>
                    <td class="text-danger">{result.incorrect_count:,}</td>
                    <td>{fmt_pct(result.accuracy)}</td>
                    <td>{result.correct_count}/{result.support}</td>
                    <td>{fmt_stat(result.score_mean)}</td>
                    <td>{fmt_stat(result.score_median)}</td>
                    <td>{fmt_stat(result.score_q25)}</td>
                    <td>{fmt_stat(result.score_q75)}</td>
                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                </tr>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 4</div>
            <h2 class="section-title">Detailed Metric Analysis</h2>

            <p>This section provides a comprehensive breakdown of performance metrics for each
            evaluation dimension. Results include support counts and capture ratios.</p>

            <div class="table-scroll-container">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Support</th>
                        <th>Pass</th>
                        <th>Fail</th>
                        <th>Pass Rate</th>
                        <th>Pass Ratio</th>
                        <th>Mean Score</th>
                        <th>Median</th>
                        <th>Q25</th>
                        <th>Q75</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            </div>
        </section>"""

    def _generate_sample_analysis_section(self) -> str:
        """Generate sample analysis section with examples in table format."""
        if not self.gen.examples:
            return ""

        # Generate tabs
        tabs_html = ""
        for i, metric_name in enumerate(self.gen.examples.keys()):
            active_class = "active" if i == 0 else ""
            display_name = metric_name.replace('_', ' ').title()
            tabs_html += f'<button class="sample-tab {active_class}" onclick="showSamples(\'{metric_name}\')">{html.escape(display_name)}</button>'

        # Generate content for each metric
        contents_html = ""
        for i, (metric_name, examples) in enumerate(self.gen.examples.items()):
            active_class = "active" if i == 0 else ""
            samples_html = self._generate_sample_table(examples, metric_name)

            # Count by example type (for new format) or by is_correct (for legacy)
            false_neg_count = sum(1 for e in examples if e.example_type == 'false_negative')
            false_pos_count = sum(1 for e in examples if e.example_type == 'false_positive')

            # Count by split
            dev_count = sum(1 for e in examples if e.metadata.get('dev_test') == 'dev')
            test_count = sum(1 for e in examples if e.metadata.get('dev_test') == 'test')
            has_splits = dev_count > 0 or test_count > 0

            # Split filter buttons
            split_buttons = ""
            if has_splits:
                split_buttons = f"""
                    <span class="filter-divider">|</span>
                    <button class="filter-btn split-filter" onclick="filterSampleRowsBySplit('{metric_name}', 'dev', this)">
                        <span class="split-badge dev">DEV</span> ({dev_count})
                    </button>
                    <button class="filter-btn split-filter" onclick="filterSampleRowsBySplit('{metric_name}', 'test', this)">
                        <span class="split-badge test">TEST</span> ({test_count})
                    </button>"""

            # If no example_type set (legacy format), fall back to is_correct
            if false_neg_count == 0 and false_pos_count == 0:
                correct_count = sum(1 for e in examples if e.is_correct)
                incorrect_count = len(examples) - correct_count
                filter_buttons = f"""
                    <button class="filter-btn active" onclick="filterSampleRows('{metric_name}', 'all', this)">
                        All Samples ({len(examples)})
                    </button>
                    <button class="filter-btn" onclick="filterSampleRows('{metric_name}', 'pass', this)">
                        Passed ({correct_count})
                    </button>
                    <button class="filter-btn" onclick="filterSampleRows('{metric_name}', 'fail', this)">
                        Failed ({incorrect_count})
                    </button>
                    {split_buttons}"""
            else:
                filter_buttons = f"""
                    <button class="filter-btn active" onclick="filterSampleRows('{metric_name}', 'all', this)">
                        All Samples ({len(examples)})
                    </button>
                    <button class="filter-btn" onclick="filterSampleRows('{metric_name}', 'false_negative', this)">
                        False Negatives ({false_neg_count})
                    </button>
                    <button class="filter-btn" onclick="filterSampleRows('{metric_name}', 'false_positive', this)">
                        False Positives ({false_pos_count})
                    </button>
                    {split_buttons}"""

            contents_html += f"""
            <div id="samples-{metric_name}" class="sample-content {active_class}">
                <div class="filter-controls">
                    {filter_buttons}
                </div>
                <div class="sample-table-container">
                    {samples_html}
                </div>
            </div>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 4</div>
            <h2 class="section-title">Sample Analysis</h2>

            <p>This section provides representative examples from the evaluation, showing the faithfulness analysis
            with supported and unsupported facts from the context.</p>

            <div class="sample-tabs">
                {tabs_html}
            </div>
            {contents_html}
        </section>"""

    def _generate_sample_table(self, examples: List, metric_name: str) -> str:
        """Generate HTML table for samples with Dev/Test badges."""
        rows_html = ""

        # Detect if we have example_type data (new format)
        has_example_type = any(e.example_type for e in examples)
        # Detect if we have split data
        has_split = any(e.metadata.get('dev_test') for e in examples)

        for example in examples:
            # Determine status class and text based on format
            if example.example_type:
                # New format: use example_type
                status_class = example.example_type  # 'false_negative' or 'false_positive'
                status_text = "FN" if example.example_type == 'false_negative' else "FP"
                status_badge_class = "fail"  # Both are error cases
            else:
                # Legacy format: use is_correct
                status_class = "pass" if example.is_correct else "fail"
                status_text = "PASS" if example.is_correct else "FAIL"
                status_badge_class = "pass" if example.is_correct else "fail"

            # Get split info for badge
            split = example.metadata.get('dev_test', '')
            split_badge = f'<span class="split-badge {split}">{split.upper()}</span>' if split else ''
            data_split = f'data-split="{split}"' if split else ''

            # Truncate long text for display
            input_display = example.input_text[:150] + "..." if len(example.input_text) > 150 else example.input_text
            output_display = example.output_text[:200] + "..." if len(example.output_text) > 200 else example.output_text
            reason_display = example.reason[:150] + "..." if len(example.reason) > 150 else example.reason

            rows_html += f"""
                <tr class="sample-row" data-status="{status_class}" {data_split}>
                    <td class="cell-split">{split_badge}</td>
                    <td class="cell-input">
                        <div class="cell-content">{html.escape(input_display)}</div>
                    </td>
                    <td class="cell-output">
                        <div class="cell-content">{html.escape(output_display)}</div>
                    </td>
                    <td class="cell-score">{example.score:.4f}</td>
                    <td class="cell-status">
                        <span class="status-badge {status_badge_class}">{status_text}</span>
                    </td>
                    <td class="cell-reason">{html.escape(reason_display)}</td>
                </tr>"""

        type_header = "Type" if has_example_type else "Status"
        split_header = '<th style="width: 8%;">Split</th>' if has_split else ''
        input_width = "22%" if has_split else "25%"
        output_width = "27%" if has_split else "30%"

        return f"""
            <table class="sample-data-table" id="sample-table-{metric_name}">
                <thead>
                    <tr>
                        {split_header}
                        <th style="width: {input_width};">Customer Utterance</th>
                        <th style="width: {output_width};">Response</th>
                        <th style="width: 8%;">Score</th>
                        <th style="width: 8%;">{type_header}</th>
                        <th style="width: 27%;">Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>"""

    def _generate_appendix(self) -> str:
        """Generate appendix section with column definitions."""
        return """
        <section class="appendix-section page-break">
            <div class="section-number">Appendix A</div>
            <h2 class="section-title">Column Definitions</h2>

            <h3 class="subsection-title">A.1 Input Column Definitions</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 200px;">Column Name</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><strong>Dev_Test</strong></td><td>Split indicator - "dev" or "test"</td></tr>
                    <tr><td><strong>Conv_id</strong></td><td>Unique conversation identifier</td></tr>
                    <tr><td><strong>Turn</strong></td><td>Turn number within the conversation</td></tr>
                    <tr><td><strong>Original_customer_utterance</strong></td><td>The original customer input/question</td></tr>
                    <tr><td><strong>Model_Response</strong></td><td>The model's generated response</td></tr>
                    <tr><td><strong>chat_history</strong></td><td>Previous conversation context</td></tr>
                    <tr><td><strong>neo4j_data</strong></td><td>Knowledge graph data used as context</td></tr>
                    <tr><td><strong>Hallucination (PASS/BLOCK)</strong></td><td>Manual hallucination label</td></tr>
                    <tr><td><strong>Comments</strong></td><td>Reviewer comments and notes</td></tr>
                </tbody>
            </table>

            <h3 class="subsection-title">A.2 Output Column Definitions</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 200px;">Column Name</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><strong>total_success</strong></td><td>Overall evaluation pass (1) or fail (0)</td></tr>
                    <tr><td><strong>faithfulness_success</strong></td><td>Faithfulness check pass (1) or fail (0)</td></tr>
                    <tr><td><strong>faithfulness_score</strong></td><td>Capture rate - proportion of facts grounded in context</td></tr>
                    <tr><td><strong>faithfulness_reason</strong></td><td>List of supported and unsupported facts</td></tr>
                    <tr><td><strong>custom_hallucination_success</strong></td><td>Custom hallucination check pass (1) or fail (0)</td></tr>
                    <tr><td><strong>custom_hallucination_score</strong></td><td>Hallucination confidence score (0-1)</td></tr>
                    <tr><td><strong>custom_hallucination_reason</strong></td><td>Explanation for hallucination detection</td></tr>
                </tbody>
            </table>
        </section>"""

    def _generate_footer(self) -> str:
        """Generate document footer."""
        return f"""
        <div class="document-footer">
            <div class="footer-content">
                <div class="footer-meta">
                    <div>Run Id: {html.escape(self.gen.report_metadata.document_id)}</div>
                    <div>Generated: {self.gen.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
                </div>
            </div>
        </div>"""

    def _generate_scripts(self) -> str:
        """Generate JavaScript for interactivity."""
        return """
    <script>
        function showSamples(metricName) {
            // Update tabs
            document.querySelectorAll('.sample-tab').forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');

            // Update content
            document.querySelectorAll('.sample-content').forEach(content => content.classList.remove('active'));
            document.getElementById('samples-' + metricName).classList.add('active');
        }

        function filterSampleRows(metricName, filter, button) {
            // Update button states (only for type filter buttons, not split)
            const container = document.getElementById('samples-' + metricName);
            container.querySelectorAll('.filter-btn:not(.split-filter)').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Reset split filter buttons
            container.querySelectorAll('.filter-btn.split-filter').forEach(btn => btn.classList.remove('active'));

            // Filter table rows
            const table = document.getElementById('sample-table-' + metricName);
            if (table) {
                table.querySelectorAll('.sample-row').forEach(row => {
                    const status = row.dataset.status;
                    if (filter === 'all') {
                        row.classList.remove('hidden');
                    } else if (filter === 'pass') {
                        row.classList.toggle('hidden', status !== 'pass');
                    } else if (filter === 'fail') {
                        row.classList.toggle('hidden', status !== 'fail');
                    } else if (filter === 'false_negative') {
                        row.classList.toggle('hidden', status !== 'false_negative');
                    } else if (filter === 'false_positive') {
                        row.classList.toggle('hidden', status !== 'false_positive');
                    }
                });
            }
        }

        function filterSampleRowsBySplit(metricName, split, button) {
            // Update split button states
            const container = document.getElementById('samples-' + metricName);
            container.querySelectorAll('.filter-btn.split-filter').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Reset type filter buttons and set "All" as active
            container.querySelectorAll('.filter-btn:not(.split-filter)').forEach(btn => btn.classList.remove('active'));
            container.querySelector('.filter-btn:not(.split-filter)').classList.add('active');

            // Filter table rows by split
            const table = document.getElementById('sample-table-' + metricName);
            if (table) {
                table.querySelectorAll('.sample-row').forEach(row => {
                    const rowSplit = row.dataset.split;
                    row.classList.toggle('hidden', rowSplit !== split);
                });
            }
        }

        // Print functionality
        function printReport() {
            window.print();
        }
    </script>"""
