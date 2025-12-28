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
    <title>Model Evaluation Report - {html.escape(self.gen.report_metadata.document_id)}</title>
    {self._generate_styles()}
</head>
<body>
    <div class="document">
        {self._generate_cover_page(compliance_status)}
        {self._generate_table_of_contents()}
        {self._generate_executive_summary(overall_stats, compliance_status, compliance_summary, compliance_issues)}
        {self._generate_methodology_section()}
        {self._generate_dataset_section()}
        {self._generate_results_section(overall_stats)}
        {self._generate_detailed_metrics_section()}
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
        .cover-page { background: #1e3a8a; color: white; padding: 40px; margin-bottom: 20px; }
        section { margin-bottom: 30px; padding: 20px; border: 1px solid #e5e7eb; }
        .data-table { width: 100%; border-collapse: collapse; }
        .data-table th, .data-table td { padding: 10px; border: 1px solid #e5e7eb; text-align: left; }
        .data-table th { background: #1e3a8a; color: white; }
    </style>"""

    def _generate_cover_page(self, compliance_status: str) -> str:
        """Generate the cover page."""
        status_class = 'pass' if compliance_status == 'PASS' else ('conditional' if compliance_status == 'CONDITIONAL' else 'review')
        status_icon = '&#10003;' if compliance_status == 'PASS' else ('!' if compliance_status == 'CONDITIONAL' else '&#10007;')

        return f"""
        <div class="cover-page">
            <div class="cover-header">
                <div class="cover-classification">{html.escape(self.gen.report_metadata.classification)}</div>
            </div>

            <div class="cover-title-section">
                <div class="cover-subtitle">AI/ML Model Evaluation Report</div>
                <h1 class="cover-title">{html.escape(self.gen.dataset_info.name)}</h1>
                <div class="cover-run-name">{html.escape(self.gen.run_name)}</div>
                <div class="cover-status {status_class}">
                    <span class="cover-status-icon">{status_icon}</span>
                    <span>Evaluation Status: {compliance_status}</span>
                </div>
            </div>

            <div class="cover-meta">
                <div class="cover-meta-item">
                    <label>Document ID</label>
                    <value>{html.escape(self.gen.report_metadata.document_id)}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Report Version</label>
                    <value>{html.escape(self.gen.report_metadata.report_version)}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Generated Date</label>
                    <value>{self.gen.generated_at.strftime('%B %d, %Y')}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Evaluation Run ID</label>
                    <value class="font-mono">{html.escape(self.gen.run_id)}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Prepared By</label>
                    <value>{html.escape(self.gen.report_metadata.prepared_by or 'Automated Pipeline')}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Department</label>
                    <value>{html.escape(self.gen.report_metadata.department)}</value>
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

        def get_status_class(value: float, threshold: float) -> str:
            if value >= threshold:
                return "success"
            elif value >= threshold * 0.9:
                return "warning"
            return "danger"

        issues_html = ""
        if issues:
            issues_items = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
            issues_html = f"""
            <div class="issues-list">
                <h5>Items Requiring Attention</h5>
                <ul>{issues_items}</ul>
            </div>"""

        findings_html = ""
        for metric_name, result in self.gen.metric_results.items():
            indicator_class = "success" if result.accuracy >= self.gen.thresholds['accuracy'] else "danger"
            findings_html += f"""
            <div class="finding-item">
                <span class="finding-indicator {indicator_class}"></span>
                <span><strong>{html.escape(result.display_name)}:</strong> {result.accuracy:.1%} accuracy with {result.support:,} samples evaluated</span>
            </div>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 1</div>
            <h2 class="section-title">Executive Summary</h2>

            <div class="exec-summary-grid">
                <div class="summary-text">
                    <p>This report presents the evaluation results for <strong>{html.escape(self.gen.dataset_info.name)}</strong>,
                    conducted as part of the ongoing model risk management and validation process. The evaluation
                    assessed model performance across {stats['metrics_count']} distinct metrics using a dataset of
                    {stats['total_support']:,} samples.</p>

                    <div class="summary-highlight">
                        <div class="summary-highlight-title">Overall Assessment: {status}</div>
                        <p>{summary}</p>
                    </div>

                    <p>The evaluation methodology follows established best practices for AI/ML model validation,
                    measuring key performance indicators including accuracy, precision, recall, and F1-score.
                    Results are compared against predefined thresholds to determine compliance status.</p>

                    {issues_html}
                </div>

                <div class="key-findings">
                    <h4>Key Findings by Metric</h4>
                    {findings_html}
                </div>
            </div>

            <div class="metrics-overview">
                <div class="metric-card {get_status_class(stats['avg_accuracy'], self.gen.thresholds['accuracy'])}">
                    <div class="metric-value">{stats['avg_accuracy']:.1%}</div>
                    <div class="metric-label">Average Accuracy</div>
                    <div class="metric-threshold">Threshold: {self.gen.thresholds['accuracy']:.0%}</div>
                </div>
                <div class="metric-card {get_status_class(stats['avg_precision'], self.gen.thresholds['precision'])}">
                    <div class="metric-value">{stats['avg_precision']:.1%}</div>
                    <div class="metric-label">Average Precision</div>
                    <div class="metric-threshold">Threshold: {self.gen.thresholds['precision']:.0%}</div>
                </div>
                <div class="metric-card {get_status_class(stats['avg_recall'], self.gen.thresholds['recall'])}">
                    <div class="metric-value">{stats['avg_recall']:.1%}</div>
                    <div class="metric-label">Average Recall</div>
                    <div class="metric-threshold">Threshold: {self.gen.thresholds['recall']:.0%}</div>
                </div>
                <div class="metric-card {get_status_class(stats['avg_f1'], self.gen.thresholds['f1'])}">
                    <div class="metric-value">{stats['avg_f1']:.1%}</div>
                    <div class="metric-label">Average F1 Score</div>
                    <div class="metric-threshold">Threshold: {self.gen.thresholds['f1']:.0%}</div>
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
            <div class="section-number">Section 3</div>
            <h2 class="section-title">Dataset Information</h2>

            <p>This section provides details about the evaluation dataset, including its source,
            composition, and key characteristics relevant to the evaluation process.</p>

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
                    <label>Source</label>
                    <value>{html.escape(self.gen.dataset_info.source or 'Not specified')}</value>
                </div>
                <div class="info-cell">
                    <label>Created Date</label>
                    <value>{html.escape(self.gen.dataset_info.created_at or 'Not specified')}</value>
                </div>
                <div class="info-cell">
                    <label>Metrics Evaluated</label>
                    <value>{len(self.gen.metrics)}</value>
                </div>
                {additional_cells}
            </div>

            <h3 class="subsection-title">3.1 Dataset Description</h3>
            <p>{html.escape(self.gen.dataset_info.description or 'No description provided.')}</p>
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
            status_class = "pass" if result.accuracy >= self.gen.thresholds['accuracy'] else "fail"
            status_text = "PASS" if result.accuracy >= self.gen.thresholds['accuracy'] else "FAIL"

            rows_html += f"""
                <tr>
                    <td><strong>{html.escape(result.display_name)}</strong></td>
                    <td>{result.correct_count:,}</td>
                    <td class="text-danger">{result.incorrect_count:,}</td>
                    <td>{result.null_count:,}</td>
                    <td>{result.support:,}</td>
                    <td>{fmt_pct(result.accuracy)}</td>
                    <td>{fmt_pct(result.precision)}</td>
                    <td>{fmt_pct(result.recall)}</td>
                    <td>{fmt_stat(result.f1)}</td>
                    <td>{fmt_stat(result.score_mean)}</td>
                    <td>{fmt_stat(result.score_median)}</td>
                    <td>{fmt_stat(result.score_q25)}</td>
                    <td>{fmt_stat(result.score_q75)}</td>
                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                </tr>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 5</div>
            <h2 class="section-title">Detailed Metric Analysis</h2>

            <p>This section provides a comprehensive breakdown of performance metrics for each
            evaluation dimension. Results are compared against established thresholds to determine
            compliance status.</p>

            <div class="table-scroll-container">
            <table class="data-table compact-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Correct</th>
                        <th>Incorrect</th>
                        <th>Null</th>
                        <th>Total</th>
                        <th>Acc</th>
                        <th>Prec</th>
                        <th>Rec</th>
                        <th>F1</th>
                        <th>Mean</th>
                        <th>Med</th>
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
                    </button>"""
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
                    </button>"""

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
            <div class="section-number">Section 6</div>
            <h2 class="section-title">Sample Analysis</h2>

            <p>This section provides representative examples from the evaluation, including
            false positives and false negatives. These examples support qualitative review and error pattern analysis.</p>

            <div class="sample-tabs">
                {tabs_html}
            </div>
            {contents_html}
        </section>"""

    def _generate_sample_table(self, examples: List, metric_name: str) -> str:
        """Generate HTML table for samples."""
        rows_html = ""

        # Detect if we have example_type data (new format)
        has_example_type = any(e.example_type for e in examples)

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

            # Truncate long text for display
            input_display = example.input_text[:150] + "..." if len(example.input_text) > 150 else example.input_text
            output_display = example.output_text[:200] + "..." if len(example.output_text) > 200 else example.output_text
            reason_display = example.reason[:150] + "..." if len(example.reason) > 150 else example.reason

            rows_html += f"""
                <tr class="sample-row" data-status="{status_class}">
                    <td class="cell-input">
                        <div class="cell-label">User Input</div>
                        <div class="cell-content">{html.escape(input_display)}</div>
                    </td>
                    <td class="cell-output">
                        <div class="cell-label">LLM Output</div>
                        <div class="cell-content">{html.escape(output_display)}</div>
                    </td>
                    <td class="cell-score">{example.score:.4f}</td>
                    <td class="cell-status">
                        <span class="status-badge {status_badge_class}">{status_text}</span>
                    </td>
                    <td class="cell-reason">{html.escape(reason_display)}</td>
                </tr>"""

        type_header = "Type" if has_example_type else "Status"

        return f"""
            <table class="sample-data-table" id="sample-table-{metric_name}">
                <thead>
                    <tr>
                        <th style="width: 25%;">User Input</th>
                        <th style="width: 30%;">LLM Output</th>
                        <th style="width: 10%;">Score</th>
                        <th style="width: 10%;">{type_header}</th>
                        <th style="width: 25%;">Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>"""

    def _generate_appendix(self) -> str:
        """Generate appendix section."""
        return """
        <section class="appendix-section page-break">
            <div class="section-number">Appendix A</div>
            <h2 class="section-title">Definitions & Methodology</h2>

            <h3 class="subsection-title">A.1 Metric Definitions</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 150px;">Metric</th>
                        <th>Definition</th>
                        <th style="width: 200px;">Formula</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Accuracy</strong></td>
                        <td>The proportion of correct predictions among all predictions made.</td>
                        <td class="font-mono">(TP + TN) / Total</td>
                    </tr>
                    <tr>
                        <td><strong>Precision</strong></td>
                        <td>The proportion of true positive predictions among all positive predictions.</td>
                        <td class="font-mono">TP / (TP + FP)</td>
                    </tr>
                    <tr>
                        <td><strong>Recall</strong></td>
                        <td>The proportion of actual positives correctly identified.</td>
                        <td class="font-mono">TP / (TP + FN)</td>
                    </tr>
                    <tr>
                        <td><strong>F1 Score</strong></td>
                        <td>The harmonic mean of precision and recall.</td>
                        <td class="font-mono">2 * (P * R) / (P + R)</td>
                    </tr>
                </tbody>
            </table>

            <h3 class="subsection-title">A.2 Abbreviations</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 100px;">Term</th>
                        <th>Definition</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><strong>TP</strong></td><td>True Positive - Correctly predicted positive class</td></tr>
                    <tr><td><strong>TN</strong></td><td>True Negative - Correctly predicted negative class</td></tr>
                    <tr><td><strong>FP</strong></td><td>False Positive - Incorrectly predicted positive class</td></tr>
                    <tr><td><strong>FN</strong></td><td>False Negative - Incorrectly predicted negative class</td></tr>
                </tbody>
            </table>
        </section>"""

    def _generate_footer(self) -> str:
        """Generate document footer."""
        return f"""
        <div class="document-footer">
            <div class="footer-content">
                <div class="footer-notice">
                    <strong>Confidentiality Notice:</strong><br>
                    {html.escape(self.gen.report_metadata.confidentiality_notice)}
                </div>
                <div class="footer-meta">
                    <div>Document ID: {html.escape(self.gen.report_metadata.document_id)}</div>
                    <div>Generated: {self.gen.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
                    <div>Page <span class="page-number"></span></div>
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
            // Update button states
            const container = document.getElementById('samples-' + metricName);
            container.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

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

        // Print functionality
        function printReport() {
            window.print();
        }
    </script>"""
