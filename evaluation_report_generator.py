"""
MLflow Evaluation Report Generator

Generates self-contained HTML reports for ML evaluation runs,
designed for regulatory compliance and professional documentation.
"""

import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import html
import hashlib


@dataclass
class DatasetInfo:
    """Metadata about the evaluated dataset."""
    name: str
    description: str = ""
    size: int = 0
    created_at: str = ""
    version: str = "1.0"
    source: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportMetadata:
    """Metadata for the regulatory report."""
    organization: str = "Organization Name"
    department: str = "AI/ML Risk & Compliance"
    classification: str = "INTERNAL USE ONLY"
    prepared_by: str = ""
    reviewed_by: str = ""
    approved_by: str = ""
    report_version: str = "1.0"
    document_id: str = ""
    confidentiality_notice: str = "This document contains confidential information intended for regulatory review purposes only."


@dataclass
class MetricResult:
    """Results for a single evaluation metric."""
    name: str
    display_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    roc_auc: Optional[float] = None
    error_count: int = 0
    error_rate: float = 0.0
    threshold: float = 0.5
    additional_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    """An example from the evaluation."""
    id: str
    input_text: str
    output_text: str
    expected: Any
    predicted: Any
    score: float
    metric_name: str
    is_correct: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColumnMapping:
    """Maps DataFrame columns to expected fields."""

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def get(self, field: str, default: str = None) -> str:
        return self.mapping.get(field, default or field)

    def has(self, field: str) -> bool:
        return field in self.mapping


class EvaluationReportGenerator:
    """Generates professional HTML reports for ML evaluation runs."""

    def __init__(
        self,
        df: pd.DataFrame,
        column_mapping: ColumnMapping,
        dataset_info: DatasetInfo,
        report_metadata: ReportMetadata = None,
        metrics: List[str] = None,
        run_id: str = None,
        run_name: str = None,
        thresholds: Dict[str, float] = None,
        embed_css: bool = True
    ):
        self.df = df.copy()
        self.column_mapping = column_mapping
        self.dataset_info = dataset_info
        self.report_metadata = report_metadata or ReportMetadata()
        self.run_id = run_id or self._generate_run_id()
        self.run_name = run_name or "Model Evaluation Assessment"
        self.generated_at = datetime.now()
        self.embed_css = embed_css  # If True, embed CSS in HTML; if False, link to external file
        self.thresholds = thresholds or {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.80,
            'f1': 0.80
        }

        # Generate document ID if not provided
        if not self.report_metadata.document_id:
            self.report_metadata.document_id = f"EVAL-{self.generated_at.strftime('%Y%m%d')}-{self.run_id[:8].upper()}"

        if metrics is None:
            metric_col = column_mapping.get('metric')
            if metric_col in df.columns:
                self.metrics = df[metric_col].unique().tolist()
            else:
                self.metrics = ['overall']
        else:
            self.metrics = metrics

        self._compute_results()

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _compute_results(self):
        """Compute metric results and extract examples."""
        self.metric_results: Dict[str, MetricResult] = {}
        self.examples: Dict[str, List[Example]] = {}

        metric_col = self.column_mapping.get('metric')

        for metric in self.metrics:
            if metric_col in self.df.columns:
                metric_df = self.df[self.df[metric_col] == metric]
            else:
                metric_df = self.df

            if len(metric_df) == 0:
                continue

            is_correct_col = self.column_mapping.get('is_correct', 'is_correct')
            predicted_col = self.column_mapping.get('predicted', 'predicted')
            expected_col = self.column_mapping.get('expected', 'expected')
            score_col = self.column_mapping.get('score', 'score')

            if is_correct_col in metric_df.columns:
                correct = metric_df[is_correct_col].sum()
                total = len(metric_df)
                accuracy = correct / total if total > 0 else 0

                if expected_col in metric_df.columns and predicted_col in metric_df.columns:
                    y_true = metric_df[expected_col]
                    y_pred = metric_df[predicted_col]

                    tp = ((y_true == 1) & (y_pred == 1)).sum()
                    fp = ((y_true == 0) & (y_pred == 1)).sum()
                    fn = ((y_true == 1) & (y_pred == 0)).sum()

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = recall = f1 = accuracy
            else:
                accuracy = precision = recall = f1 = 0
                correct = 0
                total = len(metric_df)

            roc_auc = None
            if score_col in metric_df.columns and expected_col in metric_df.columns:
                try:
                    from sklearn.metrics import roc_auc_score
                    y_true = metric_df[expected_col]
                    y_score = metric_df[score_col]
                    if len(y_true.unique()) > 1:
                        roc_auc = roc_auc_score(y_true, y_score)
                except:
                    pass

            error_count = total - correct if is_correct_col in metric_df.columns else 0
            error_rate = error_count / total if total > 0 else 0

            self.metric_results[metric] = MetricResult(
                name=metric,
                display_name=metric.replace('_', ' ').title(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                support=total,
                roc_auc=roc_auc,
                error_count=error_count,
                error_rate=error_rate
            )

            self.examples[metric] = self._extract_examples(metric_df, metric)

    def _extract_examples(self, df: pd.DataFrame, metric_name: str, max_examples: int = 20) -> List[Example]:
        """Extract good and bad examples for a metric."""
        examples = []

        id_col = self.column_mapping.get('id', 'id')
        input_col = self.column_mapping.get('input', 'input')
        output_col = self.column_mapping.get('output', 'output')
        expected_col = self.column_mapping.get('expected', 'expected')
        predicted_col = self.column_mapping.get('predicted', 'predicted')
        score_col = self.column_mapping.get('score', 'score')
        is_correct_col = self.column_mapping.get('is_correct', 'is_correct')
        reason_col = self.column_mapping.get('reason', 'reason')

        for idx, row in df.head(max_examples).iterrows():
            is_correct = bool(row.get(is_correct_col, False)) if is_correct_col in df.columns else False

            # Get reason from data or generate default reason
            if reason_col in df.columns:
                reason = str(row.get(reason_col, ''))
            else:
                # Generate default reason based on pass/fail status
                if is_correct:
                    reason = "Model prediction matches expected outcome"
                else:
                    expected = row.get(expected_col, None) if expected_col in df.columns else None
                    predicted = row.get(predicted_col, None) if predicted_col in df.columns else None
                    reason = f"Expected {expected}, predicted {predicted}"

            example = Example(
                id=str(row.get(id_col, idx)) if id_col in df.columns else str(idx),
                input_text=str(row.get(input_col, '')) if input_col in df.columns else '',
                output_text=str(row.get(output_col, '')) if output_col in df.columns else '',
                expected=row.get(expected_col, None) if expected_col in df.columns else None,
                predicted=row.get(predicted_col, None) if predicted_col in df.columns else None,
                score=float(row.get(score_col, 0)) if score_col in df.columns else 0.0,
                metric_name=metric_name,
                is_correct=is_correct,
                reason=reason
            )
            examples.append(example)

        examples.sort(key=lambda x: (x.is_correct, -x.score))
        return examples

    def _compute_overall_stats(self) -> Dict[str, Any]:
        """Compute overall statistics across all metrics."""
        if not self.metric_results:
            return {}

        results = list(self.metric_results.values())

        return {
            'avg_accuracy': np.mean([r.accuracy for r in results]),
            'avg_precision': np.mean([r.precision for r in results]),
            'avg_recall': np.mean([r.recall for r in results]),
            'avg_f1': np.mean([r.f1 for r in results]),
            'total_support': sum(r.support for r in results),
            'total_errors': sum(r.error_count for r in results),
            'metrics_count': len(results),
            'min_accuracy': min(r.accuracy for r in results),
            'max_accuracy': max(r.accuracy for r in results),
        }

    def _get_compliance_status(self, stats: Dict) -> tuple:
        """Determine overall compliance status."""
        issues = []
        if stats['avg_accuracy'] < self.thresholds['accuracy']:
            issues.append(f"Average accuracy ({stats['avg_accuracy']:.1%}) below threshold ({self.thresholds['accuracy']:.1%})")
        if stats['avg_precision'] < self.thresholds['precision']:
            issues.append(f"Average precision ({stats['avg_precision']:.1%}) below threshold ({self.thresholds['precision']:.1%})")
        if stats['avg_recall'] < self.thresholds['recall']:
            issues.append(f"Average recall ({stats['avg_recall']:.1%}) below threshold ({self.thresholds['recall']:.1%})")

        if not issues:
            return ('PASS', 'All metrics meet or exceed defined thresholds', issues)
        elif len(issues) <= 2:
            return ('CONDITIONAL', 'Some metrics require attention', issues)
        else:
            return ('REQUIRES REVIEW', 'Multiple metrics below threshold', issues)

    def generate_html(self) -> str:
        """Generate the complete HTML report."""
        overall_stats = self._compute_overall_stats()
        compliance_status, compliance_summary, compliance_issues = self._get_compliance_status(overall_stats)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report - {html.escape(self.report_metadata.document_id)}</title>
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

    def _generate_styles(self) -> str:
        """Generate link to external CSS file or embed styles."""
        if self.embed_css:
            # Read and embed the CSS file
            css_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.css')
            try:
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                return f"<style>\n{css_content}\n</style>"
            except FileNotFoundError:
                # Fallback: return minimal inline styles if CSS file not found
                return self._get_fallback_styles()
        else:
            # Link to external CSS file
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
                <div class="cover-classification">{html.escape(self.report_metadata.classification)}</div>
            </div>

            <div class="cover-title-section">
                <div class="cover-subtitle">AI/ML Model Evaluation Report</div>
                <h1 class="cover-title">{html.escape(self.dataset_info.name)}</h1>
                <div class="cover-run-name">{html.escape(self.run_name)}</div>
                <div class="cover-status {status_class}">
                    <span class="cover-status-icon">{status_icon}</span>
                    <span>Evaluation Status: {compliance_status}</span>
                </div>
            </div>

            <div class="cover-meta">
                <div class="cover-meta-item">
                    <label>Document ID</label>
                    <value>{html.escape(self.report_metadata.document_id)}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Report Version</label>
                    <value>{html.escape(self.report_metadata.report_version)}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Generated Date</label>
                    <value>{self.generated_at.strftime('%B %d, %Y')}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Evaluation Run ID</label>
                    <value class="font-mono">{html.escape(self.run_id)}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Prepared By</label>
                    <value>{html.escape(self.report_metadata.prepared_by or 'Automated Pipeline')}</value>
                </div>
                <div class="cover-meta-item">
                    <label>Department</label>
                    <value>{html.escape(self.report_metadata.department)}</value>
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
        for metric_name, result in self.metric_results.items():
            indicator_class = "success" if result.accuracy >= self.thresholds['accuracy'] else "danger"
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
                    <p>This report presents the evaluation results for <strong>{html.escape(self.dataset_info.name)}</strong>,
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
                <div class="metric-card {get_status_class(stats['avg_accuracy'], self.thresholds['accuracy'])}">
                    <div class="metric-value">{stats['avg_accuracy']:.1%}</div>
                    <div class="metric-label">Average Accuracy</div>
                    <div class="metric-threshold">Threshold: {self.thresholds['accuracy']:.0%}</div>
                </div>
                <div class="metric-card {get_status_class(stats['avg_precision'], self.thresholds['precision'])}">
                    <div class="metric-value">{stats['avg_precision']:.1%}</div>
                    <div class="metric-label">Average Precision</div>
                    <div class="metric-threshold">Threshold: {self.thresholds['precision']:.0%}</div>
                </div>
                <div class="metric-card {get_status_class(stats['avg_recall'], self.thresholds['recall'])}">
                    <div class="metric-value">{stats['avg_recall']:.1%}</div>
                    <div class="metric-label">Average Recall</div>
                    <div class="metric-threshold">Threshold: {self.thresholds['recall']:.0%}</div>
                </div>
                <div class="metric-card {get_status_class(stats['avg_f1'], self.thresholds['f1'])}">
                    <div class="metric-value">{stats['avg_f1']:.1%}</div>
                    <div class="metric-label">Average F1 Score</div>
                    <div class="metric-threshold">Threshold: {self.thresholds['f1']:.0%}</div>
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
        for key, value in self.dataset_info.additional_info.items():
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
                    <value>{html.escape(self.dataset_info.name)}</value>
                </div>
                <div class="info-cell">
                    <label>Version</label>
                    <value>{html.escape(self.dataset_info.version)}</value>
                </div>
                <div class="info-cell">
                    <label>Total Samples</label>
                    <value>{self.dataset_info.size:,}</value>
                </div>
                <div class="info-cell">
                    <label>Source</label>
                    <value>{html.escape(self.dataset_info.source or 'Not specified')}</value>
                </div>
                <div class="info-cell">
                    <label>Created Date</label>
                    <value>{html.escape(self.dataset_info.created_at or 'Not specified')}</value>
                </div>
                <div class="info-cell">
                    <label>Metrics Evaluated</label>
                    <value>{len(self.metrics)}</value>
                </div>
                {additional_cells}
            </div>

            <h3 class="subsection-title">3.1 Dataset Description</h3>
            <p>{html.escape(self.dataset_info.description or 'No description provided.')}</p>
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
                        <div class="value">{self.thresholds['accuracy']:.0%}</div>
                        <div class="label">Accuracy</div>
                    </div>
                    <div class="threshold-item">
                        <div class="value">{self.thresholds['precision']:.0%}</div>
                        <div class="label">Precision</div>
                    </div>
                    <div class="threshold-item">
                        <div class="value">{self.thresholds['recall']:.0%}</div>
                        <div class="label">Recall</div>
                    </div>
                    <div class="threshold-item">
                        <div class="value">{self.thresholds['f1']:.0%}</div>
                        <div class="label">F1 Score</div>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_detailed_metrics_section(self) -> str:
        """Generate detailed metrics breakdown."""
        if not self.metric_results:
            return ""

        rows_html = ""
        for metric_name, result in self.metric_results.items():
            status_class = "pass" if result.accuracy >= self.thresholds['accuracy'] else "fail"
            status_text = "PASS" if result.accuracy >= self.thresholds['accuracy'] else "BELOW THRESHOLD"
            progress_class = "success" if result.accuracy >= self.thresholds['accuracy'] else ("warning" if result.accuracy >= self.thresholds['accuracy'] * 0.9 else "danger")
            roc_auc_str = f"{result.roc_auc:.4f}" if result.roc_auc is not None else "N/A"

            rows_html += f"""
                <tr>
                    <td><strong>{html.escape(result.display_name)}</strong></td>
                    <td>
                        <div class="progress-container">
                            <div class="progress-bar">
                                <div class="progress-fill {progress_class}" style="width: {result.accuracy*100}%"></div>
                            </div>
                            <span class="progress-value">{result.accuracy:.1%}</span>
                        </div>
                    </td>
                    <td>{result.precision:.4f}</td>
                    <td>{result.recall:.4f}</td>
                    <td>{result.f1:.4f}</td>
                    <td>{roc_auc_str}</td>
                    <td>{result.support:,}</td>
                    <td class="text-danger">{result.error_count:,}</td>
                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                </tr>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 5</div>
            <h2 class="section-title">Detailed Metric Analysis</h2>

            <p>This section provides a comprehensive breakdown of performance metrics for each
            evaluation dimension. Results are compared against established thresholds to determine
            compliance status.</p>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="width: 180px;">Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>ROC-AUC</th>
                        <th>Samples</th>
                        <th>Errors</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </section>"""

    def _generate_sample_analysis_section(self) -> str:
        """Generate sample analysis section with examples in table format."""
        if not self.examples:
            return ""

        # Generate tabs
        tabs_html = ""
        for i, metric_name in enumerate(self.examples.keys()):
            active_class = "active" if i == 0 else ""
            display_name = metric_name.replace('_', ' ').title()
            tabs_html += f'<button class="sample-tab {active_class}" onclick="showSamples(\'{metric_name}\')">{html.escape(display_name)}</button>'

        # Generate content for each metric
        contents_html = ""
        for i, (metric_name, examples) in enumerate(self.examples.items()):
            active_class = "active" if i == 0 else ""
            samples_html = self._generate_sample_table(examples, metric_name)

            correct_count = sum(1 for e in examples if e.is_correct)
            incorrect_count = len(examples) - correct_count

            contents_html += f"""
            <div id="samples-{metric_name}" class="sample-content {active_class}">
                <div class="filter-controls">
                    <button class="filter-btn active" onclick="filterSampleRows('{metric_name}', 'all', this)">
                        All Samples ({len(examples)})
                    </button>
                    <button class="filter-btn" onclick="filterSampleRows('{metric_name}', 'pass', this)">
                        Passed ({correct_count})
                    </button>
                    <button class="filter-btn" onclick="filterSampleRows('{metric_name}', 'fail', this)">
                        Failed ({incorrect_count})
                    </button>
                </div>
                <div class="sample-table-container">
                    {samples_html}
                </div>
            </div>"""

        return f"""
        <section class="page-break">
            <div class="section-number">Section 6</div>
            <h2 class="section-title">Sample Analysis</h2>

            <p>This section provides representative examples from the evaluation, including both
            passed and failed samples. These examples support qualitative review and error pattern analysis.</p>

            <div class="sample-tabs">
                {tabs_html}
            </div>
            {contents_html}
        </section>"""

    def _generate_sample_table(self, examples: List[Example], metric_name: str) -> str:
        """Generate HTML table for samples."""
        rows_html = ""
        for example in examples:
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

        return f"""
            <table class="sample-data-table" id="sample-table-{metric_name}">
                <thead>
                    <tr>
                        <th style="width: 25%;">User Input</th>
                        <th style="width: 30%;">LLM Output</th>
                        <th style="width: 10%;">Score</th>
                        <th style="width: 10%;">Status</th>
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
                    <tr>
                        <td><strong>ROC-AUC</strong></td>
                        <td>Area under the receiver operating characteristic curve.</td>
                        <td>Integral of TPR vs FPR</td>
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
                    <tr><td><strong>ROC</strong></td><td>Receiver Operating Characteristic</td></tr>
                    <tr><td><strong>AUC</strong></td><td>Area Under Curve</td></tr>
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
                    {html.escape(self.report_metadata.confidentiality_notice)}
                </div>
                <div class="footer-meta">
                    <div>Document ID: {html.escape(self.report_metadata.document_id)}</div>
                    <div>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
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
                    } else {
                        row.classList.toggle('hidden', status !== 'fail');
                    }
                });
            }
        }

        // Print functionality
        function printReport() {
            window.print();
        }
    </script>"""

    def save(self, filepath: str):
        """Save the report to an HTML file."""
        html_content = self.generate_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report saved to: {filepath}")
        return filepath


def generate_evaluation_report(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    dataset_name: str,
    output_path: str,
    dataset_description: str = "",
    run_id: str = None,
    run_name: str = None,
    organization: str = "Organization",
    classification: str = "INTERNAL USE ONLY",
    thresholds: Dict[str, float] = None,
    **dataset_kwargs
) -> str:
    """
    Generate an evaluation report from a DataFrame.

    Args:
        df: DataFrame with evaluation results
        column_mapping: Dict mapping standard fields to column names
        dataset_name: Name of the dataset
        output_path: Path to save the HTML report
        dataset_description: Description of the dataset
        run_id: Optional MLflow run ID
        run_name: Optional run name
        organization: Organization name for the report
        classification: Document classification level
        thresholds: Dict of metric thresholds (accuracy, precision, recall, f1)
        **dataset_kwargs: Additional dataset info fields

    Returns:
        Path to the generated report
    """
    dataset_info = DatasetInfo(
        name=dataset_name,
        description=dataset_description,
        size=len(df),
        **{k: v for k, v in dataset_kwargs.items() if k in DatasetInfo.__dataclass_fields__}
    )

    if 'additional_info' not in dataset_kwargs:
        dataset_info.additional_info = {
            k: v for k, v in dataset_kwargs.items()
            if k not in DatasetInfo.__dataclass_fields__
        }

    report_metadata = ReportMetadata(
        organization=organization,
        classification=classification
    )

    mapping = ColumnMapping(column_mapping)
    generator = EvaluationReportGenerator(
        df=df,
        column_mapping=mapping,
        dataset_info=dataset_info,
        report_metadata=report_metadata,
        run_id=run_id,
        run_name=run_name,
        thresholds=thresholds
    )

    return generator.save(output_path)
