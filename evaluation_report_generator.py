"""
MLflow Evaluation Report Generator

Generates self-contained HTML reports for ML evaluation runs,
designed for regulatory compliance and professional documentation.
"""

import pandas as pd
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from report_templates import ReportHTMLGenerator


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
    correct_count: int = 0
    incorrect_count: int = 0
    null_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    threshold: float = 0.5
    score_mean: Optional[float] = None
    score_median: Optional[float] = None
    score_std: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_q25: Optional[float] = None
    score_q75: Optional[float] = None
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
    example_type: str = ""  # 'false_positive' or 'false_negative'
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColumnMapping:
    """Maps DataFrame columns to expected fields.

    Supports two formats:
    1. Long format (row per metric): Each row has a 'metric' column
    2. Wide format (columns per metric): Each metric has its own columns
       e.g., toxicity_score, toxicity_success, toxicity_reason
    """

    def __init__(self, mapping: Dict[str, Any]):
        self.mapping = mapping
        # Detect if this is wide format (has 'metrics' list)
        self.is_wide_format = 'metrics' in mapping and isinstance(mapping['metrics'], list)

        if self.is_wide_format:
            self.metrics = mapping['metrics']
            self.score_suffix = mapping.get('score_suffix', '_score')
            self.success_suffix = mapping.get('success_suffix', '_success')
            self.reason_suffix = mapping.get('reason_suffix', '_reason')
        else:
            self.metrics = None

    def get(self, field: str, default: str = None) -> str:
        return self.mapping.get(field, default or field)

    def has(self, field: str) -> bool:
        return field in self.mapping

    def get_metric_columns(self, metric: str) -> Dict[str, str]:
        """Get column names for a specific metric (wide format only)."""
        if not self.is_wide_format:
            raise ValueError("get_metric_columns only works with wide format")
        return {
            'score': f'{metric}{self.score_suffix}',
            'success': f'{metric}{self.success_suffix}',
            'reason': f'{metric}{self.reason_suffix}',
        }


class ExamplesColumnMapping:
    """Maps raw example columns to required report fields.

    Example mapping:
        {
            'input': 'user_input',           # raw column -> report field
            'output': 'output_text',
            'score': 'bias_score',
            'success': 'bias_success',
            'reason': 'bias_reason'
        }
    """

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def get(self, field: str, default: str = None) -> str:
        """Get the raw column name for a report field."""
        return self.mapping.get(field, default or field)

    def has(self, field: str) -> bool:
        return field in self.mapping


class EvaluationReportGenerator:
    """Generates professional HTML reports for ML evaluation runs.

    Supports two input modes:
    1. Legacy mode: Raw DataFrame with all samples
    2. New mode: Results DataFrame (aggregated stats) + examples list (false positives/negatives)
    """

    def __init__(
        self,
        results_df: pd.DataFrame = None,
        examples_list: List[Dict[str, Any]] = None,
        examples_column_mapping: Dict[str, str] = None,
        dataset_info: DatasetInfo = None,
        report_metadata: ReportMetadata = None,
        run_id: str = None,
        run_name: str = None,
        thresholds: Dict[str, float] = None,
        embed_css: bool = True,
        # Legacy parameters for backward compatibility
        df: pd.DataFrame = None,
        column_mapping: ColumnMapping = None,
        metrics: List[str] = None,
    ):
        """
        Initialize the report generator.

        New input format:
            results_df: DataFrame with columns:
                - metric: metric name
                - correct_count, incorrect_count, null_count, total_count
                - accuracy, precision, recall, f1_score
                - mean, median, std, min, max, q25, q75

            examples_list: List of dicts, each with:
                - 'metric': metric name
                - 'false_negatives': DataFrame or list of raw example dicts
                - 'false_positives': DataFrame or list of raw example dicts

            examples_column_mapping: Dict mapping raw column names to required fields:
                - 'input': column name for user input
                - 'output': column name for model output
                - 'score': column name for metric score
                - 'success': column name for success flag
                - 'reason': column name for reason/explanation

        Legacy input format (for backward compatibility):
            df: Raw DataFrame with all samples
            column_mapping: ColumnMapping object
        """
        self.report_metadata = report_metadata or ReportMetadata()
        self.run_id = run_id or self._generate_run_id()
        self.run_name = run_name or "Model Evaluation Assessment"
        self.generated_at = datetime.now()
        self.embed_css = embed_css
        self.thresholds = thresholds or {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.80,
            'f1': 0.80
        }

        # Determine input mode
        self.use_new_format = results_df is not None

        if self.use_new_format:
            # New input format
            self.results_df = results_df.copy()
            self.examples_list = examples_list or []
            self.examples_column_mapping = ExamplesColumnMapping(examples_column_mapping or {})
            self.dataset_info = dataset_info or DatasetInfo(name="Evaluation Dataset")

            # Extract metrics from results DataFrame
            if 'metric' in self.results_df.columns:
                self.metrics = self.results_df['metric'].tolist()
            else:
                self.metrics = ['overall']

            # Calculate total samples from results
            if 'total_count' in self.results_df.columns:
                self.dataset_info.size = int(self.results_df['total_count'].sum())

            # Legacy attributes set to None
            self.df = None
            self.column_mapping = None

        else:
            # Legacy input format
            if df is None:
                raise ValueError("Either results_df (new format) or df (legacy format) must be provided")

            self.df = df.copy()
            self.column_mapping = column_mapping
            self.dataset_info = dataset_info or DatasetInfo(name="Evaluation Dataset", size=len(df))
            self.results_df = None
            self.examples_list = None
            self.examples_column_mapping = None

            if metrics is None:
                if column_mapping and column_mapping.is_wide_format:
                    self.metrics = column_mapping.metrics
                else:
                    metric_col = column_mapping.get('metric') if column_mapping else 'metric'
                    if metric_col in df.columns:
                        self.metrics = df[metric_col].unique().tolist()
                    else:
                        self.metrics = ['overall']
            else:
                self.metrics = metrics

        # Generate document ID if not provided
        if not self.report_metadata.document_id:
            self.report_metadata.document_id = f"EVAL-{self.generated_at.strftime('%Y%m%d')}-{self.run_id[:8].upper()}"

        self._compute_results()

        # Initialize HTML generator
        self._html_generator = ReportHTMLGenerator(self)

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _compute_results(self):
        """Compute metric results and extract examples."""
        self.metric_results: Dict[str, MetricResult] = {}
        self.examples: Dict[str, List[Example]] = {}

        if self.use_new_format:
            self._compute_results_from_aggregated()
        elif self.column_mapping.is_wide_format:
            self._compute_results_wide_format()
        else:
            self._compute_results_long_format()

    def _compute_results_from_aggregated(self):
        """Compute results from pre-aggregated results DataFrame and examples list."""
        # Process results DataFrame
        for _, row in self.results_df.iterrows():
            metric_name = row.get('metric', 'overall')

            # Helper to safely get float values
            def safe_float(val, default=None):
                if pd.isna(val):
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # Helper to safely get int values
            def safe_int(val, default=0):
                if pd.isna(val):
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            total_count = safe_int(row.get('total_count', 0))
            correct_count = safe_int(row.get('correct_count', 0))
            incorrect_count = safe_int(row.get('incorrect_count', 0))
            null_count = safe_int(row.get('null_count', 0))

            self.metric_results[metric_name] = MetricResult(
                name=metric_name,
                display_name=metric_name.replace('_', ' ').title(),
                accuracy=safe_float(row.get('accuracy', 0), 0.0),
                precision=safe_float(row.get('precision', 0), 0.0),
                recall=safe_float(row.get('recall', 0), 0.0),
                f1=safe_float(row.get('f1_score', 0), 0.0),
                support=total_count,
                correct_count=correct_count,
                incorrect_count=incorrect_count,
                null_count=null_count,
                error_count=incorrect_count,
                error_rate=incorrect_count / total_count if total_count > 0 else 0,
                score_mean=safe_float(row.get('mean')),
                score_median=safe_float(row.get('median')),
                score_std=safe_float(row.get('std')),
                score_min=safe_float(row.get('min')),
                score_max=safe_float(row.get('max')),
                score_q25=safe_float(row.get('q25')),
                score_q75=safe_float(row.get('q75')),
            )

            # Initialize examples for this metric
            self.examples[metric_name] = []

        # Process examples list
        for example_dict in self.examples_list:
            metric_name = example_dict.get('metric', 'overall')

            if metric_name not in self.examples:
                self.examples[metric_name] = []

            # Process false negatives (can be DataFrame or list of dicts)
            false_negatives = example_dict.get('false_negatives', [])
            if isinstance(false_negatives, pd.DataFrame):
                for _, row in false_negatives.iterrows():
                    example = self._convert_raw_example(row.to_dict(), metric_name, 'false_negative')
                    self.examples[metric_name].append(example)
            else:
                for raw_example in false_negatives:
                    example = self._convert_raw_example(raw_example, metric_name, 'false_negative')
                    self.examples[metric_name].append(example)

            # Process false positives (can be DataFrame or list of dicts)
            false_positives = example_dict.get('false_positives', [])
            if isinstance(false_positives, pd.DataFrame):
                for _, row in false_positives.iterrows():
                    example = self._convert_raw_example(row.to_dict(), metric_name, 'false_positive')
                    self.examples[metric_name].append(example)
            else:
                for raw_example in false_positives:
                    example = self._convert_raw_example(raw_example, metric_name, 'false_positive')
                    self.examples[metric_name].append(example)

    def _convert_raw_example(self, raw: Dict[str, Any], metric_name: str, example_type: str) -> Example:
        """Convert a raw example dict to an Example object using the column mapping."""
        mapping = self.examples_column_mapping

        # Get mapped values with defaults
        input_col = mapping.get('input', 'input')
        output_col = mapping.get('output', 'output')
        id_col = mapping.get('id', 'id')

        # For score, success, and reason, try metric-specific columns first
        # e.g., bias_score, bias_success, bias_reason
        score_col = mapping.get('score', 'score')
        success_col = mapping.get('success', 'success')
        reason_col = mapping.get('reason', 'reason')

        # Try metric-specific column names
        metric_score_col = f'{metric_name}_score'
        metric_success_col = f'{metric_name}_success'
        metric_reason_col = f'{metric_name}_reason'

        # Extract values from raw example
        input_text = str(raw.get(input_col, ''))
        output_text = str(raw.get(output_col, ''))
        example_id = str(raw.get(id_col, ''))

        # Try metric-specific score first, then generic
        if metric_score_col in raw:
            score = float(raw.get(metric_score_col, 0)) if raw.get(metric_score_col) is not None else 0.0
        else:
            score = float(raw.get(score_col, 0)) if raw.get(score_col) is not None else 0.0

        # Try metric-specific success first, then generic
        if metric_success_col in raw:
            is_correct = bool(raw.get(metric_success_col, False))
        else:
            is_correct = bool(raw.get(success_col, False))

        # Try metric-specific reason first, then generic
        if metric_reason_col in raw:
            reason = str(raw.get(metric_reason_col, ''))
        else:
            reason = str(raw.get(reason_col, ''))

        # Generate default reason if not provided
        if not reason:
            if example_type == 'false_negative':
                reason = "Model failed to detect expected outcome"
            else:
                reason = "Model incorrectly flagged as positive"

        return Example(
            id=example_id,
            input_text=input_text,
            output_text=output_text,
            expected=None,
            predicted=None,
            score=score,
            metric_name=metric_name,
            is_correct=is_correct,
            example_type=example_type,
            reason=reason,
            metadata=raw  # Store full raw data as metadata
        )

    def _compute_results_wide_format(self):
        """Compute results for wide format (columns per metric)."""
        for metric in self.metrics:
            metric_cols = self.column_mapping.get_metric_columns(metric)
            score_col = metric_cols['score']
            success_col = metric_cols['success']

            if success_col not in self.df.columns:
                continue

            total = len(self.df)
            correct = self.df[success_col].sum()
            accuracy = correct / total if total > 0 else 0

            # For wide format, we use success as the primary metric
            # Precision/recall/f1 are computed based on success column
            precision = recall = f1 = accuracy

            # Calculate score statistics
            score_mean = score_median = score_min = score_max = score_q25 = score_q75 = None
            if score_col in self.df.columns:
                scores = self.df[score_col].dropna()
                if len(scores) > 0:
                    score_mean = float(scores.mean())
                    score_median = float(scores.median())
                    score_min = float(scores.min())
                    score_max = float(scores.max())
                    score_q25 = float(scores.quantile(0.25))
                    score_q75 = float(scores.quantile(0.75))

            error_count = total - correct
            error_rate = error_count / total if total > 0 else 0

            self.metric_results[metric] = MetricResult(
                name=metric,
                display_name=metric.replace('_', ' ').title(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                support=total,
                correct_count=int(correct),
                incorrect_count=int(error_count),
                null_count=0,
                error_count=int(error_count),
                error_rate=error_rate,
                score_mean=score_mean,
                score_median=score_median,
                score_min=score_min,
                score_max=score_max,
                score_q25=score_q25,
                score_q75=score_q75
            )

            self.examples[metric] = self._extract_examples_wide_format(metric)

    def _compute_results_long_format(self):
        """Compute results for long format (row per metric)."""
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

            # Calculate score statistics
            score_mean = score_median = score_min = score_max = score_q25 = score_q75 = None
            if score_col in metric_df.columns:
                scores = metric_df[score_col].dropna()
                if len(scores) > 0:
                    score_mean = float(scores.mean())
                    score_median = float(scores.median())
                    score_min = float(scores.min())
                    score_max = float(scores.max())
                    score_q25 = float(scores.quantile(0.25))
                    score_q75 = float(scores.quantile(0.75))

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
                correct_count=int(correct),
                incorrect_count=int(error_count),
                null_count=0,
                error_count=error_count,
                error_rate=error_rate,
                score_mean=score_mean,
                score_median=score_median,
                score_min=score_min,
                score_max=score_max,
                score_q25=score_q25,
                score_q75=score_q75
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

    def _extract_examples_wide_format(self, metric_name: str, max_examples: int = 20) -> List[Example]:
        """Extract examples for a metric in wide format."""
        examples = []

        id_col = self.column_mapping.get('id', 'id')
        input_col = self.column_mapping.get('input', 'input')
        output_col = self.column_mapping.get('output', 'output')

        metric_cols = self.column_mapping.get_metric_columns(metric_name)
        score_col = metric_cols['score']
        success_col = metric_cols['success']
        reason_col = metric_cols['reason']

        for idx, row in self.df.head(max_examples).iterrows():
            is_correct = bool(row.get(success_col, False)) if success_col in self.df.columns else False

            # Get reason from metric-specific reason column
            if reason_col in self.df.columns:
                reason = str(row.get(reason_col, ''))
            else:
                reason = "Pass" if is_correct else "Fail"

            score = float(row.get(score_col, 0)) if score_col in self.df.columns else 0.0

            example = Example(
                id=str(row.get(id_col, idx)) if id_col in self.df.columns else str(idx),
                input_text=str(row.get(input_col, '')) if input_col in self.df.columns else '',
                output_text=str(row.get(output_col, '')) if output_col in self.df.columns else '',
                expected=None,
                predicted=None,
                score=score,
                metric_name=metric_name,
                is_correct=is_correct,
                reason=reason
            )
            examples.append(example)

        examples.sort(key=lambda x: (x.is_correct, -x.score))
        return examples

    def generate_html(self) -> str:
        """Generate the complete HTML report."""
        return self._html_generator.generate_html()

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
