"""
Generate a sample hallucination evaluation report using synthetic data.

This script demonstrates how to use the EvaluationReportGenerator
for hallucination evaluation with Dev/Test splits:
- results_df: Aggregated metrics DataFrame split by dev/test
- examples_list: List of dicts with false_positives and false_negatives
- examples_column_mapping: Mapping from raw column names to report fields
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

from evaluation_report_generator import (
    EvaluationReportGenerator,
    DatasetInfo,
    ReportMetadata,
    ExamplesColumnMapping,
)


def create_results_dataframe(seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic results DataFrame with aggregated metrics split by dev/test.

    Columns:
        - metric: metric name
        - split: 'dev' or 'test'
        - correct_count, incorrect_count, null_count, total_count
        - accuracy, precision, recall, f1_score
        - mean, median, std, min, max, q25, q75
    """
    np.random.seed(seed)

    metrics = ['faithfulness', 'custom_hallucination']
    splits = ['dev', 'test']

    data = []
    for metric in metrics:
        for split in splits:
            # Different sample sizes for dev/test
            if split == 'dev':
                total = np.random.randint(150, 200)
            else:
                total = np.random.randint(300, 400)

            # Different accuracy rates for different metrics
            accuracy_rates = {
                'faithfulness': 0.82 if split == 'dev' else 0.78,
                'custom_hallucination': 0.88 if split == 'dev' else 0.85,
            }
            base_accuracy = accuracy_rates.get(metric, 0.85)

            correct = int(total * base_accuracy * np.random.uniform(0.95, 1.05))
            correct = min(correct, total)
            null_count = np.random.randint(0, 10)
            incorrect = total - correct - null_count

            # Score statistics (capture rate for faithfulness)
            if metric == 'faithfulness':
                # Capture rate typically higher
                scores = np.random.beta(8, 2, total)
            else:
                scores = np.random.beta(7, 3, total)

            # Precision/Recall
            precision = base_accuracy * np.random.uniform(0.95, 1.05)
            recall = base_accuracy * np.random.uniform(0.90, 1.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            display_name = metric.replace('_', ' ').title()
            if metric == 'faithfulness':
                display_name = 'Faithfulness (Capture Rate)'

            data.append({
                'metric': metric,
                'display_name': display_name,
                'split': split,
                'correct_count': correct,
                'incorrect_count': incorrect,
                'null_count': null_count,
                'total_count': total,
                'accuracy': correct / total if total > 0 else 0,
                'precision': min(precision, 1.0),
                'recall': min(recall, 1.0),
                'f1_score': min(f1, 1.0),
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
            })

    return pd.DataFrame(data)


def generate_faithfulness_reason(is_hallucination: bool, seed_idx: int) -> str:
    """
    Generate a faithfulness_reason with supported/unsupported facts structure.
    """
    np.random.seed(seed_idx)

    supported_facts = [
        "The account balance is $5,432.10",
        "The transaction was processed on January 15, 2024",
        "The customer has been a member since 2019",
        "The interest rate is 4.5% APR",
        "The minimum payment is $25",
        "The credit limit is $10,000",
        "The payment due date is the 15th of each month",
        "The account was opened in New York branch",
        "The last statement was generated on December 31",
        "The customer's preferred contact method is email",
        "The account type is Premium Checking",
        "Auto-pay is currently enabled",
        "The routing number is verified",
        "The account is in good standing",
        "Two-factor authentication is active",
    ]

    unsupported_facts = [
        "The promotion ends next week",
        "You will receive a bonus of $100",
        "The fee will be waived automatically",
        "Your application was pre-approved",
        "The upgrade is available immediately",
        "The rate is guaranteed for 5 years",
        "All transactions are covered by insurance",
        "The refund will appear in 24 hours",
        "Your credit score will improve by 50 points",
        "The service includes free international transfers",
        "Premium support is included at no cost",
        "The account comes with travel insurance",
        "You qualify for the highest tier",
        "The interest compounds daily",
        "Unlimited cashback is available",
    ]

    num_supported = np.random.randint(2, 5)
    num_unsupported = np.random.randint(0, 3) if is_hallucination else 0

    selected_supported = random.sample(supported_facts, min(num_supported, len(supported_facts)))
    selected_unsupported = random.sample(unsupported_facts, min(num_unsupported, len(unsupported_facts))) if is_hallucination else []

    reason_parts = []
    if selected_supported:
        reason_parts.append("Supported facts from context: " + "; ".join(selected_supported))
    if selected_unsupported:
        reason_parts.append("Unsupported facts (hallucinated): " + "; ".join(selected_unsupported))
    else:
        reason_parts.append("No unsupported facts detected")

    return " | ".join(reason_parts)


def create_examples_list(seed: int = 42) -> list:
    """
    Create synthetic examples list with false positives and false negatives.
    Includes Dev/Test split information.

    Each dict contains:
        - 'metric': metric name
        - 'false_negatives': DataFrame with raw examples (hallucinations not caught)
        - 'false_positives': DataFrame with raw examples (good responses flagged)
    """
    np.random.seed(seed)
    random.seed(seed)

    metrics = ['faithfulness', 'custom_hallucination']
    splits = ['dev', 'test']

    # Sample customer utterances (banking context)
    sample_inputs = [
        "What is my current account balance?",
        "Can you explain the recent transaction fee?",
        "How do I set up automatic payments?",
        "What are my credit card rewards options?",
        "When is my next payment due?",
        "Can I increase my credit limit?",
        "What is the interest rate on my savings account?",
        "How do I dispute a charge on my statement?",
        "Can you help me understand my mortgage terms?",
        "What happens if I miss a payment?",
        "How do I transfer money to another bank?",
        "What are the fees for international transactions?",
        "Can I get a copy of my statement?",
        "How do I update my contact information?",
        "What security features protect my account?",
    ]

    # Sample good responses (grounded in context)
    sample_outputs_good = [
        "Based on your account records, your current balance is $5,432.10 as of today. This includes your recent deposit of $1,200 that was credited yesterday.",
        "I can see from your transaction history that a $2.50 maintenance fee was applied. This is a standard monthly fee for accounts with balances under $1,500.",
        "To set up automatic payments, you can use our online banking portal. Navigate to 'Bill Pay' and select 'Automatic Payments'. You'll need the payee's account number and routing number.",
        "Your credit card offers 1.5% cashback on all purchases. Additionally, you earn 3% on dining and 2% on groceries. Points expire 24 months after earning.",
        "According to your account, your next payment of $250 is due on February 15th. You can pay online, by phone, or by mail.",
        "Based on your account history and credit profile, you may be eligible for a credit limit increase. I can submit a request for review by our credit team.",
        "Your savings account currently earns 4.5% APY. This rate is variable and may change based on market conditions.",
        "To dispute a charge, please fill out the dispute form available in online banking or call our dispute resolution team. You have 60 days from the statement date.",
        "Your current mortgage has a 30-year fixed rate of 6.5%. Your principal balance is $285,000 with 22 years remaining.",
        "If you miss a payment, a late fee of $35 may apply after the 10-day grace period. This could also affect your credit score.",
        "You can transfer funds through ACH, which takes 1-3 business days, or wire transfer for same-day delivery. ACH is free; wire transfers cost $25.",
        "International transactions incur a 3% foreign transaction fee. This applies to purchases made in foreign currencies or with merchants outside the US.",
        "You can download statements from online banking going back 7 years. Paper statements can be mailed for a $5 fee per statement.",
        "You can update your contact information through online banking under 'Profile Settings' or by visiting any branch with valid ID.",
        "Your account is protected by 256-bit encryption, two-factor authentication, and real-time fraud monitoring. We also offer account alerts.",
    ]

    # Sample hallucinated responses (contain unsupported claims)
    sample_outputs_bad = [
        "Your account balance is $5,432.10, and I see you'll be receiving a promotional bonus of $500 next month for being a loyal customer.",
        "The transaction fee will be automatically refunded within 24 hours. We're waiving all fees for preferred customers like yourself this quarter.",
        "Automatic payments are set up, and I've also enrolled you in our premium rewards program which gives you double points on all purchases.",
        "You have unlimited cashback rewards! Plus, as a special offer, we're upgrading your card to platinum status with no annual fee forever.",
        "Your payment is due February 15th. I've also extended your grace period to 30 days as a courtesy, so no worries about late fees.",
        "Great news! Your credit limit has been automatically increased to $25,000 and your interest rate has been reduced to 0% for 18 months.",
        "Your savings rate is 4.5% APY, guaranteed to never decrease. Plus, you're earning bonus interest of 1% on top of that.",
        "I've already processed your dispute and credited $150 back to your account. All disputes are automatically approved for trusted customers.",
        "Your mortgage rate can be lowered to 4.5% immediately with no closing costs. I've pre-approved this refinance option for you.",
        "Don't worry about missed payments! As a valued customer, we've removed all late fees from your account permanently.",
        "Transfers are always instant and free to any bank worldwide. There are no limits on transfer amounts either.",
        "International fees have been waived for your account permanently. You'll never pay foreign transaction fees again.",
        "I'm sending you statements for the last 20 years at no charge, and they should arrive within 24 hours.",
        "Your information has been updated across all systems instantly, and I've also upgraded your account to premium status.",
        "Your account has military-grade security and is insured up to $10 million. Any unauthorized transactions are instantly reversed.",
    ]

    # Reasons for false positives and false negatives
    fn_reasons = {
        'faithfulness': [
            "Model failed to detect unsupported promotional claims",
            "Hallucinated bonus or reward not in context",
            "Fabricated account upgrade or status change",
            "Unsupported fee waiver claim not detected",
        ],
        'custom_hallucination': [
            "Missed hallucinated timeline or guarantee",
            "Failed to flag unsupported automatic approval",
            "Did not detect fabricated policy exception",
            "Overlooked invented customer benefits",
        ],
    }

    fp_reasons = {
        'faithfulness': [
            "Valid paraphrase incorrectly flagged as unfaithful",
            "Reasonable inference from context marked as hallucination",
            "Standard banking terminology misidentified",
            "Correct information flagged due to format difference",
        ],
        'custom_hallucination': [
            "Legitimate clarification marked as fabrication",
            "Accurate summary flagged incorrectly",
            "Standard disclaimer misidentified as hallucination",
            "Correct regulatory information flagged in error",
        ],
    }

    examples_list = []

    for metric in metrics:
        for split in splits:
            false_negatives_data = []
            false_positives_data = []

            # Generate false negatives (5-10 per metric per split)
            num_fn = np.random.randint(5, 11)
            for i in range(num_fn):
                idx = (i + (0 if split == 'dev' else 7)) % len(sample_inputs)
                conv_id = f"CONV_{split.upper()}_{i+1:04d}"
                turn = np.random.randint(1, 5)

                # False negatives: hallucinated output not caught
                false_negatives_data.append({
                    'id': f'{metric}_{split}_fn_{i+1}',
                    'dev_test': split,
                    'conv_id': conv_id,
                    'turn': turn,
                    'customer_utterance': sample_inputs[idx],
                    'model_response': sample_outputs_bad[idx],
                    'chat_history': f"Previous {turn-1} turns of conversation about account inquiry",
                    'neo4j_data': f"Customer profile data, account balance: $5,432.10, member since 2019",
                    f'{metric}_score': round(np.random.uniform(0.6, 0.85), 4),  # High score but actually bad
                    f'{metric}_success': False,  # Metric said OK, but actually hallucinated
                    f'{metric}_reason': generate_faithfulness_reason(True, i * 100 + idx) if metric == 'faithfulness' else random.choice(fn_reasons[metric]),
                })

            # Generate false positives (3-8 per metric per split)
            num_fp = np.random.randint(3, 9)
            for i in range(num_fp):
                idx = (i + (5 if split == 'dev' else 10)) % len(sample_inputs)
                conv_id = f"CONV_{split.upper()}_{i+100:04d}"
                turn = np.random.randint(1, 5)

                # False positives: good output incorrectly flagged
                false_positives_data.append({
                    'id': f'{metric}_{split}_fp_{i+1}',
                    'dev_test': split,
                    'conv_id': conv_id,
                    'turn': turn,
                    'customer_utterance': sample_inputs[idx],
                    'model_response': sample_outputs_good[idx],
                    'chat_history': f"Previous {turn-1} turns of conversation about account inquiry",
                    'neo4j_data': f"Customer profile data, verified account information",
                    f'{metric}_score': round(np.random.uniform(0.2, 0.45), 4),  # Low score but actually good
                    f'{metric}_success': False,  # Metric said bad, but actually OK
                    f'{metric}_reason': generate_faithfulness_reason(False, i * 200 + idx) if metric == 'faithfulness' else random.choice(fp_reasons[metric]),
                })

            # Convert to DataFrames
            false_negatives_df = pd.DataFrame(false_negatives_data)
            false_positives_df = pd.DataFrame(false_positives_data)

            examples_list.append({
                'metric': metric,
                'split': split,
                'false_negatives': false_negatives_df,
                'false_positives': false_positives_df,
            })

    return examples_list


def main():
    """Generate sample hallucination evaluation report with Dev/Test splits."""

    print("=" * 60)
    print("Hallucination Evaluation Report Generator")
    print("=" * 60)

    # Create synthetic data
    print("\n1. Creating synthetic data with Dev/Test splits...")

    results_df = create_results_dataframe()
    print(f"   Results DataFrame: {len(results_df)} rows (metrics x splits)")
    print(f"   Columns: {list(results_df.columns)}")

    examples_list = create_examples_list()
    total_examples = sum(
        len(e['false_negatives']) + len(e['false_positives'])
        for e in examples_list
    )
    print(f"   Examples: {total_examples} total across {len(examples_list)} metric/split combinations")

    # Column mapping for examples
    examples_column_mapping = {
        'id': 'id',
        'input': 'customer_utterance',
        'output': 'model_response',
        'split': 'dev_test',
    }

    # Create dataset info
    total_samples = int(results_df['total_count'].sum())
    dev_samples = int(results_df[results_df['split'] == 'dev']['total_count'].sum())
    test_samples = int(results_df[results_df['split'] == 'test']['total_count'].sum())

    dataset_info = DatasetInfo(
        name="Hallucination Evaluation Dataset",
        description="Evaluation dataset for testing model responses against knowledge base context. Measures faithfulness (capture rate) and custom hallucination detection across dev and test splits.",
        size=total_samples,
        version="1.0.0",
        source="Internal Evaluation Pipeline",
        created_at=datetime.now().strftime("%Y-%m-%d"),
        additional_info={
            'Model Under Test': 'AI Assistant v2.0',
            'Evaluation Framework': 'Hallucination Detection Pipeline',
            'Dev Samples': dev_samples,
            'Test Samples': test_samples,
            'Metrics': 'faithfulness, custom_hallucination',
        }
    )

    # Create report metadata
    report_metadata = ReportMetadata(
        organization="",
        department="AI/ML Model Risk Management",
        classification="INTERNAL USE ONLY",
        prepared_by="ML Evaluation Pipeline",
        reviewed_by="Model Risk Team",
        approved_by="",
        report_version="1.0",
        confidentiality_notice=""
    )

    # Define thresholds
    thresholds = {
        'accuracy': 0.80,
        'precision': 0.75,
        'recall': 0.75,
        'f1': 0.75
    }

    # Generate report
    print("\n2. Generating report...")

    generator = EvaluationReportGenerator(
        results_df=results_df,
        examples_list=examples_list,
        examples_column_mapping=examples_column_mapping,
        dataset_info=dataset_info,
        report_metadata=report_metadata,
        run_id="hallucination_eval_001",
        run_name="Hallucination Evaluation - Dev/Test Analysis",
        thresholds=thresholds,
    )

    output_path = "evaluation_report.html"
    generator.save(output_path)

    print(f"\n3. Report generated successfully!")
    print(f"   Output: {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Show dev/test breakdown
    print("\nDev/Test Split Analysis:")
    for split in ['dev', 'test']:
        split_df = results_df[results_df['split'] == split]
        print(f"\n  {split.upper()} Split:")
        for _, row in split_df.iterrows():
            print(f"    - {row['metric']}: {row['accuracy']:.1%} accuracy ({row['correct_count']}/{row['total_count']})")

    print(f"\nGenerated file: {output_path}")

    return results_df, examples_list, generator


if __name__ == "__main__":
    results_df, examples_list, generator = main()
