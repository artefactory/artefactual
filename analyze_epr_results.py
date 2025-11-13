#!/usr/bin/env python3
"""
Analyze EPR results from demo_uncertainty.py output.

This script reads the JSON output and creates a summary showing EPR scores
in descending order for each question.

Usage:
    python analyze_epr_results.py <input_json_file> [--output OUTPUT_FILE]

Example:
    python analyze_epr_results.py generation_test/DEMO_Ministral-8B-Instruct-2410_1.0_10_with_EPR.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys


def analyze_epr_results(input_file: str, output_file: str = None):
    """
    Analyze EPR results and create a summary file.
    
    Args:
        input_file: Path to the JSON output from demo_uncertainty.py
        output_file: Path to save the summary (optional, auto-generated if None)
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Auto-generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_EPR_SUMMARY.txt"
    
    # Create the summary
    lines = []
    lines.append("=" * 100)
    lines.append("EPR ANALYSIS SUMMARY - Entropy Production Rate per Question")
    lines.append("=" * 100)
    lines.append("")
    
    # Add metadata
    metadata = data.get('metadata', {})
    lines.append(f"Model: {metadata.get('generator_model', 'N/A')}")
    lines.append(f"Date: {metadata.get('date', 'N/A')}")
    lines.append(f"Temperature: {metadata.get('temperature', 'N/A')}")
    lines.append(f"Iterations per query: {metadata.get('iterations', 'N/A')}")
    lines.append(f"Top-K logprobs: {metadata.get('number_logprobs', 'N/A')}")
    lines.append(f"Total queries: {metadata.get('n_queries', 'N/A')}")
    lines.append("")
    lines.append("=" * 100)
    lines.append("")
    
    # Process each question
    results = data.get('results', [])
    
    # Collect all questions with their EPR scores for overall ranking
    all_questions_epr = []
    
    for idx, result in enumerate(results, 1):
        query = result.get('query', 'N/A')
        query_id = result.get('query_id', 'N/A')
        expected_answer = result.get('expected_answer', 'N/A')
        epr_stats = result.get('epr_statistics', {})
        full_info = result.get('full_info', [])
        
        # Extract EPR scores and answers for this question
        epr_data = []
        for iteration_idx, info in enumerate(full_info, 1):
            epr_score = info.get('epr_score', 0.0)
            answer_text = info.get('answer_text', 'N/A')
            epr_data.append({
                'iteration': iteration_idx,
                'epr_score': epr_score,
                'answer': answer_text
            })
        
        # Sort by EPR score in descending order
        epr_data_sorted = sorted(epr_data, key=lambda x: x['epr_score'], reverse=True)
        
        # Store for overall ranking
        mean_epr = epr_stats.get('mean', 0.0)
        all_questions_epr.append({
            'idx': idx,
            'query': query,
            'mean_epr': mean_epr,
            'epr_data': epr_data_sorted
        })
        
        # Format this question's output
        lines.append(f"Question {idx}: {query}")
        lines.append(f"ID: {query_id}")
        lines.append(f"Expected Answer: {expected_answer}")
        lines.append("")
        lines.append(f"EPR Statistics:")
        lines.append(f"  Mean:   {epr_stats.get('mean', 0):.6f}")
        lines.append(f"  Std:    {epr_stats.get('std', 0):.6f}")
        lines.append(f"  Min:    {epr_stats.get('min', 0):.6f}")
        lines.append(f"  Max:    {epr_stats.get('max', 0):.6f}")
        lines.append(f"  Median: {epr_stats.get('median', 0):.6f}")
        lines.append("")
        lines.append("EPR Scores by Iteration (Descending Order):")
        lines.append("-" * 100)
        
        for rank, item in enumerate(epr_data_sorted, 1):
            iter_num = item['iteration']
            epr = item['epr_score']
            answer = item['answer'][:80]  # Truncate long answers
            lines.append(f"  Rank {rank} | Iteration {iter_num} | EPR: {epr:.6f} | Answer: {answer}")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
    
    # Add overall ranking by mean EPR
    lines.append("")
    lines.append("=" * 100)
    lines.append("OVERALL RANKING - Questions by Mean EPR (Highest Uncertainty First)")
    lines.append("=" * 100)
    lines.append("")
    
    # Sort questions by mean EPR descending
    all_questions_sorted = sorted(all_questions_epr, key=lambda x: x['mean_epr'], reverse=True)
    
    for rank, q_data in enumerate(all_questions_sorted, 1):
        idx = q_data['idx']
        query = q_data['query'][:70]  # Truncate long questions
        mean_epr = q_data['mean_epr']
        lines.append(f"{rank:2d}. Q{idx:2d} | Mean EPR: {mean_epr:.6f} | {query}")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("")
    
    # Interpretation guide
    lines.append("INTERPRETATION GUIDE:")
    lines.append("-" * 100)
    lines.append("• LOW EPR (< 0.3):    High confidence, likely correct answer")
    lines.append("• MEDIUM EPR (0.3-0.5): Moderate uncertainty")
    lines.append("• HIGH EPR (> 0.5):   High uncertainty, potential hallucination risk")
    lines.append("• VERY HIGH EPR (> 0.8): Very uncertain, likely hallucination or wrong answer")
    lines.append("")
    lines.append("=" * 100)
    
    # Write to file
    output_text = "\n".join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"✓ EPR analysis complete!")
    print(f"  Summary saved to: {output_file}")
    print(f"  Analyzed {len(results)} questions")
    print(f"  Total iterations: {sum(len(r.get('full_info', [])) for r in results)}")
    print()
    
    # Print quick summary to console
    print("Quick Summary - Top 5 Most Uncertain Questions:")
    print("-" * 80)
    for rank, q_data in enumerate(all_questions_sorted[:5], 1):
        idx = q_data['idx']
        query = q_data['query'][:60]
        mean_epr = q_data['mean_epr']
        print(f"{rank}. Q{idx} | EPR: {mean_epr:.4f} | {query}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze EPR results from demo_uncertainty.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSON output file from demo_uncertainty.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the summary file (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        logging.error(f"Input file not found: {args.input_file}")
        return 1
    
    analyze_epr_results(args.input_file, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
