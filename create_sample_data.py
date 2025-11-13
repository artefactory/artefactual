#!/usr/bin/env python3
"""
Example: Create a sample dataset for demo_uncertainty.py

This script shows the expected format for input data and creates a small test dataset.
"""

import json

# Example dataset with question-answer pairs
# Mix of easy questions and hallucination-prone questions
sample_data = [
    # Easy, factual questions (low uncertainty expected)
    {
        "question": "What is the capital of France?",
        "question_id": "q001",
        "short_answer": "Paris",
        "answer_aliases": ["Paris", "paris", "PARIS"],
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "question_id": "q002",
        "short_answer": "William Shakespeare",
        "answer_aliases": ["Shakespeare", "William Shakespeare", "W. Shakespeare"],
    },
    
    # Hallucination-prone questions (high uncertainty expected)
    {
        "question": "What is the exact number of hairs on Albert Einstein's head when he discovered relativity?",
        "question_id": "q003_hallucination",
        "short_answer": "Unknown/Impossible to know",
        "answer_aliases": ["Unknown", "impossible to know", "cannot be determined", "no one knows"],
    },
    {
        "question": "In which year did the fictional character Sherlock Holmes first visit the real city of Paris according to the original stories?",
        "question_id": "q004_hallucination",
        "short_answer": "Never explicitly stated/unclear",
        "answer_aliases": ["unclear", "not stated", "unknown", "never mentioned"],
    },
    {
        "question": "What was the name of Napoleon's third cousin's pet dog?",
        "question_id": "q005_hallucination",
        "short_answer": "Unknown/No historical record",
        "answer_aliases": ["Unknown", "no record", "not documented", "impossible to know"],
    },
    {
        "question": "How many words did Shakespeare write on the exact date of March 15, 1599?",
        "question_id": "q006_hallucination",
        "short_answer": "Unknown/Impossible to determine",
        "answer_aliases": ["Unknown", "impossible", "no record", "cannot be known"],
    },
    
    # Trick questions with common misconceptions
    {
        "question": "How many moons does Mercury have?",
        "question_id": "q007_trick",
        "short_answer": "0",
        "answer_aliases": ["0", "zero", "none", "no moons"],
    },
    {
        "question": "What color is a mirror?",
        "question_id": "q008_trick",
        "short_answer": "Green (or silver/white)",
        "answer_aliases": ["green", "silver", "white", "reflective"],
    },
    
    # Ambiguous recent/temporal questions (model training cutoff issues)
    {
        "question": "Who is the current Prime Minister of France?",
        "question_id": "q009_temporal",
        "short_answer": "Michel Barnier (as of Nov 2024)",
        "answer_aliases": ["Michel Barnier", "Barnier"],
    },
    {
        "question": "What is the latest version of Python released?",
        "question_id": "q010_temporal",
        "short_answer": "Depends on current date",
        "answer_aliases": ["3.12", "3.13", "unknown"],
    },
]

# Save to file
output_file = "sample_qa_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

print(f"✓ Created sample dataset: {output_file}")
print(f"  Contains {len(sample_data)} question-answer pairs")
print()
print("Usage:")
print(f"  python demo_uncertainty.py \\")
print(f"    --model_checkpoint YOUR_MODEL \\")
print(f"    --data_path {output_file} \\")
print(f"    --n_queries {len(sample_data)}")
