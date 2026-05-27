# src/tests/test_evaluation.py
import pytest
from datasets import Dataset
import json
import os
from src import config

def test_ragas_dataset_structural_schema():
    """
    Unit Test: Verifies that our internal evaluation dictionaries match the exact
    structural array formatting boundaries dictated by Hugging Face and RAGAS core.
    """
    mock_eval_payload = {
        "question": ["What is step 1?"],
        "answer": ["Step 1 is initialization."],
        "contexts": [["Source document chunk string text layer ABC"]],  # MUST be a list of lists of strings!
        "ground_truth": ["Step 1 is initialization."]
    }
    
    # Try compiling the dataset structure
    dataset = Dataset.from_dict(mock_eval_payload)
    
    # Assert structural integrity metrics
    assert "question" in dataset.features
    assert "contexts" in dataset.features
    assert isinstance(dataset["contexts"][0], list)


def test_golden_dataset_file_loading_integrity():
    """
    Unit Test: Verifies that golden_dataset.json exists on disk, contains valid JSON,
    and adheres strictly to the list-of-dictionaries schema structure.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, config.GOLDEN_DATASET_PATH)
    
    # 1. Assert file exists
    assert os.path.exists(file_path), f"Expected data file missing at: {file_path}"
    
    # 2. Assert valid JSON formatting parses cleanly
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 3. Assert structural container bounds
    assert isinstance(data, list), "Golden dataset must be a root-level JSON list array."
    assert len(data) > 0, "Golden dataset cannot be empty."
    
    # 4. Assert key schema constraints exist on the first block
    first_item = data[0]
    assert "question" in first_item, "Dataset items must contain a 'question' key."
    assert "ground_truth" in first_item, "Dataset items must contain a 'ground_truth' key."