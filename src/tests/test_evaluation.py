# src/tests/test_evaluation.py
import pytest
from datasets import Dataset
import json
import os
from src import config
from unittest.mock import patch, MagicMock, AsyncMock

from src.evaluation import run_evaluation_suite


FIXTURE_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden_dataset.json"
)



def test_ragas_dataset_structural_schema():
    """
    Unit Test: Verifies that our internal evaluation dictionaries match the exact
    structural array formatting boundaries dictated by Hugging Face and RAGAS core.
    """
    mock_eval_payload = {
        "question": ["What is step 1?"],
        "answer": ["Step 1 is initialization."],
        "contexts": [
            ["Source document chunk string text layer ABC"]
        ],  # MUST be a list of lists of strings!
        "ground_truth": ["Step 1 is initialization."],
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
    file_path = FIXTURE_DATASET_PATH

    # 1. Assert file exists
    assert os.path.exists(file_path), f"Expected data file missing at: {file_path}"

    # 2. Assert valid JSON formatting parses cleanly
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Assert structural container bounds
    assert isinstance(data, list), (
        "Golden dataset must be a root-level JSON list array."
    )
    assert len(data) > 0, "Golden dataset cannot be empty."

    # 4. Assert key schema constraints exist on the first block
    first_item = data[0]
    assert "question" in first_item, "Dataset items must contain a 'question' key."
    assert "ground_truth" in first_item, (
        "Dataset items must contain a 'ground_truth' key."
    )


@pytest.mark.asyncio
async def test_live_evaluation_suite_execution_path():
    """
    Integration Test: Force pytest to execute the true optimizer pipeline
    using comprehensive mocks to verify imports, signatures, and printing loops.
    """
    # 1. Mock out the heavy vector store and core evaluation engines
    with (
        patch("src.evaluation.optimizer.initialize_vectorstore"),
        patch("src.evaluation.optimizer.AgenticRAGCore") as mock_agent_class,
        patch("src.evaluation.optimizer.evaluate") as mock_ragas_evaluate,
        patch.object(config, "GOLDEN_DATASET_PATH", FIXTURE_DATASET_PATH),
    ):
        # Create a mock instance for AgenticRAGCore
        mock_agent_instance = MagicMock()

        # Explicitly configure the execution path method to be an AsyncMock!
        # This matches our pydantic contract schema format payload seamlessly
        mock_pipeline_response = MagicMock()
        mock_pipeline_response.answer = "Mocked answer string."
        mock_pipeline_response.citations = ["Mocked citation string."]

        mock_agent_instance.aexecute_pipeline = AsyncMock(
            return_value=mock_pipeline_response
        )
        mock_agent_instance.llm = MagicMock()  # Avoid attribute extraction crashes

        # Bind our mock instance to return when the class constructor is initialized
        mock_agent_class.return_value = mock_agent_instance

        # 2. Forge a dummy evaluation return object that mimics the modern Ragas API
        mock_result = MagicMock()
        mock_dataframe = MagicMock()
        mock_dataframe.mean.return_value.to_dict.return_value = {
            "faithfulness": 0.9500,
            "answer_relevance": 0.9200,
        }
        mock_result.to_pandas.return_value = mock_dataframe
        mock_ragas_evaluate.return_value = mock_result

        # 3. Trigger the true production function execution path
        try:
            scores = await run_evaluation_suite()
            assert scores is not None
        except Exception as e:
            pytest.fail(f"Live optimizer script crashed during execution loop: {e}")
