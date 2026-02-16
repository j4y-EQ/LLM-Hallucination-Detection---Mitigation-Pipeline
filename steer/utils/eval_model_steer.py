"""Azure Evaluator Model for MCQ (Multiple Choice Question) Answering

SECURITY NOTE: This module requires Azure OpenAI credentials.
Create a .env file in the project root with the following variables:
  - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
  - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
  - AZURE_OPENAI_API_VERSION: API version (optional, default: 2024-12-01-preview)
  - EVAL_MODEL: Deployment name (optional, default: gpt-4o)

Example .env file (c:/Users/enqiy/dso-internship-all/.env):
  AZURE_OPENAI_API_KEY=your-key-here
  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
  AZURE_OPENAI_API_VERSION=2024-12-01-preview
  EVAL_MODEL=gpt-4o
"""

import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from tqdm.auto import tqdm
# Import configurations from config
from config import DEBUG_VERBOSE
# Import the consolidated logger
from logger import consolidated_logger as eval_logger

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Global variable to store output directory for eval logging
_eval_output_dir = None

def set_eval_output_directory(output_dir: str):
    """Set the output directory for eval logger to sync with baseline_run"""
    global _eval_output_dir
    _eval_output_dir = output_dir
    eval_logger.set_output_directory(output_dir)

# Azure Evaluator Model Setup - Load from environment variables
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o")

# Validate required environment variables
required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}\n"
        f"Please create .env file in project root with:\n"
        f"  AZURE_OPENAI_API_KEY=your-key-here\n"
        f"  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/"
    )

client = openai.AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

def test_evaluator_connectivity():
    eval_logger.info("\n" + "="*60)
    eval_logger.info("EVALUATOR CONNECTIVITY TEST")
    eval_logger.info("="*60)
    try:
        rsp = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "You are a ping responder."},
                {"role": "user", "content": "Reply exactly with OK"},
            ],
            max_tokens=2,
            temperature=0,
        )
        content = (rsp.choices[0].message.content or "").strip()
        eval_logger.info(f"Evaluator response: {content!r}")
        if not content:
            raise RuntimeError("Empty evaluator response")
        eval_logger.info("Evaluator connectivity: OK")
        eval_logger.info("="*60)
        return True
    except Exception as e:
        eval_logger.error(f"Evaluator connectivity FAILED: {e}")
        eval_logger.error("Please verify Azure credentials and EVAL_MODEL (deployment) name.")
        eval_logger.error("Aborting run before processing.")
        eval_logger.error("="*60)
        raise

def _extract_retry_after_seconds(error_message: str, default: int = 60) -> int:
    """
    Extract retry-after seconds from Azure rate limit error message.
    
    Azure returns messages like:
    "...Please retry after 35 seconds..."
    
    Args:
        error_message: The error message string from Azure
        default: Default retry-after seconds if parsing fails
        
    Returns:
        Seconds to wait before retrying
    """
    try:
        # Look for "retry after X seconds" pattern
        match = re.search(r'retry after (\d+) seconds', error_message, re.IGNORECASE)
        if match:
            seconds = int(match.group(1))
            return seconds
    except Exception:
        pass
    return default


def judge_answer_mcq(gt_mcq: str, cand_response: str, retry=3, delay=3, max_attempts=100) -> int:
    """
    MCQ-specific answer evaluation with indefinite retry on rate limit.
    
    Compares a candidate response against ground truth MCQ option in format "A. answer_text".
    Uses GPT-4o to perform flexible matching (candidate can be in any format) with strict
    comparison (both option letter and text must match ground truth).
    
    RATE LIMIT HANDLING: On Azure RateLimitError, extracts retry-after seconds from error
    message and retries indefinitely until evaluation succeeds. Never gives up on rate limits.
    
    Args:
        gt_mcq: Ground truth MCQ option in format "A. answer_text"
        cand_response: Candidate response (any format)
        retry: Number of retries for non-rate-limit errors (default: 3)
        delay: Delay between retries for non-rate-limit errors in seconds (default: 3)
        max_attempts: Maximum total attempts to prevent infinite loops (default: 100)
    
    Returns:
        0: Candidate response is CORRECT (matches ground truth MCQ option)
        1: Candidate response is INCORRECT/HALLUCINATED (does NOT match ground truth MCQ option)
        2: API failure or error during evaluation (non-rate-limit errors only)
    """
    # Convert to strings to handle numeric inputs and prevent AttributeError
    gt_mcq = str(gt_mcq) if gt_mcq is not None else ""
    cand_response = str(cand_response) if cand_response is not None else ""
    
    if cand_response.strip() == "":
        return 1  # Empty response = INCORRECT/HALLUCINATED (return 1)

    messages = [
        {"role": "system",
         "content": ("You are an expert evaluator for multiple-choice question (MCQ) answering systems.\n\n"
                     "Task: Determine if the candidate response is CORRECT or INCORRECT.\n\n"
                     
                     "Ground truth format: '[Letter]. [Answer Text]' (e.g., 'A. Paris')\n\n"
                     
                     "IMPORTANT: Return value semantics:\n"
                     "- Return '0' if the candidate response is CORRECT\n"
                     "- Return '1' if the candidate response is INCORRECT or HALLUCINATED\n"
                     "- Return only '0' or '1'\n\n"
                     
                     "Evaluation Rules:\n"
                     "A response is CORRECT (return 0) if ANY of these conditions are met:\n"
                     "1. BOTH CORRECT: Candidate identifies both the correct option letter AND correct answer text\n"
                     "2. TEXT ONLY CORRECT: Candidate provides only the correct answer text (letter missing is OK)\n"
                     "3. LETTER ONLY CORRECT: Candidate provides only the correct option letter (text missing is OK)\n\n"
                     
                     "A response is INCORRECT (return 1) if ANY of these conditions are met:\n"
                     "1. Wrong text with any letter (correct or incorrect)\n"
                     "2. Wrong letter with any text (correct or incorrect)\n"
                     "3. Both text and letter are wrong\n"
                     "4. Response is empty or contains no answer\n\n"
                     
                     "Detailed Examples:\n\n"
                     
                     "Example 1:\n"
                     "Ground Truth: 'A. Paris'\n"
                     "Candidate: 'A. Paris' → 0 (CORRECT - both letter and text match)\n"
                     "Candidate: 'Paris' → 0 (CORRECT - text is correct)\n"
                     "Candidate: 'A' → 0 (CORRECT - letter is correct)\n"
                     "Candidate: 'The answer is A: Paris' → 0 (CORRECT - both components present and correct)\n"
                     "Candidate: 'B. Paris' → 1 (INCORRECT - letter is wrong)\n"
                     "Candidate: 'A. London' → 1 (INCORRECT - text is wrong)\n"
                     "Candidate: 'B. London' → 1 (INCORRECT - both are wrong)\n"
                     "Candidate: 'London' → 1 (INCORRECT - text is wrong)\n"
                     "Candidate: 'B' → 1 (INCORRECT - letter is wrong)\n"
                     "Candidate: 'I am not sure' → 1 (INCORRECT - no answer)\n\n"
                     
                     "Example 2:\n"
                     "Ground Truth: 'C. The Renaissance'\n"
                     "Candidate: 'C. The Renaissance' → 0 (CORRECT - both match)\n"
                     "Candidate: 'The Renaissance' → 0 (CORRECT - text is correct)\n"
                     "Candidate: 'C' → 0 (CORRECT - letter is correct)\n"
                     "Candidate: 'Option C is The Renaissance' → 0 (CORRECT - both components correct)\n"
                     "Candidate: 'C. Medieval Period' → 1 (INCORRECT - text is wrong)\n"
                     "Candidate: 'B. The Renaissance' → 1 (INCORRECT - letter is wrong)\n"
                     "Candidate: 'Medieval Period' → 1 (INCORRECT - text is wrong)\n"
                     "Candidate: 'B' → 1 (INCORRECT - letter is wrong)\n\n"
                     
                     "Example 3:\n"
                     "Ground Truth: 'D. 1969'\n"
                     "Candidate: 'D. 1969' → 0 (CORRECT - both match)\n"
                     "Candidate: '1969' → 0 (CORRECT - text is correct)\n"
                     "Candidate: 'D' → 0 (CORRECT - letter is correct)\n"
                     "Candidate: 'The year is D: 1969' → 0 (CORRECT - both correct)\n"
                     "Candidate: 'D. 1970' → 1 (INCORRECT - text is wrong)\n"
                     "Candidate: 'A. 1969' → 1 (INCORRECT - letter is wrong)\n"
                     "Candidate: '1970' → 1 (INCORRECT - text is wrong)\n"
                     "Candidate: 'A' → 1 (INCORRECT - letter is wrong)\n\n"
                     
                     "Rules Summary:\n"
                     "- Case-insensitive for option letters (A/a are equivalent)\n"
                     "- Case-insensitive for answer text\n"
                     "- Accept any candidate format as long as components are correctly identified\n"
                     "- Reply with ONLY '0' for CORRECT or '1' for INCORRECT")},
        {"role": "user",
         "content": (f"Ground truth MCQ option:\n{gt_mcq.strip()}\n\n"
                     f"Candidate response:\n{cand_response.strip()}")}
    ]

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            rsp = client.chat.completions.create(
                model       = EVAL_MODEL,
                messages    = messages,
                max_tokens  = 1,
                temperature = 0,
            )
            token = rsp.choices[0].message.content.strip()[:1]
            # token == '0' means CORRECT (no hallucination) → return 0
            # token == '1' means INCORRECT/HALLUCINATION → return 1
            is_hallucination = 1 if token == "1" else 0
            return is_hallucination

        # Treat Azure API/content filter errors as API failure = 2
        except openai.BadRequestError as e:          # includes content-filter hits
            eval_logger.warning(f"API Evaluation Failed - BadRequest (Content Filter/Invalid Request): {e}")
            return 2
        except openai.RateLimitError as e:
            # Extract retry-after seconds from Azure error message
            error_str = str(e)
            retry_after_secs = _extract_retry_after_seconds(error_str, default=60)
            eval_logger.warning(f"API Rate Limit Hit (attempt {attempt}/{max_attempts}) - Waiting {retry_after_secs}s: {e}")
            time.sleep(retry_after_secs)
            # Continue the loop to retry indefinitely on rate limit
            continue
        except openai.OpenAIError as e:
            eval_logger.warning(f"API Evaluation Failed - OpenAI Error: {e}")
            return 2

    # If we exhausted max_attempts, log and return failure
    eval_logger.error(f"Evaluation exhausted max attempts ({max_attempts}) - returning API failure")
    return 2


def batch_judge_answers_mcq(evaluation_pairs, max_workers=10):
    """
    Parallel MCQ evaluation of multiple answer pairs using ThreadPoolExecutor.
    evaluation_pairs: list of (gt_mcq, candidate_response, row_data) tuples
    Returns: list of results in same order as input
    """
    def evaluate_single(pair):
        gt_mcq, cand_response, row_data = pair
        result = judge_answer_mcq(gt_mcq, cand_response)
        return result, row_data
    
    results = [None] * len(evaluation_pairs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(evaluate_single, pair): i 
            for i, pair in enumerate(evaluation_pairs)
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_index), 
                          total=len(evaluation_pairs), 
                          desc="Evaluating MCQ answers"):
            index = future_to_index[future]
            try:
                result, row_data = future.result()
                results[index] = (result, row_data)
            except Exception as e:
                eval_logger.error(f"Evaluation failed for index {index}: {e}")
                results[index] = (1, evaluation_pairs[index][2])  # fallback to hallucination
    
    return results
