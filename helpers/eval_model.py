"""Azure Evaluator Model for Question Answering

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


def test_evaluator_connectivity():
    """
    Test connectivity to the Azure OpenAI evaluator service.
    
    Sends a simple ping request to verify that the Azure OpenAI API is accessible
    and responding correctly. This function should be called before processing
    to ensure the evaluator is available.
    
    Returns:
        bool: True if connectivity test passes
    
    Raises:
        RuntimeError: If the evaluator returns an empty response
        Exception: If the evaluator fails to respond or connection fails
    
    Logs:
        - INFO: Test initiation and successful response
        - ERROR: Connection failures or unexpected responses, with abort message
    """
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

def judge_answer(gt: str, cand: str, retry=3, delay=3) -> int:
    """
    Returns 1 if cand conveys the same factual answer as gt, else 0.
    Gracefully marks API failures as 2 for downstream filtering.
    """
    # Convert to strings to handle numeric inputs and prevent AttributeError
    gt = str(gt) if gt is not None else ""
    cand = str(cand) if cand is not None else ""
    
    if cand.strip() == "":
        return 1

    messages = [
        {"role": "system",
         "content": ("You are an expert evaluator for question-answering systems.\n\n"
                     "Task: Determine if the candidate answer provides the same factual information as the reference answer, allowing for synonyms and equivalent expressions.\n\n"

                     "Evaluation criteria:\n"
                     "1. SEMANTIC EQUIVALENCE: The candidate must convey the same core factual information as the reference\n"
                     "2. SYNONYM ACCEPTANCE: Accept direct synonyms, alternative phrasings, and equivalent expressions\n"
                     "3. PRECISION REQUIREMENT: The candidate must be factually correct and complete for the question asked\n"
                     "4. CONTEXT TOLERANCE: Accept additional context as long as the core answer is present and correct\n\n"

                     "Evaluation examples:\n\n"
                     "Q: What movie starred Tom Cruise?\nRef: Top Gun\nCandidate: 'Top Gun' → 1 (exact match)\n"
                     "Q: What movie starred Tom Cruise?\nRef: Top Gun\nCandidate: 'The movie Top Gun' → 1 (synonym: 'the movie')\n"
                     "Q: What movie starred Tom Cruise?\nRef: Top Gun\nCandidate: 'Top Gun starring Tom Cruise' → 1 (contains correct answer with context)\n"
                     "Q: What movie starred Tom Cruise?\nRef: Top Gun\nCandidate: 'Brad Pitt was in Top Gun' → 0 (wrong person)\n\n"
                     "Q: What genre is the film?\nRef: action\nCandidate: 'action' → 1 (exact match)\n"
                     "Q: What genre is the film?\nRef: action\nCandidate: 'action film' → 1 (synonym: 'action film' = 'action')\n"

                     "Q: What city is the capital?\nRef: Paris\nCandidate: 'Paris' → 1 (exact match)\n"
                     "Q: What city is the capital?\nRef: Paris\nCandidate: 'the capital is Paris' → 1 (equivalent phrasing)\n"
                     "Q: What city is the capital?\nRef: Paris\nCandidate: 'What city is the capital?' → 0 (repeats question)\n\n"

                     "Q: Who directed the film?\nRef: Spielberg\nCandidate: 'Spielberg' → 1 (exact match)\n"
                     "Q: Who directed the film?\nRef: Spielberg\nCandidate: 'Steven Spielberg' → 1 (synonym: full name)\n"
                     "Q: Who directed the film?\nRef: Spielberg\nCandidate: 'Spielberg directed many films' → 1 (contains correct answer)\n\n"

                     "Q: What year was it released?\nRef: 1995\nCandidate: '1995' → 1 (exact match)\n"
                     "Q: What year was it released?\nRef: 1995\nCandidate: 'in 1995' → 1 (equivalent phrasing)\n"
                     "Q: What year was it released?\nRef: 1995\nCandidate: 'I have no idea' → 0 (no answer provided)\n\n"

                     "Q: What state is it in?\nRef: California\nCandidate: 'California' → 1 (exact match)\n"
                     "Q: What state is it in?\nRef: California\nCandidate: 'Located in Los Angeles' → 1 (geographic equivalent: LA is in CA)\n"
                     "Q: What state is it in?\nRef: California\nCandidate: 'CA' → 1 (abbreviation synonym)\n\n"
                     
                     "Strict rules:\n"
                     "- '1' ONLY if candidate provides the same factual information as reference (allowing synonyms/equivalents)\n"
                     "- '0' if wrong facts, incomplete answers, question repetition, no answer, or wrong entity\n"
                     "- Accept synonyms, abbreviations, alternative phrasings, and geographic/temporal equivalents\n"
                     "- Reject partial or ambiguous answers that don't clearly specify the requested information\n"
                     "- Additional context is acceptable only if it doesn't contradict or obscure the core answer\n\n"
                     "Reply with only '1' or '0'.")},
        {"role": "user",
         "content": (f"Reference answer:\n{gt.strip()}\n\n"
                     f"Candidate answer:\n{cand.strip()}")}
    ]

    for _ in range(retry):
        try:
            rsp = client.chat.completions.create(
                model       = EVAL_MODEL,
                messages    = messages,
                max_tokens  = 1,
                temperature = 0,
            )
            token = rsp.choices[0].message.content.strip()[:1]
            # token == '1' means CORRECT per prompt → hallucination = 0
            is_correct = 1 if token == "1" else 0
            is_hallucination = 1 - is_correct
            return is_hallucination

        # Treat Azure API/content filter errors as API failure = 2
        except openai.BadRequestError as e:          # includes content-filter hits
            eval_logger.warning(f"Judge skipped (BadRequest): {e}")
            return 2
        except openai.RateLimitError:
            time.sleep(delay)
        except openai.OpenAIError as e:
            eval_logger.warning(f"Judge skipped (OpenAIError): {e}")
            return 2

    return 2  # fallback if retries exhausted


def batch_judge_answers(evaluation_pairs, max_workers=10):
    """
    Parallel evaluation of multiple answer pairs using ThreadPoolExecutor.
    evaluation_pairs: list of (gt_answer, candidate_answer, row_data) tuples
    Returns: list of results in same order as input
    """
    def evaluate_single(pair):
        gt, cand, row_data = pair
        result = judge_answer(gt, cand)
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
                          desc="Evaluating answers"):
            index = future_to_index[future]
            try:
                result, row_data = future.result()
                results[index] = (result, row_data)
            except Exception as e:
                eval_logger.error(f"Evaluation failed for index {index}: {e}")
                results[index] = (0, evaluation_pairs[index][2])  # fallback
    
    return results

