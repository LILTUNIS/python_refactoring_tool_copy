import google.generativeai as genai
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GOOGLE_API_KEY environment variable is not set.")

def ai_merge_functions(func_a_code: str, func_b_code: str, new_func_name: str) -> str:
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    prompt = f"""
    Merge the following two Python functions into a single function named '{new_func_name}'.
    Ensure that the merged function keeps all behaviors intact and avoids redundancy.

    Function A:
    {func_a_code}

    Function B:
    {func_b_code}

    Provide only the merged function as output, without extra explanations.
    """

    logger.debug("Calling Gemini API for function merge.")

    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = model.generate_content(prompt)
        merged_code = response.text.strip()
        logger.debug(f"Merged Code: {merged_code[:100]}...")
        return merged_code

    except Exception as e:
        logger.error(f"Gemini API Error: {e}", exc_info=True)
        raise
