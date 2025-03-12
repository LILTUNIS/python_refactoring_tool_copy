import google.generativeai as genai
import logging

# Set your Gemini API key
GOOGLE_API_KEY = "AIzaSyC33ksJ5eUmR1Fe87Bv0wWF49WrO9bGtOw"
genai.configure(api_key=GOOGLE_API_KEY)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ai_merge_functions(func_a_code: str, func_b_code: str, new_func_name: str) -> str:
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
