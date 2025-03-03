#token_based_det.py
import tokenize
import io
import ast
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
from model.static_analyzer import calculate_similarity  # Import AST-based similarity function


class TokenBasedCloneDetector:
    @staticmethod
    def tokenize_code(source_code: str) -> List[Tuple[str, str]]:
        """
        Tokenizes Python source code while removing whitespace and comments.
        """
        tokens = []
        try:
            token_stream = tokenize.generate_tokens(io.StringIO(source_code).readline)
            for toknum, tokval, _, _, _ in token_stream:
                if toknum in {tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT}:
                    continue
                tokens.append((tokenize.tok_name[toknum], tokval))
        except tokenize.TokenError as e:
            print(f"ERROR: Tokenization failed: {e}")
        return tokens

    @staticmethod
    def compute_token_similarity(tokens1: List[Tuple[str, str]], tokens2: List[Tuple[str, str]]) -> float:
        """
        Computes similarity between two token sequences.
        """
        token_str1 = " ".join(f"{tok[0]}:{tok[1]}" for tok in tokens1)
        token_str2 = " ".join(f"{tok[0]}:{tok[1]}" for tok in tokens2)
        return SequenceMatcher(None, token_str1, token_str2).ratio()

    @staticmethod
    def extract_function_bodies(tree: ast.AST, source_code: str) -> Dict[str, str]:
        """
        Extracts function bodies from an AST tree.
        """
        function_bodies = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line + 1
                function_bodies[node.name] = "\n".join(source_code.splitlines()[start_line:end_line])
        return function_bodies

    @staticmethod
    def detect_token_clones_with_ast(file_path: str, similarity_threshold: float = 0.8) -> List[Dict[str, any]]:
        """
        Detects token-based clones in a Python file while also utilizing AST-based structure analysis.

        """

        # Step 1: Read the source code from the provided file path
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Step 2: Parse the source code into an Abstract Syntax Tree (AST)
        tree = ast.parse(source_code)

        # Step 3: Extract function bodies from the AST
        functions = TokenBasedCloneDetector.extract_function_bodies(tree, source_code)

        # Step 4: Tokenize function bodies for token-based similarity comparison
        function_tokens = {
            func_name: TokenBasedCloneDetector.tokenize_code(code)
            for func_name, code in functions.items()
        }

        # Step 5: Initialize list to store detected function clones
        clones = []

        # Step 6: Create a list of function names for pairwise comparison
        function_names = list(function_tokens.keys())

        # Step 7: Compare each function with every other function (pairwise comparison)
        for i in range(len(function_names)):
            for j in range(i + 1, len(function_names)):  # Avoid duplicate comparisons

                func1, func2 = function_names[i], function_names[j]
                tokens1, tokens2 = function_tokens[func1], function_tokens[func2]

                # Step 8: Compute token similarity between the two functions
                token_similarity = TokenBasedCloneDetector.compute_token_similarity(tokens1, tokens2)

                # Step 9: Extract AST structures before calculating AST-based similarity
                try:
                    func1_ast = ast.parse(functions[func1]).body[0]  # Parse AST for first function
                    func2_ast = ast.parse(functions[func2]).body[0]  # Parse AST for second function
                    ast_similarity = calculate_similarity(func1_ast, func2_ast)  # Compute AST similarity
                except Exception as e:
                    ast_similarity = 0.0  # Set AST similarity to 0 if parsing fails

                # Debug output for developers to track function comparisons

                # Step 10: Store clone information if similarity meets the threshold
                if token_similarity >= similarity_threshold or ast_similarity >= similarity_threshold:
                    clones.append({
                        "func1": func1,
                        "func2": func2,
                        "token_similarity": token_similarity,
                        "ast_similarity": ast_similarity,
                        "func1_code": functions[func1],  # Store source code of function 1
                        "func2_code": functions[func2],  # Store source code of function 2
                    })

        # Step 11: Return the list of detected function clones
        return clones
