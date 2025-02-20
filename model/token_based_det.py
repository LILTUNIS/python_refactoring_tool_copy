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
    def detect_token_clones_with_ast(file_path: str, similarity_threshold: float = 0.8) -> List[Dict[str, float]]:
        """
        Detects token-based clones in a Python file while also utilizing AST-based structure analysis.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        tree = ast.parse(source_code)
        functions = TokenBasedCloneDetector.extract_function_bodies(tree, source_code)
        function_tokens = {func_name: TokenBasedCloneDetector.tokenize_code(code) for func_name, code in
                           functions.items()}

        clones = []
        function_names = list(function_tokens.keys())

        for i in range(len(function_names)):
            for j in range(i + 1, len(function_names)):
                func1, func2 = function_names[i], function_names[j]
                tokens1, tokens2 = function_tokens[func1], function_tokens[func2]

                token_similarity = TokenBasedCloneDetector.compute_token_similarity(tokens1, tokens2)

                # Updated: Extract the FunctionDef node from the parsed code before calculating AST similarity
                try:
                    func1_ast = ast.parse(functions[func1]).body[0]
                    func2_ast = ast.parse(functions[func2]).body[0]
                    ast_similarity = calculate_similarity(func1_ast, func2_ast)  # Use static analyzer's AST similarity
                except Exception as e:
                    ast_similarity = 0.0

                if token_similarity >= similarity_threshold or ast_similarity >= similarity_threshold:
                    clones.append({
                        "func1": func1,
                        "func2": func2,
                        "token_similarity": token_similarity,
                        "ast_similarity": ast_similarity
                    })

        return clones
