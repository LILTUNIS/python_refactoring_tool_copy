import tokenize
import io
import ast
from typing import List, Tuple, Dict, Any
from services.static.ast_utils import normalized_ast_string  # Import the centralized normalization

class TokenBasedCloneDetector:
    @staticmethod
    def tokenize_code(source_code: str) -> List[Tuple[str, str]]:
        """
        Tokenizes Python source code while removing whitespace and comments.
        Returns a list of (token_type, token_value).
        """
        tokens = []
        try:
            token_stream = tokenize.generate_tokens(io.StringIO(source_code).readline)
            for toknum, tokval, _, _, _ in token_stream:
                # Skip comments, newlines, indentation tokens, etc.
                if toknum in {tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT}:
                    continue
                tokens.append((tokenize.tok_name[toknum], tokval))
        except tokenize.TokenError as e:
            print(f"ERROR: Tokenization failed: {e}")
        print(f"[DEBUG] Tokenized {len(tokens)} tokens.")
        return tokens

    @staticmethod
    def _ngram_list(tokens: List[str], n: int) -> List[str]:
        """
        Generate n-grams from a list of token strings.
        Returns a list (not a set) so we can count frequencies.
        """
        if len(tokens) < n:
            # If not enough tokens, treat the entire set as one n-gram.
            return [" ".join(tokens)]
        ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        print(f"[DEBUG] Generated {len(ngrams)} {n}-grams.")
        return ngrams

    @staticmethod
    def gather_ngrams(token_pairs: List[Tuple[str, str]],
                      n_values=(1, 2, 3),
                      filter_small_ops: bool = False) -> Dict[str, int]:
        """
        Build a frequency dictionary for multiple n-gram sizes.
        If filter_small_ops=True, skip single-character operators to reduce noise.
        """
        token_strs = []
        for ttype, tval in token_pairs:
            if filter_small_ops and len(tval) == 1 and not tval.isalnum():
                continue
            token_strs.append(f"{ttype}:{tval}")
        print(f"[DEBUG] Token strings: {token_strs}")
        freq_map: Dict[str, int] = {}
        for n in n_values:
            ngram_list = TokenBasedCloneDetector._ngram_list(token_strs, n)
            for ngram in ngram_list:
                freq_map[ngram] = freq_map.get(ngram, 0) + 1
        print(f"[DEBUG] Frequency map has {len(freq_map)} unique n-grams.")
        return freq_map

    @staticmethod
    def weighted_jaccard(freqs1: Dict[str, int], freqs2: Dict[str, int]) -> float:
        """
        Weighted Jaccard similarity for two frequency dictionaries of n-grams.
        Weighted Jaccard = sum(min(freq1, freq2)) / sum(max(freq1, freqs2)).
        """
        if not freqs1 and not freqs2:
            return 1.0
        if not freqs1 or not freqs2:
            return 0.0

        all_keys = set(freqs1.keys()).union(set(freqs2.keys()))
        numerator = 0
        denominator = 0
        for key in all_keys:
            f1 = freqs1.get(key, 0)
            f2 = freqs2.get(key, 0)
            numerator += min(f1, f2)
            denominator += max(f1, f2)
        print(f"[DEBUG] Weighted Jaccard: numerator={numerator}, denominator={denominator}")
        if denominator == 0:
            return 0.0
        sim = numerator / denominator
        print(f"[DEBUG] Weighted Jaccard similarity: {sim:.4f}")
        return sim

    @staticmethod
    def compute_multi_ngram_similarity(
        tokens1: List[Tuple[str, str]],
        tokens2: List[Tuple[str, str]],
        filter_small_ops: bool = False
    ) -> float:
        """
        Compute a combined similarity across multiple n-gram sizes,
        using weighted Jaccard for each n, then averaging.
        """
        n_values = (1, 2, 3)
        freqs1 = TokenBasedCloneDetector.gather_ngrams(tokens1, n_values, filter_small_ops=filter_small_ops)
        freqs2 = TokenBasedCloneDetector.gather_ngrams(tokens2, n_values, filter_small_ops=filter_small_ops)
        sim = TokenBasedCloneDetector.weighted_jaccard(freqs1, freqs2)
        print(f"[DEBUG] Combined multi-n-gram similarity: {sim:.4f}")
        return sim

    @staticmethod
    def detect_token_clones_with_ast(
        file_path: str,
        token_threshold: float,
        filter_small_ops: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detects token-based clones in a Python file using multi-n-gram Weighted Jaccard.
        Uses normalized_ast_string for each function’s code, tokenizes it,
        and checks if the similarity is above the token_threshold.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Parse the entire file into an AST
        tree = ast.parse(source_code)
        # Extract function bodies as a list of (func_name, func_code)
        functions = TokenBasedCloneDetector.extract_function_bodies(tree, source_code)
        print(f"[DEBUG] Extracted functions: {[f[0] for f in functions]}")

        # Normalize each function’s code and store in a list of (func_name, normalized_code)
        normalized_functions: List[Tuple[str, str]] = []
        for (func_name, code) in functions:
            try:
                func_ast = ast.parse(code)
                normalized_code = normalized_ast_string(func_ast)
                normalized_functions.append((func_name, normalized_code))
                print(f"[DEBUG] Normalized code for {func_name} ({len(normalized_code)} characters.)")
            except Exception as e:
                print(f"Normalization failed for {func_name}: {e}")
                # Fallback to original code if normalization fails
                normalized_functions.append((func_name, code))

        # Tokenize each normalized function and store in a list of (func_name, tokens, normalized_code)
        function_tokens: List[Tuple[str, List[Tuple[str, str]], str]] = []
        for (func_name, normalized_code) in normalized_functions:
            tokens = TokenBasedCloneDetector.tokenize_code(normalized_code)
            function_tokens.append((func_name, tokens, normalized_code))
            print(f"[DEBUG] {func_name} token count: {len(tokens)}")

        # Compare each function against every other for clone detection
        clones = []
        for i in range(len(function_tokens)):
            for j in range(i + 1, len(function_tokens)):
                func1_name, tokens1, code1 = function_tokens[i]
                func2_name, tokens2, code2 = function_tokens[j]
                print(f"[DEBUG] Comparing {func1_name} and {func2_name}")
                sim_score = TokenBasedCloneDetector.compute_multi_ngram_similarity(
                    tokens1, tokens2, filter_small_ops
                )
                print(f"[DEBUG] Similarity between {func1_name} and {func2_name}: {sim_score:.4f}")
                if sim_score >= token_threshold and func1_name.strip() and func2_name.strip():
                    clones.append({
                        "func1": func1_name.strip(),
                        "func2": func2_name.strip(),
                        "token_similarity": sim_score,
                        "func1_code": code1,
                        "func2_code": code2,
                    })

        print("\n========== TOKEN CLONE DETECTION ==========")
        print(f"[DEBUG] Token clone detection found {len(clones)} clone(s)")
        for clone in clones:
            print(
                f"✅ Clone Found -> Func1: {clone['func1']}, Func2: {clone['func2']}, Token Similarity: {clone['token_similarity']:.4f}"
            )
        return clones

    @staticmethod
    def extract_function_bodies(tree: ast.AST, source_code: str) -> List[Tuple[str, str]]:
        """
        Extracts function bodies from an AST tree using accurate source code slicing.
        Returns a list of (function_name, function_code) to allow duplicates.
        """
        function_bodies: List[Tuple[str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get the function's source code segment
                func_code = ast.get_source_segment(source_code, node)
                if func_code and node.name.strip():
                    function_bodies.append((node.name.strip(), func_code))
        return function_bodies
