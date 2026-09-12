from typing import List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
import re


class ValueMatch(BaseModel):
    table: str
    column: str
    db_value: Any
    matched_term: str
    score: float
    filter_hint: str


class ValueLookupEngine:
    """
    Independent Value-Lookup Engine for categorical/enumerated DB values.
    Solves the hallucinated values problem (e.g. mapping user query '果汁' -> '100%纯果汁').
    Uses inverted index, substring containment, and character n-gram overlap.
    """

    def __init__(self):
        # Store: term_lower -> list of (table, column, original_val, is_synonym)
        self.inverted_index: Dict[str, List[Tuple[str, str, Any, str]]] = {}
        # Store all unique indexed values for fuzzy scan: (table, column, str_val, original_val)
        self.indexed_entries: List[Tuple[str, str, str, Any]] = []

    def index_column_values(
        self,
        table: str,
        column: str,
        values: List[Any],
        synonyms_map: Dict[str, List[str]] = None
    ) -> None:
        """Indexes all distinct values for a given dimension column, with optional synonyms."""
        synonyms_map = synonyms_map or {}
        for val in values:
            if val is None:
                continue
            str_val = str(val).strip()
            if not str_val:
                continue

            self.indexed_entries.append((table, column, str_val, val))

            # Index exact string
            clean_term = str_val.lower()
            self._add_term(clean_term, table, column, val, str_val)

            # Index sub-tokens (e.g. split by whitespace/underscore/hyphen)
            tokens = re.split(r"[\s_\-]+", clean_term)
            for tok in tokens:
                if len(tok) >= 2:
                    self._add_term(tok, table, column, val, str_val)

            # Index synonyms if provided (e.g. '已支付' -> 'PAID')
            if str_val in synonyms_map:
                for syn in synonyms_map[str_val]:
                    self._add_term(syn.lower().strip(), table, column, val, str_val)

    def _add_term(self, term: str, table: str, column: str, original_val: Any, display_val: str):
        if term not in self.inverted_index:
            self.inverted_index[term] = []
        # Avoid duplicate entries
        entry = (table, column, original_val, display_val)
        if entry not in self.inverted_index[term]:
            self.inverted_index[term].append(entry)

    def search_query(self, user_query: str, top_k: int = 5) -> List[ValueMatch]:
        """
        Scans user query text and identifies exact, token, or substring matches against DB dimension values.
        """
        query_lower = user_query.lower()
        scored_matches: Dict[Tuple[str, str, Any], float] = {}
        match_terms: Dict[Tuple[str, str, Any], str] = {}

        # 1. Check inverted index keywords directly appearing in query
        for term, entries in self.inverted_index.items():
            if len(term) >= 2 and term in query_lower:
                # Calculate relevance score based on length of matched term
                weight = 1.0 if len(term) == len(query_lower) else min(0.95, 0.5 + (len(term) / max(len(query_lower), 10)))
                for table, col, orig_val, disp_val in entries:
                    key = (table, col, orig_val)
                    if key not in scored_matches or scored_matches[key] < weight:
                        scored_matches[key] = weight
                        match_terms[key] = term

        # 2. Substring scan: DB values appearing inside the query or query tokens appearing in DB values
        for table, col, str_val, orig_val in self.indexed_entries:
            key = (table, col, orig_val)
            val_lower = str_val.lower()

            # Exact match
            if val_lower == query_lower:
                scored_matches[key] = 1.0
                match_terms[key] = str_val
            # Value is contained in query (e.g. query has "已完成", value is "已完成")
            elif len(val_lower) >= 2 and val_lower in query_lower:
                score = 0.9
                if key not in scored_matches or scored_matches[key] < score:
                    scored_matches[key] = score
                    match_terms[key] = str_val
            # Part of DB value is mentioned (e.g. query mentions "果汁", DB value is "100%纯果汁")
            else:
                # Check character bigrams overlap
                if len(query_lower) >= 2 and len(val_lower) >= 2:
                    overlap_chars = set(query_lower) & set(val_lower)
                    # Exclude common stop characters
                    stop_chars = set(" 的是否个在这我有你有")
                    informative_overlap = overlap_chars - stop_chars
                    if len(informative_overlap) >= 2:
                        ratio = len(informative_overlap) / max(len(val_lower), len(query_lower))
                        if ratio > 0.3:
                            score = round(0.5 + ratio * 0.4, 3)
                            if key not in scored_matches or scored_matches[key] < score:
                                scored_matches[key] = score
                                match_terms[key] = "".join(informative_overlap)

        # Sort matches by score desc
        sorted_keys = sorted(scored_matches.keys(), key=lambda k: scored_matches[k], reverse=True)[:top_k]
        results = []
        for table, col, orig_val in sorted_keys:
            score = scored_matches[(table, col, orig_val)]
            term = match_terms[(table, col, orig_val)]
            formatted_val = f"'{orig_val}'" if isinstance(orig_val, str) else str(orig_val)
            filter_hint = f"{table}.{col} = {formatted_val}"
            results.append(
                ValueMatch(
                    table=table,
                    column=col,
                    db_value=orig_val,
                    matched_term=term,
                    score=score,
                    filter_hint=filter_hint
                )
            )
        return results

    def format_matches_for_prompt(self, matches: List[ValueMatch]) -> str:
        """Formats Value-Lookup results into an explicit constraint block for LLM prompt."""
        if not matches:
            return "No specific categorical value constraints detected."

        lines = ["### [Value-Lookup Alignment Constraints]:"]
        lines.append("The user query mentions concepts that map directly to the following verified database values:")
        for m in matches:
            lines.append(
                f"- User matched concept '{m.matched_term}' -> Target filter: `{m.filter_hint}` (Confidence: {m.score:.2f})"
            )
        lines.append("CRITICAL: Always use the exact DB value listed above rather than guessing literal text.")
        return "\n".join(lines)
