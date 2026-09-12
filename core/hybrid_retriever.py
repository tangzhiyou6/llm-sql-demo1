import math
import re
from typing import List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field


class ColumnMetadata(BaseModel):
    name: str
    data_type: str
    business_name: str
    description: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: Optional_Target = None
    sample_values: List[Any] = Field(default_factory=list)


# Optional string target for foreign keys
Optional_Target = Any


class TableMetadata(BaseModel):
    table_name: str
    business_name: str
    description: str
    columns: List[ColumnMetadata]
    foreign_keys: List[Dict[str, str]] = Field(default_factory=list)

    def to_search_document(self) -> str:
        col_names = " ".join([f"{c.name} ({c.business_name}: {c.description})" for c in self.columns])
        fk_str = " ".join([f"{fk.get('from_column')}->{fk.get('to_table')}.{fk.get('to_column')}" for fk in self.foreign_keys])
        return (
            f"Table: {self.table_name} ({self.business_name})\n"
            f"Description: {self.description}\n"
            f"Columns: {col_names}\n"
            f"Foreign Keys: {fk_str}"
        )


class HybridRetriever:
    """
    Two-Level Metadata Hybrid Retrieval Engine.
    Level 1: Table-level Dense + BM25 Hybrid Retrieval + RRF (Reciprocal Rank Fusion).
    Level 2: Column-level pruning inside retrieved tables to avoid context pollution.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.tables: Dict[str, TableMetadata] = {}
        # BM25 table statistics
        self.table_docs: Dict[str, List[str]] = {}
        self.table_doc_lens: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.table_df: Dict[str, int] = {}
        self.num_docs: int = 0

    def register_table(self, table: TableMetadata) -> None:
        """Registers a table metadata entry and indexes it for hybrid search."""
        self.tables[table.table_name] = table
        self._rebuild_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer handling English words, Chinese characters/bi-grams."""
        text_lower = text.lower()
        # English tokens / identifiers
        tokens = re.findall(r"[a-z0-9_]+", text_lower)
        # Chinese characters & bigrams
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text_lower)
        tokens.extend(chinese_chars)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i+1])
        return tokens

    def _rebuild_index(self) -> None:
        self.num_docs = len(self.tables)
        if self.num_docs == 0:
            return

        self.table_docs.clear()
        self.table_doc_lens.clear()
        self.table_df.clear()

        total_len = 0
        for name, table in self.tables.items():
            doc_text = table.to_search_document()
            tokens = self._tokenize(doc_text)
            self.table_docs[name] = tokens
            self.table_doc_lens[name] = len(tokens)
            total_len += len(tokens)

            unique_tokens = set(tokens)
            for tok in unique_tokens:
                self.table_df[tok] = self.table_df.get(tok, 0) + 1

        self.avg_doc_len = total_len / max(self.num_docs, 1)

    def _bm25_search_tables(self, query: str) -> List[Tuple[str, float]]:
        """Sparse BM25 ranking over table documents."""
        q_tokens = self._tokenize(query)
        scores: Dict[str, float] = {name: 0.0 for name in self.tables}
        k1 = 1.5
        b = 0.75

        for tok in q_tokens:
            df = self.table_df.get(tok, 0)
            if df == 0:
                continue
            # IDF with floor
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for name, tokens in self.table_docs.items():
                tf = tokens.count(tok)
                if tf > 0:
                    doc_len = self.table_doc_lens[name]
                    denom = tf + k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1)))
                    scores[name] += idf * (tf * (k1 + 1)) / denom

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _dense_semantic_search_tables(self, query: str) -> List[Tuple[str, float]]:
        """
        Dense concept matching simulating vector similarity.
        Computes cosine overlap of n-gram concept features with heavy weight on business intent.
        """
        q_tokens = set(self._tokenize(query))
        scores: Dict[str, float] = {}

        for name, table in self.tables.items():
            # Match against table business name, description and column business descriptions
            doc_text = f"{table.business_name} {table.description} " + " ".join(
                [f"{c.business_name} {c.description}" for c in table.columns]
            )
            doc_tokens = set(self._tokenize(doc_text))
            intersection = q_tokens & doc_tokens
            # Jaccard / Cosine approximation
            if doc_tokens and q_tokens:
                score = len(intersection) / math.sqrt(len(q_tokens) * len(doc_tokens))
            else:
                score = 0.0
            scores[name] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve_tables_hybrid(self, query: str, top_k: int = 5) -> List[TableMetadata]:
        """
        Level 1: Table-Level Hybrid Retrieval using Reciprocal Rank Fusion (RRF).
        RRF formula: score(d) = 1/(k + rank_bm25) + 1/(k + rank_dense)
        """
        bm25_ranked = self._bm25_search_tables(query)
        dense_ranked = self._dense_semantic_search_tables(query)

        rrf_scores: Dict[str, float] = {name: 0.0 for name in self.tables}

        # Sparse ranks
        for rank, (name, _) in enumerate(bm25_ranked):
            rrf_scores[name] += 1.0 / (self.rrf_k + rank + 1)

        # Dense ranks
        for rank, (name, _) in enumerate(dense_ranked):
            rrf_scores[name] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score
        sorted_tables = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        selected_table_names = [name for name, score in sorted_tables[:top_k]]
        return [self.tables[name] for name in selected_table_names if name in self.tables]

    def prune_and_format_schema(self, query: str, candidate_tables: List[TableMetadata], max_cols_per_table: int = 8) -> str:
        """
        Level 2: Column-Level Pruning inside Candidate Tables.
        Keeps primary keys, foreign keys, and ranks the most query-relevant dimension/measure columns.
        Outputs compact Markdown DDL for prompt injection.
        """
        q_tokens = set(self._tokenize(query))
        lines = ["### [Pruned Schema Context]:"]

        for table in candidate_tables:
            lines.append(f"\nTable: `{table.table_name}` ({table.business_name})")
            lines.append(f"Description: {table.description}")

            # Collect columns: PK and FK are always preserved
            mandatory_cols = []
            optional_cols = []

            for col in table.columns:
                if col.is_primary_key or col.is_foreign_key:
                    mandatory_cols.append(col)
                else:
                    # Score relevance to query
                    col_doc = f"{col.name} {col.business_name} {col.description}"
                    col_tokens = set(self._tokenize(col_doc))
                    relevance = len(q_tokens & col_tokens)
                    optional_cols.append((col, relevance))

            # Sort optional columns by relevance desc
            optional_cols.sort(key=lambda x: x[1], reverse=True)

            # Limit columns
            kept_cols = mandatory_cols[:]
            slots_left = max(0, max_cols_per_table - len(mandatory_cols))
            kept_cols.extend([col for col, score in optional_cols[:slots_left]])

            lines.append("Columns:")
            for c in kept_cols:
                pk_fk_tag = ""
                if c.is_primary_key:
                    pk_fk_tag += " [PRIMARY KEY]"
                if c.is_foreign_key:
                    pk_fk_tag += f" [FOREIGN KEY -> {c.foreign_key_target}]"

                sample_str = f", e.g. {c.sample_values[:3]}" if c.sample_values else ""
                lines.append(f"  - `{c.name}` ({c.data_type}): {c.business_name} - {c.description}{pk_fk_tag}{sample_str}")

            if table.foreign_keys:
                lines.append("Relationships:")
                for fk in table.foreign_keys:
                    lines.append(f"  - Join: `{table.table_name}.{fk['from_column']} = {fk['to_table']}.{fk['to_column']}`")

        return "\n".join(lines)
