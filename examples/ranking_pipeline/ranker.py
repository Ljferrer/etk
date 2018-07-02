import re
import json
import spacy

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from etk.knowledge_graph import KGSchema
from etk.etk import ETK
from etk.extractors.date_extractor import DateExtractor
from etk.document_selector import DefaultDocumentSelector
from etk.doc_retrieve_processor import DocRetrieveProcessor


class Ranker(object):

    """
    Ranker creates 3 dictionaries that help query/doc ranking:
        1) formatted_queries: dict  -> { original_query: [no_date_query, query_ents_regex] }
        2) docs: dict               -> { doc_id: doc_object }
        3) selected_doc_ids: dict   -> { original_query: doc_id }

    The .rank_all() method uses these dictionaries to rank the documents by similarity to the queries

        top_ranked_docs: dict       -> { original_query: [top_k_docs] }

    """

    def __init__(self, queries: list, k: int, json_doc: str,
                 master_config: dict = None) -> None:

        self.queries = queries
        self.top_k = k
        self.json_doc = json_doc

        if master_config is None:
            master_config = {"fields":
                                {
                                    "developer": {"type": "string"},
                                    "student_developer": {"type": "string"},
                                    "spacy_name": {"type": "string"},
                                    "date": {"type": "date"}
                                }
                            }

        self.kg_schema = KGSchema(master_config)
        self.etk = ETK(self.kg_schema, ["./extraction_modules/"])
        self.date_extractor = DateExtractor(etk=self.etk)
        self.doc_selector = DefaultDocumentSelector()
        self.nlp = spacy.load("en_core_web_lg")

    def format_queries(self) -> dict:
        formatted_queries = dict()

        for query in self.queries:
            original_query = query.replace("?", "").replace("\n", "")

            # Find dates
            dates = self.date_extractor.extract(text=original_query)

            # Remove all dates
            form_query = original_query
            for date in dates:
                d_start, d_end = float("inf"), -1
                d_start = min(d_start, date.provenance["start_char"])
                d_end = max(d_end, date.provenance["end_char"])
                form_query = form_query[:d_start] + form_query[d_end+1:]

            # Regex entities
            query_ents_regex = list()
            nlp_query = self.nlp(form_query)
            for ent in nlp_query.ents:
                query_ents_regex.append(re.escape(ent.text.strip()))
            query_ents_regex = [nf_ent for nf_ent in query_ents_regex if nf_ent]    # Pythonic False element removal

            formatted_queries[original_query] = [form_query, query_ents_regex]

        return formatted_queries

    def load_documents(self) -> dict:
        docs = dict()

        with open(self.json_doc) as f:
            for line in f:
                json_obj = json.loads(line)
                document = self.etk.create_document(json_obj)
                docs[document.doc_id] = document

        return docs

    def select_documents(self, formatted_queries: dict, docs: dict) -> dict:
        selected_doc_ids = dict()

        for original_query, [_, query_ents_regex] in formatted_queries.items():
            keep_docs = list()
            for doc_id, doc in docs.items():
                if self.doc_selector.select_document(document=doc,
                                                     json_paths=["$.lexisnexis.doc_description"],
                                                     json_paths_regex=query_ents_regex):
                    keep_docs.append(doc_id)
            selected_doc_ids[original_query] = keep_docs

        return selected_doc_ids

    def rank_all(self, formatted_queries: dict, docs: dict, selected_doc_ids: dict,
                 sentences: bool = False) -> dict:
        top_ranked_docs = dict()

        for original_query, [no_date_query, _] in formatted_queries.items():
            doc_ret_pro = DocRetrieveProcessor(etk=self.etk, ifp_id="1234",
                                               ifp_title=no_date_query, orig_ifp_title=original_query)

            top_docs = list()
            for doc_id in selected_doc_ids[original_query]:
                document = docs[doc_id]
                if sentences:
                    processed_doc_obj = doc_ret_pro.process_by_sentence(doc=document, threshold=0).cdr_document
                else:
                    processed_doc_obj = doc_ret_pro.process_by_title(doc=document, threshold=0).cdr_document

                top_docs.append([processed_doc_obj["similarity"], doc_id, processed_doc_obj])
            top_docs = sorted(top_docs, key=lambda x: x[0], reverse=True)   # Sort by descending similarity
            top_ranked_docs[original_query] = top_docs[:self.top_k]

        return top_ranked_docs
