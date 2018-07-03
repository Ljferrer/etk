class DocumentUniqueifier(object):

    def __init__(self) -> None:
        self.uniqueifier = "|::|"

    def join_ids_from_doc(self, doc: dict) -> str:
        etk_id = doc["doc_id"]
        lex_id = doc["lexisnexis"]["doc_id"]
        timestamp = doc["@timestamp"]
        identifiers = [etk_id, lex_id, timestamp]
        return self.uniqueifier.join(identifiers)

    def join_ids(self, ids_to_join: list) -> str:
        return self.uniqueifier.join(ids_to_join)

    def split_ids(self, this_id: str) -> list:
        return this_id.split(self.uniqueifier)
