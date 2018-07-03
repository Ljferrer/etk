import os
import json
import time
import spacy

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from idUniqueifier import DocumentUniqueifier

__all__ = ["SentenceExtractor",
           "load_docs",
           "prepare_text_from_docs",
           "make_data_iterator",
           "load_embedder",
           "run_embeddings",
           "prepare_to_serialize",
           "serialize_in_chunks"]


class SentenceExtractor(object):

    def __init__(self) -> None:
        self.nlp = spacy.load('en_core_web_sm')

    def strip_text(self, doc: dict) -> list:
        title = doc["lexisnexis"]["doc_title"].replace("\n", " ")
        text = doc["lexisnexis"]["doc_description"].replace("\n", " ")
        sentences = self.nlp(text)
        sentences = [sent.string.strip() for sent in sentences.sents]
        sentences.insert(0, title)

        return sentences


def load_docs(filename: str, start: int = 0, end: int = float("inf")) -> list:
    cdr_docs = list()
    loaded_doc_ids = set()

    # Report indexing
    name = filename.split("/")[-1]
    if start == 0 and end == float("inf"):
        print("  Loading all documents in {}...".format(name))
    else:
        if end == float("inf"):
            ending = "end"
        else:
            ending = end
        print("  Loading documents {} through {} from {}...".format(start, ending, name))

    skipped = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if start <= i < end:
                raw_doc = json.loads(line)

                # Ensure document is unique and has content
                doc_id = raw_doc["doc_id"]
                if doc_id in loaded_doc_ids:
                    skipped += 1
                else:
                    content = raw_doc["lexisnexis"]["doc_description"]
                    if content != "" and content != "DELETED_STORY":
                        loaded_doc_ids.add(doc_id)
                        cdr_docs.append(raw_doc)
                    else:
                        skipped += 1

    print("  {} docs loaded".format(len(cdr_docs)))
    if skipped > 0:
        print("  Note: {} non-unique/empty docs skipped".format(skipped))

    return cdr_docs


def prepare_text_from_docs(all_docs: list, batch_size: int = 500) -> tuple:
    tensor_batches = list()
    init_doc_emb_dicts = list()

    uniqueifier = DocumentUniqueifier()
    sentence_extractor = SentenceExtractor()

    biggest_doc = 0
    all_sentences_by_doc = list()
    for doc in all_docs:

        # Make unique ID
        unique_id = uniqueifier.join_ids_from_doc(doc)

        # Get sentences
        doc_sentences = sentence_extractor.strip_text(doc)
        all_sentences_by_doc.append(doc_sentences)

        # Find length of longest document for tensor padding
        n_sentences = len(doc_sentences)
        if n_sentences > biggest_doc:
            biggest_doc = n_sentences

        # Initialize doc_emb_dict
        doc_emb_dict = dict()
        doc_emb_dict[unique_id] = n_sentences
        init_doc_emb_dicts.append(doc_emb_dict)

    sentence_batch = list()
    all_sentence_batches = list()
    all_sentences_by_doc_copy = list(all_sentences_by_doc)
    one_doc_sentences = all_sentences_by_doc_copy.pop(0)
    while len(all_sentences_by_doc_copy) >= 0:
        if len(one_doc_sentences) == 0 and len(all_sentences_by_doc_copy) > 0:
            one_doc_sentences = all_sentences_by_doc_copy.pop(0)

        elif len(sentence_batch) < batch_size and len(one_doc_sentences) > 0:
            sentence_batch.append(one_doc_sentences.pop(0))

        elif len(sentence_batch) == batch_size:
            all_sentence_batches.append(sentence_batch)
            sentence_batch = list()

        else:
            all_sentence_batches.append(sentence_batch)
            break

    # Convert batches to tensors
    for i, sentence_batch in enumerate(all_sentence_batches):
        # Pad last batch to be uniform length
        while len(sentence_batch) < batch_size:
            sentence_batch.append("")
        tensor_batch = tf.constant(sentence_batch, dtype=tf.string)
        tensor_batches.append(tensor_batch)

    return tensor_batches, init_doc_emb_dicts


def make_data_iterator(list_of_tensors: list) -> tf.data.Iterator:
    dataset = tf.data.Dataset.from_tensor_slices(list_of_tensors)
    return dataset.make_one_shot_iterator()


def load_embedder(data: tf.data.Iterator, model_location: str = None,
                  large_model: bool = False) -> hub.Module:

    if model_location is None and large_model is False:
        model_location = "https://tfhub.dev/google/universal-sentence-encoder/2"
    elif model_location is None and large_model is True:
        model_location = "https://tfhub.dev/google/universal-sentence-encoder-large/2"

    embedder = hub.Module(model_location)
    return embedder(data.get_next())


def session_initializer(sess: tf.Session) -> None:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])


def run_embeddings(embedder_model: hub.Module,
                   print_progress_report: bool = False,
                   report_interval: int = 1000) -> list:

    if print_progress_report:
        t_init = time.time()

    sess = tf.Session()
    session_initializer(sess)

    if print_progress_report:
        i = 0
        t_start = time.time()
        print("Time spent initializing Tensor Flow Graph: {:.4f}s".format(t_start-t_init))

    tensor_embeddings = list()
    while True:
        try:
            tensor_embeddings.append(sess.run(embedder_model))

            if print_progress_report:
                i += 1
                if i % report_interval == 0:
                    t_run = time.time()
                    print("  {} document batches embedded in {:.4f}s".format(i, t_run-t_start))

        except tf.errors.OutOfRangeError:

            if print_progress_report:
                t_end = time.time()
                print("  {} document batches embedded in {:.4f}s".format(i, t_end-t_start))

            sess.close()
            if print_progress_report:
                t_close = time.time()
                print("Time spent creating embeddings: {:.4f}s".format(t_close-t_init))

            return tensor_embeddings


def gen_norm_embeddings(all_embedding_tensors: list, init_doc_emb_dicts: list) -> np.array:

    counter = 0
    bookmark = 0
    all_doc_emb_dicts = list(init_doc_emb_dicts)
    tensor_batch = np.array(all_embedding_tensors.pop(0))
    while len(all_doc_emb_dicts) >= 0:

        assert counter >= 0, "Counter became negative somehow"

        if counter == 0:
            doc_emb_dict = all_doc_emb_dicts.pop(0)
            assert len(doc_emb_dict.items()) == 1, "Improperly initialized doc_emb_dict"

            for _, n_sentences_this_doc in doc_emb_dict.items():
                counter = n_sentences_this_doc

        elif bookmark+counter < tensor_batch.shape[0]:
            one_doc_embeddings = tensor_batch[bookmark:bookmark+counter]
            norm_doc_embeddings = one_doc_embeddings / np.linalg.norm(one_doc_embeddings, axis=0)

            bookmark += counter
            counter = 0
            yield list(norm_doc_embeddings.tolist())

        elif bookmark <= tensor_batch.shape[0] <= bookmark+counter:
            first_half = tensor_batch[bookmark:]
            counter -= tensor_batch.shape[0] - bookmark
            assert counter >= 0, "Cnt: {}, Bmk: {}, Shape: {}".format(counter, bookmark, tensor_batch.shape[0])
            bookmark = 0

            if len(all_embedding_tensors) > 0:
                tensor_batch = np.array(all_embedding_tensors.pop(0))
                second_half = tensor_batch[:counter]
                one_doc_embeddings = np.concatenate((first_half, second_half))

            else:
                one_doc_embeddings = first_half

            norm_doc_embeddings = one_doc_embeddings / np.linalg.norm(one_doc_embeddings, axis=0)

            bookmark += counter
            counter = 0
            yield list(norm_doc_embeddings.tolist())

        else:
            print("Error: Unhandled case...")
            print("Cnt: {}, Bmk: {}".format(counter, bookmark))
            print("Tensor_batch.shape: {}".format(tensor_batch.shape))

            return None


def prepare_to_serialize(all_embedding_tensors: list,
                         init_doc_emb_dicts: list,
                         all_docs: list) -> list:

    all_serializable_docs = list()

    uniqueifier = DocumentUniqueifier()

    # Format tensors into serializable list
    embedding_generator = gen_norm_embeddings(all_embedding_tensors, init_doc_emb_dicts)

    # Create datastructure to serialize
    for doc_emb_dict, doc_embs, original_doc in zip(init_doc_emb_dicts, embedding_generator, all_docs):

        # Sanity check to ensure embeddings correspond to correct doc
        assert len(doc_emb_dict.items()) == 1, "Improperly initialized doc_emb_dict"
        for unique_id, n_sentences_this_doc in doc_emb_dict.items():
            assert n_sentences_this_doc == len(doc_embs), "Document length mismatch" \
                "n_sent: {}, len(doc_embs): {}".format(n_sentences_this_doc, len(doc_embs))
            etk_id, lexis_id, timestamp = uniqueifier.split_ids(unique_id)
            assert etk_id == original_doc["doc_id"], "ID mismatch: ETK"
            assert lexis_id == original_doc["lexisnexis"]["doc_id"], "ID mismatch: LexisNexis"
            assert timestamp == original_doc["@timestamp"], "ID mismatch: @timestamp"
        # Sanity check passed

        for i, embedding in enumerate(doc_embs):
            unique_id = uniqueifier.join_ids([etk_id, lexis_id, timestamp, str(i)])
            doc_emb_dict[unique_id] = embedding

        all_serializable_docs.append(doc_emb_dict)

    return all_serializable_docs


def serialize(serializable_docs: list, output_filename: str) -> None:
    with open(output_filename, "xb") as f:
        for doc in serializable_docs:
            doc_dict = json.dumps(doc) + "\n"
            f.write(doc_dict.encode())


def serialize_in_chunks(all_serializable_docs: list,
                        output_filename_prefix: str,
                        output_filetype: str = ".json",
                        directory: str = "",
                        chunk_size: int = 5000) -> None:

    n_docs = len(all_serializable_docs)
    if n_docs < chunk_size:
        m_full_chunks = -1
    else:
        m_full_chunks = n_docs // chunk_size

    if directory == "":
        directory = os.getcwd()
        print("Output file directory set to {}".format(directory))

    if not os.path.isdir(directory):
        print("{} must be an existing directory".format(directory))
        directory = input("Please input a properly formatted directory: ")
        assert os.path.isdir(directory), \
            "Path Error: {} must be an existing directory".format(directory)

    # Save full chunks in individual files
    for chunk_num in range(m_full_chunks):
        subset_of_docs = all_serializable_docs[chunk_num * chunk_size:(chunk_num+1) * chunk_size]
        filename = directory + output_filename_prefix + str(chunk_num) + output_filetype
        serialize(subset_of_docs, filename)

    # Save final chunk
    if m_full_chunks > 0:
        subset_of_docs = all_serializable_docs[m_full_chunks * chunk_size:]
    else:
        subset_of_docs = all_serializable_docs
    filename = directory + output_filename_prefix + str(m_full_chunks) + output_filetype
    serialize(subset_of_docs, filename)
