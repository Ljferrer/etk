import os
import sys
import time

from doc_preprocessing_funcs import *


def main():

    t0 = time.time()

    # docs_file = sys.argv[1]
    docs_file = "./resources/new_2018-04-03-first-10000.jl"
    cdr_docs = load_docs(docs_file)
    n_docs = len(cdr_docs)

    t1 = time.time()
    print("Time spent loading docs: {:.4f}s".format(t1-t0))

    all_tensors, init_doc_dicts = prepare_text_from_docs(cdr_docs)

    t2 = time.time()
    print("Time spent preparing text from docs: {:.4f}s".format(t2-t1))

    input_docs = make_data_iterator(all_tensors)

    t3 = time.time()
    print("Time spent making iterable Dataset object: {:.4f}s".format(t3-t2))

    embedder = load_embedder(input_docs)

    t4 = time.time()
    print("Time spent loading Universal Sentence Encoder: {:.4f}s".format(t4-t3))

    tensor_embeddings = run_embeddings(embedder,
                                       print_progress_report=True,
                                       report_interval=100)

    t5 = time.time()
    print("Time spent embedding {} documents: {:.4f}s".format(n_docs, t5-t4))

    embedded_docs = prepare_to_serialize(tensor_embeddings, init_doc_dicts, cdr_docs)

    t6 = time.time()
    print("Time spent preparing document embeddings for serialization: {:.4f}s".format(t6-t5))

    # file_prefix = sys.argv[3]
    file_prefix = "speed_test_embeddings"
    # directory = sys.argv[2]
    directory = os.getcwd() + "/resources/"
    serialize_in_chunks(embedded_docs,
                        file_prefix,
                        directory=directory)

    t7 = time.time()
    print("Time spent serializing document embeddings: {:.4f}s".format(t7-t6))
    print("Program finished in {:.4f}s".format(t7-t0))


if __name__ == "__main__":
    main()
