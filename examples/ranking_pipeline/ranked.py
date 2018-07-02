import json
import time

from ranker import Ranker


def main():
    t0 = time.clock()

    query_title = './resources/ifps_titles_test.txt'
    top_k = 5
    in_filename = './resources/new_2018-04-03-first-10000.jl'

    queries = list()
    with open(query_title) as f:
        for line in f:
            queries.append(line)

    ranker = Ranker(queries=queries, k=top_k, json_doc=in_filename)
    print('Initialized in {} seconds'.format(time.clock()-t0))

    t1 = time.clock()
    f_queries = ranker.format_queries()
    print('Query formatted in {} seconds'.format(time.clock()-t1))

    t2 = time.clock()
    docs = ranker.load_documents()
    print('Documents loaded in {} seconds'.format(time.clock()-t2))

    t3 = time.clock()
    sel_docs = ranker.select_documents(f_queries, docs)
    print('Selected in {} seconds'.format(time.clock()-t3))

    t4 = time.clock()
    by_sentences = False
    ranked = ranker.rank_all(f_queries, docs, sel_docs, by_sentences)
    print('Ranked in {} seconds'.format(time.clock()-t4))

    # Save ranked results
    if by_sentences:
        out_filename = './resources/output/Results_Ranked_by_Sentences.pickle'
    else:
        out_filename = './resources/output/Results_Ranked_by_Titles.pickle'

    results = [f_queries, ranked, by_sentences]
    with open(out_filename, 'wb') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
