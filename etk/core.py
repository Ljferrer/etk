# import all extractors
from data_extractors import spacy_extractor
from data_extractors import landmark_extraction
from data_extractors import dictionary_extractor
from structured_extractors import ReadabilityExtractor, TokenizerExtractor
import json
import gzip
import re
import codecs
from jsonpath_rw import parse, jsonpath
import time

_EXTRACTION_POLICY = 'extraction_policy'
_KEEP_EXISTING = 'keep_existing'
_REPLACE = 'replace'
_ERROR_HANDLING = 'error_handling'
_IGNORE_EXTRACTION = 'ignore_extraction'
_IGNORE_DOCUMENT = 'ignore_document'
_RAISE_ERROR = 'raise_error'
_CITY = 'city'
_CONTENT_EXTRACTION = 'content_extraction'
_RAW_CONTENT = 'raw_content'
_INPUT_PATH = 'input_path'
_READABILITY = 'readability'
_LANDMARK = 'landmark'
_TITLE = 'title'
_STRICT = 'strict'
_FIELD_NAME = 'field_name'
_CONTENT_STRICT = 'content_strict'
_CONTENT_RELAXED = 'content_relaxed'
_YES = 'yes'
_NO = 'no'
_RECALL_PRIORITY = 'recall_priority'
_INFERLINK_EXTRACTIONS = 'inferlink_extractions'
_LANDMARK_THRESHOLD = 'landmark_threshold'
_LANDMARK_RULES = 'landmark_rules'
_URL = 'url'
_RESOURCES = 'resources'
_DATA_EXTRACTION = 'data_extraction'
_FIELDS = 'fields'
_EXTRACTORS = 'extractors'
_TOKENS = 'tokens'
_SIMPLE_TOKENS = 'simple_tokens'
_TEXT = 'text'
_DICTIONARY = 'dictionary'
_NGRAMS = 'ngrams'
_JOINER = 'joiner'
_PRE_FILTER = 'pre_filter'
_POST_FILTER = 'post_filter'
_PRE_PROCESS = "pre_process"
_EXTRACT_USING_DICTIONARY = "extract_using_dictionary"
_CONFIG = "config"
_DICTIONARIES = "dictionaries"

class Core(object):

    def __init__(self, extraction_config=None, debug=False):
        if extraction_config:
            self.extraction_config = extraction_config
        self.html_title_regex = r'<title>(.*)?</title>'
        self.dictionaries_path = 'resources/dictionaries'
        self.tries = dict()
        self.global_extraction_policy = None
        self.global_error_handling = None
        # to make sure we do not parse json_paths more times than needed, we define the following 2 properties
        self.content_extraction_path = None
        self.data_extraction_path = dict()
        self.debug = debug

    """ Define all API methods """

    def process(self, doc):
        if self.extraction_config:
            if _EXTRACTION_POLICY in self.extraction_config:
                self.global_extraction_policy = self.extraction_config[_EXTRACTION_POLICY]
            if _ERROR_HANDLING in self.extraction_config:
                self.global_error_handling = self.extraction_config[_ERROR_HANDLING]

            """Handle content extraction first aka Phase 1"""
            if _CONTENT_EXTRACTION in self.extraction_config:
                if _CONTENT_EXTRACTION not in doc:
                    doc[_CONTENT_EXTRACTION] = dict()
                ce_config = self.extraction_config[_CONTENT_EXTRACTION]
                html_path = ce_config[_INPUT_PATH] if _INPUT_PATH in ce_config else None
                if not html_path:
                    raise KeyError('{} not found in extraction_config'.format(_INPUT_PATH))

                if not self.content_extraction_path:
                    start_time = time.time()
                    self.content_extraction_path = parse(html_path)
                    time_taken = time.time() - start_time
                    if self.debug:
                        print 'time taken to process parse %s' % time_taken
                start_time = time.time()
                matches = self.content_extraction_path.find(doc)
                time_taken = time.time() - start_time
                if self.debug:
                    print 'time taken to process matches %s' % time_taken
                extractors = ce_config[_EXTRACTORS]
                for index in range(len(matches)):
                    for extractor in extractors.keys():
                        if extractor == _READABILITY:
                            re_extractors = extractors[extractor]
                            if isinstance(re_extractors, dict):
                                re_extractors = [re_extractors]
                            for re_extractor in re_extractors:
                                doc[_CONTENT_EXTRACTION] = self.run_readability(doc[_CONTENT_EXTRACTION],
                                                                                matches[index].value, re_extractor)
                        elif extractor == _TITLE:
                            doc[_CONTENT_EXTRACTION] = self.run_title(doc[_CONTENT_EXTRACTION], matches[index].value,
                                                                      extractors[extractor])
                        elif extractor == _LANDMARK:
                            doc[_CONTENT_EXTRACTION] = self.run_landmark(doc[_CONTENT_EXTRACTION], matches[index].value,
                                                                         extractors[extractor], doc[_URL])
            """Phase 2: The Data Extraction"""
            if _DATA_EXTRACTION in self.extraction_config:
                de_configs = self.extraction_config[_DATA_EXTRACTION]
                if isinstance(de_configs, dict):
                    de_configs = [de_configs]

                for i in range(len(de_configs)):
                    de_config = de_configs[i]
                    input_path = de_config[_INPUT_PATH] if _INPUT_PATH in de_config else None
                    if not input_path:
                        raise KeyError('{} not found for data extraction in extraction_config'.format(_INPUT_PATH))

                    if _FIELDS in de_config:
                        if i not in self.data_extraction_path:
                            self.data_extraction_path[i] = parse(input_path)
                        matches = self.data_extraction_path[i].find(doc)
                        for match in matches:
                            # First rule of DATA Extraction club: Get tokens
                            # Get the crf tokens
                            if _TOKENS not in match.value:
                                match.value[_TOKENS] = self.extract_crftokens(match.value[_TEXT])
                            if _SIMPLE_TOKENS not in match.value:
                                match.value[_SIMPLE_TOKENS] = self.extract_tokens_from_crf(match.value[_TOKENS])
                            fields = de_config[_FIELDS]
                            for field in fields.keys():
                                if _EXTRACTORS in fields[field]:
                                    extractors = fields[field][_EXTRACTORS]
                                    for extractor in extractors.keys():
                                        try:
                                            foo = getattr(self, extractor)
                                        except Exception as e:
                                            print e
                                            foo = None
                                        if foo:
                                            if extractor == _EXTRACT_USING_DICTIONARY:
                                                print foo
                                                print foo(match.value[_TOKENS], field, extractors[extractor][_CONFIG])



        return doc

    def run_landmark(self, content_extraction, html, landmark_config, url):
        field_name = landmark_config[_FIELD_NAME] if _FIELD_NAME in landmark_config else _INFERLINK_EXTRACTIONS
        ep = self.determine_extraction_policy(landmark_config)
        extraction_rules = self.consolidate_landmark_rules()
        if _LANDMARK_THRESHOLD in landmark_config:
            pct = landmark_config[_LANDMARK_THRESHOLD]
            if not 0.0 <= pct <= 1.0:
                raise ValueError('landmark threshold should be a float between {} and {}'.format(0.0, 1.0))
        else:
            pct = 0.5
        if field_name not in content_extraction or (field_name in content_extraction and ep == _REPLACE):
            content_extraction[field_name] = dict()
            start_time = time.time()
            ifl_extractions = Core.extract_landmark(html, url, extraction_rules, pct)
            time_taken = time.time() - start_time
            if self.debug:
                print 'time taken to process landmark %s' % time_taken
            if ifl_extractions:
                for key in ifl_extractions:
                    o = dict()
                    o[key] = dict()
                    o[key]['text'] = ifl_extractions[key]
                    content_extraction[field_name].update(o)
        return content_extraction

    def consolidate_landmark_rules(self):
        rules = dict()
        if _RESOURCES in self.extraction_config:
            resources = self.extraction_config[_RESOURCES]
            if _LANDMARK in resources:
                landmark_rules_file_list = resources[_LANDMARK]
                for landmark_rules_file in landmark_rules_file_list:
                    rules.update(Core.load_json_file(landmark_rules_file))
                return rules
            else:
                raise KeyError('{}.{} not found in provided extraction config'.format(_RESOURCES, _LANDMARK))
        else:
            raise KeyError('{} not found in provided extraction config'.format(_RESOURCES))

    def get_dict_file_name_from_config(self, dict_name):
        if _RESOURCES in self.extraction_config:
            resources = self.extraction_config[_RESOURCES]
            if _DICTIONARIES in resources:
                if dict_name in resources[_DICTIONARIES]:
                    return resources[_DICTIONARIES][dict_name]
                else:
                    raise KeyError(
                        '{}.{}.{} not found in provided extraction config'.format(_RESOURCES, _DICTIONARIES, dict_name))
            else:
                raise KeyError('{}.{} not found in provided extraction config'.format(_RESOURCES, _DICTIONARIES))
        else:
            raise KeyError('{} not found in provided extraction config'.format(_RESOURCES))
    def run_title(self, content_extraction, html, title_config):
        field_name = title_config[_FIELD_NAME] if _FIELD_NAME in title_config else _TITLE
        ep = self.determine_extraction_policy(title_config)
        if field_name not in content_extraction or (field_name in content_extraction and ep == _REPLACE):
            start_time = time.time()
            content_extraction[field_name] = self.extract_title(html)
            time_taken = time.time() - start_time
            if self.debug:
                print 'time taken to process title %s' % time_taken
        return content_extraction

    def run_readability(self, content_extraction, html, re_extractor):
        recall_priority = False
        field_name = None
        if _STRICT in re_extractor:
            recall_priority = False if re_extractor[_STRICT] == _YES else True
            field_name = _CONTENT_RELAXED if recall_priority else _CONTENT_STRICT
        options = {_RECALL_PRIORITY: recall_priority}

        if _FIELD_NAME in re_extractor:
            field_name = re_extractor[_FIELD_NAME]
        ep = self.determine_extraction_policy(re_extractor)
        start_time = time.time()
        readability_text = self.extract_readability(html, options)
        time_taken = time.time() - start_time
        if self.debug:
            print 'time taken to process readability %s' % time_taken
        if readability_text:
            if field_name not in content_extraction or (field_name in content_extraction and ep == _REPLACE):
                content_extraction[field_name] = readability_text
        return content_extraction

    def determine_extraction_policy(self, config):
        ep = None
        if _EXTRACTION_POLICY in config:
            ep = config[_EXTRACTION_POLICY]
        elif self.global_extraction_policy:
            ep = self.global_extraction_policy
        if ep and ep != _KEEP_EXISTING and ep != _REPLACE:
            raise ValueError('extraction_policy can either be {} or {}'.format(_KEEP_EXISTING, _REPLACE))
        if not ep:
            ep = _REPLACE  # By default run the extraction again
        return ep

    def update_json_at_path(self, doc, match, field_name, value, parent=False):
        load_input_json = doc
        datum_object = match
        if not isinstance(datum_object, jsonpath.DatumInContext):
            raise Exception("Nothing found by the given json-path")
        path = datum_object.path
        if isinstance(path, jsonpath.Index):
            datum_object.context.value[datum_object.path.index][field_name] = value
        elif isinstance(path, jsonpath.Fields):
            datum_object.context.value[field_name] = value
        return load_input_json

    @staticmethod
    def load_json_file(file_name):
        json_x = json.load(codecs.open(file_name, 'r'))
        return json_x

    def load_trie(self, file_name):
        values = json.load(gzip.open(file_name), 'utf-8')
        trie = dictionary_extractor.populate_trie(map(lambda x: x.lower(), values))
        return trie

    def load_dictionary(self, field_name, dict_name):
        if field_name not in self.tries:
            self.tries[field_name] = self.load_trie(self.get_dict_file_name_from_config(dict_name))

    def extract_using_dictionary(self, tokens, field_name, config):
        """ Takes in tokens as input along with the dict name"""

        if _DICTIONARY not in config:
            raise KeyError('No dictionary specified for {}'.format(field_name))

        self.load_dictionary(field_name, config[_DICTIONARY])

        pre_process = None
        if _PRE_PROCESS in config and len(config[_PRE_PROCESS]) > 0:
            pre_process = self.string_to_lambda(config[_PRE_PROCESS][0])
        if not pre_process:
            pre_process = lambda x: x

        pre_filter = None
        if _PRE_FILTER in config and len(config[_PRE_FILTER]) > 0:
            pre_filter = self.string_to_lambda(config[_PRE_FILTER][0])
        if not pre_filter:
            pre_filter = lambda x: x

        post_filter = None
        if _PRE_FILTER in config and len(config[_PRE_FILTER]) > 0:
            post_filter = self.string_to_lambda(config[_PRE_FILTER][0])
        if not post_filter:
            post_filter = lambda x: isinstance(x, basestring)

        ngrams = int(config[_NGRAMS]) if _NGRAMS in config else 1

        joiner = config[_JOINER] if _JOINER in config else ' '

        return dictionary_extractor.extract_using_dictionary(tokens, pre_process=pre_process,
                                            trie=self.tries[field_name],
                                            pre_filter=pre_filter,
                                            post_filter=post_filter,
                                            ngrams=ngrams,
                                            joiner=joiner)

    @staticmethod
    def string_to_lambda(s):
        try:
            return lambda x: eval(s)
        except:
            return None

    def extract_address(self, document):
        """
        Takes text document as input.
        Note:
        1. Add keyword list as a user parameter
        2. Add documentation
        3. Add unit tests
        """

        return extract_address(document)

    def extract_readability(self, document, options={}):
        e = ReadabilityExtractor()
        return e.extract(document, options)

    def extract_title(self, html_content, options={}):
        matches = re.search(self.html_title_regex, html_content, re.IGNORECASE | re.S)
        title = None
        if matches:
            title = matches.group(1)
            title = title.replace('\r', '')
            title = title.replace('\n', '')
            title = title.replace('\t', '')
        if not title:
            title = ''
        return {'text': title}

    @staticmethod
    def extract_crftokens(text, options={}):
        t = TokenizerExtractor(recognize_linebreaks=True, create_structured_tokens=True)
        return t.extract(text)

    @staticmethod
    def extract_tokens_from_crf(crf_tokens):
        return [tk['value'] for tk in crf_tokens]

    def extract_table(self, html_doc):
        return table_extract(html_doc)

    def extract_age(self, doc):
        '''
        Args:
            doc (str): Document

        Returns:
            List of age extractions with context and value

        Examples:
            >>> tk.extract_age('32 years old')
            [{'context': {'field': 'text', 'end': 11, 'start': 0}, 'value': '32'}]
        '''

        return age_extract(doc)

    def extract_weight(self, doc):
        '''
        Args:
            doc (str): Document

        Returns:
            List of weight extractions with context and value

        Examples:
            >>> tk.extract_age('Weight 10kg')
            [{'context': {'field': 'text', 'end': 7, 'start': 11}, 'value': {'unit': 'kilogram', 'value': 10}}]
        '''

        return weight_extract(doc)

    def extract_height(self, doc):
        '''
        Args:
            doc (str): Document

        Returns:
            List of height extractions with context and value

        Examples:
            >>> tk.extract_age('Height 5'3\"')
            [{'context': {'field': 'text', 'end': 7, 'start': 12}, 'value': {'unit': 'foot/inch', 'value': '5\'3"'}}]
        '''

        return height_extract(doc)

    def extract_stock_tickers(self, doc):
        return extract_stock_tickers(doc)

    def extract_spacy(self, doc):
        return spacy_extractor.spacy_extract(doc)

    @staticmethod
    def extract_landmark(html, url, extraction_rules, threshold=0.5):
        return landmark_extraction.landmark_extractor(html, url, extraction_rules, threshold)

