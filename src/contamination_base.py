"""
Contamination module. 

As part of our study, we'd like to attribute each prediction to
specific documents in the input. To this end, we will rely on the
data contamination techniques proposed in prior work.

For now, we will focus on the following techniques:

1. Brown et al 2020, used up until 2020, 
  * significance: original gpt-3 paper
  * Technique that focuses on the n-gram overlap between training data and test examples.
  * If a test example contains a 8-gram or 13-gram overlap with the training data, 
  then it is considered to be contaminated.
  * The authors set the overlap threshold to be 13 based on the 5th percentile of example length (in words).
  * See also the discussion in Appendix A.6 of Llama-2 paper (Touvron et al 2023)
  
2. Chowdchery et al 2022, https://arxiv.org/pdf/2204.02311.pdf
  * significance: palm2 paper
  * Instead of a single 8-gram, it includes a threshold to consider it a contamination
  * Given a test example, if 70% of its 8-grams are present at least once in the training data, 
  then it is considered to be contaminated.

3. Touvron et al (2023), https://arxiv.org/pdf/2307.09288.pdf
  * significance: llama-2 paper, first bottom-up data contamination perspective 
  (focused on tokens, as opposed to substring overlap), similar to llama3.
  * token is contaminated if it appears in any token n-gram longer than 10 tokens
  in both the evaluation example and the training set.
  * the contamination of the evaluation example is determined as the ratio \tau of
  contaminated tokens in the eval example.
  * they add a skipgram budget of 4 tokens, i.e., they allow the matched n-grams
  in the training and eval set to differ by at most 4 tokens.
  * do not allow trailing mismatches or mismatches in the first 10 tokens.
  * they consider dirty samples those with \tau >= 0.8 token contamination
  * note: to address the case where the sentence is fragmented, they test
  the contamination for varying minimum matching length {1-, 20, 30, 40, 50}.
  
4. gpt-4 technical report, 2024
   * significance: gpt-4 paper
   * remove whitespaces and symbols, but consider the numbers of the original strings.
   * for each evaluation sample, randomly sample 3 substrings of 50 characters (or entire string if shorter)
   from the evaluation example. If at least one of these substrings is present
   in the pretraining data, it is considered to be contaminated.

5. DOLMA, Paloma papers
   * significance: first open source pretraining dataset, used to train OLMo
   * the authors use a fuzzy algorithm based on bloom filters to obtain a
   probabilistic estimate of the contamination of a given example.
   
Note: many of these examples rely on suffix array to make the search for matches more efficient.
These techniques do not seem to be directly available out of the box. We will need to implement it
ourselves.

Other resources concerning data contamination:
https://www.holisticai.com/blog/overview-of-data-contamination
https://github.com/lyy1994/awesome-data-contamination
"""
import editdistance
import logging
import random
import nltk
import string


from nltk.util import ngrams
from nltk.lm import NgramCounter
from nltk.corpus import stopwords
try:
    STOPWORDS = stopwords.words('english')
except:
    nltk.download('stopwords')
    STOPWORDS = stopwords.words('english')
from typing import Dict, List, Tuple, Union
from utils import create_object_from_class_string, generate_uuid
from utils_mixins import CacheMixin


def preprocess_text(text, use_lowercase, remove_stopwords, remove_punctuation) -> List[str]:
    """Preprocess the text into sequence of sentences."""
    sep = " "
    sentences: List[str] = nltk.sent_tokenize(text)
    
    if use_lowercase:
        sentences = [s.lower() for s in sentences]
    
    if remove_punctuation: # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        sentences: List[str] = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    
    if remove_stopwords:    
        tokenized_sentences: List[List[str]] = [
            [word.strip() for word in nltk.word_tokenize(s) if word not in STOPWORDS]
            for s in sentences
        ]
        return [sep.join(words) for words in tokenized_sentences]
    
    return [sep.join(s.split()) for s in sentences]

class ContaminationBase(CacheMixin):
    def __init__(self, use_lowercase=True, remove_stopwords=True, remove_punctuation=True, **kwargs):
        super().__init__(**kwargs)  
        self.preprocess_kwargs = {
            "use_lowercase": use_lowercase,
            "remove_stopwords": remove_stopwords,
            "remove_punctuation": remove_punctuation
        }        
    
    def get_uuid(self, text: str) -> str:
        doc_params = {"text": text.strip()}
        doc_params.update(**self.preprocess_kwargs)
        return generate_uuid(doc_params)
        
    def index_documents(self, documents: Dict[str, str]):
        """Creates an index of the documents or loads it into memory"""
        self.prediction_cache.setdefault("documents", {})
        self.prediction_cache.setdefault("predictions", {})
        
        for uuid, text in documents.items():
            preproc_text = self.preprocess_text(text)

            if self.prediction_cache["documents"].get(uuid, None) is None:
                if self.cache_only:
                    raise ValueError
                self.prediction_cache["documents"][uuid] = preproc_text
        self.save_prediction_cache()
                
    def preprocess_text(self, text: str):
        raise NotImplementedError
    
    def check_match(self, insight: str, doc_uuids: List[str]) -> List[bool]:
        raise NotImplementedError
    
    @staticmethod
    def from_kwargs(config_filepath=None, **kwargs) -> "ContaminationBase":
        classname = kwargs.pop("classname", None)
        module, _, name = classname.rpartition(".")
        obj = create_object_from_class_string(module, name, kwargs)
        kwargs["classname"] = classname
        return obj
    
        
class SubstringMatch(ContaminationBase):
    """Implements the gpt-4o contamination technique."""
    
    def __init__(self, substr_len: int, seed=879123, **kwargs):
        super().__init__(**kwargs)
        self.default_cache_file = \
            f'document_provenance/substring_match___substr_len-{substr_len}__cache.pkl'  
        
        assert substr_len > 0, "Substring length must be greater than 0."
        self.substr_len = substr_len
        self.seed = seed
        self.rand = random.Random(seed)
        
    def preprocess_text(self, text: str) -> str:
        preproc_sents: List[str] = preprocess_text(text, **self.preprocess_kwargs)
        return " ".join(preproc_sents)
    
    def match_condition(self, insight, text):
        return insight in text
    
    def check_match(self, insight: str, doc_uuid: str) -> bool:
        def get_match(preproc_insight: str) -> bool:            
            # Get all beginning of words to check for matches
            sampling_positions = [0] + [i+1 for i, c in enumerate(preproc_insight[:-self.substr_len]) if c == " "]

            for sampling_pos in sampling_positions:
                substr = preproc_insight[sampling_pos:sampling_pos+self.substr_len]
                if self.match_condition(substr, preproc_doc):
                    return preproc_doc.index(substr)
            return -1
        
        preproc_doc = self.prediction_cache["documents"][doc_uuid]
        preproc_insight = self.preprocess_text(insight)
        hashed = self.get_uuid(f"{self.substr_len} {preproc_insight} {preproc_doc}")
        if self.prediction_cache["predictions"].get(hashed) is not None:
            logging.info('Restoring hashed instead of predicting with model.') # change it to debug
            response = self.prediction_cache["predictions"][hashed]
            assert response == get_match(preproc_insight)
        else:
            if self.cache_only:
                raise ValueError
            response = get_match(preproc_insight)
            # ^We'll keep track of the position in the insight where it occurs
            # This is more helpful and allow us to apply different preprocessing
            # afterwords without having to recompute the matches. For example,
            # if we want to re-use more fine-grained information about the location
            # of these matches.
            self.prediction_cache["predictions"][hashed] = response
        
        # note: -1 indicates no match
        return response != -1 


class LevensteinSubstringMatch(SubstringMatch):
    def __init__(self, fraction: float=0.2, chunk_mode: str | None = None, chunk_kwargs: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.default_cache_file = f'document_provenance/levenstein_substring_match__cache.pkl'
        self.fraction = fraction
        self.chunk_kwargs = chunk_kwargs
        self.chunk_mode = chunk_mode
        
    def get_chunks(self, text, def_chunk_size=50):
        if self.chunk_mode is None:
            chunks = [text]
    
        elif self.chunk_mode == "sliding_window":
            chunk_size = self.chunk_kwargs.get("chunk_size", def_chunk_size)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
        elif self.chunk_mode == "random":
            chunk_size = self.chunk_kwargs.get("chunk_size", def_chunk_size)
            num_chunks = self.chunk_kwargs.get("num_chunks", ((len(text) - chunk_size) // chunk_size) + 10) 

            chunks = []
            for _ in num_chunks:
                start = self.rand.randint(0, len(text) - chunk_size)
                chunks.append(text[start:start+chunk_size])
        else:
            raise ValueError(f"Unknown chunk mode: {self.chunk_mode}")
        return chunks

    def match_condition(self, insight: str, text: str) -> bool:
        chunks = self.get_chunks(text, chunk_size=len(insight)) # not super efficient but its ok for now
        
        # If any of the chunks exhibits a smaller levenstein distance, than the
        # specified threshold, then it is a match
        for chunk in chunks:
            dist = editdistance.eval(insight, def_chunk_size=chunk) 
            # ^Note: supports both list of words and strings
            if dist <= self.fraction * len(insight):
                return True


class NgramOverlapMatch(ContaminationBase):
    def __init__(self, ngram: Union[Tuple[int, int], int], threshold: float=0.7, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        if isinstance(ngram, int):
            ngram = (ngram, ngram)
        assert ngram[0] > 0, "Ngram size must be greater than 0."
        assert 0 < threshold <= 1
        self.ngram = ngram
        self.threshold = threshold
        self.default_cache_file = f'document_provenance/{ngram[0]}-{ngram[1]}__ngram_overlap__match__cache.pkl' 
        
    