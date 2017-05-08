import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = None
        best_score = float('inf')
        logN = np.log(len(self.X))
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                p = n**2 + 2*n*model.n_features
                score = p*logN -2*logL

                if score < best_score:
                    if(self.verbose):
                        print("In %d component,the score is %f, smaller than %f" % (n, score, best_score))
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                if(self.verbose):
                    print("Exception:"+str(e))
                
        return best_model
            
class SelectorDIC(ModelSelector):
    ''' 
    select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = None
        best_score = float('-inf')
        #logN = np.log(len(self.X))
        words_count = len(self.hwords.keys())
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                n_logL = 0
                for w in self.hwords:
                    if w != self.this_word:
                        X, lengths = self.hwords[w]
                        n_logL += model.score(X, lengths)

                score = logL - 1./(words_count-1)*(n_logL-logL)
                if score > best_score:
                    if(self.verbose):
                        print("In %d component,the score is %f, larger than %f" % (n, score, best_score))
                    best_model = model
                    best_score = score
            except Exception as e:
                if(self.verbose):
                    print(e)

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection using CV
        best_score = float('-inf')
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1 ):
            try:
                score_list = []
                if len(self.lengths) > 2:
                    split_method = KFold(n_splits=2)#(n_splits=len(self.lengths)-5 if len(self.lengths)-5>2 else 2)
                    try:
                        for train_idx,test_idx in split_method.split(self.sequences):
                            temp_X, temp_lengths = self.X, self.lengths
                            self.X, self.lengths= combine_sequences(train_idx, self.sequences)
                            model = self.base_model(n)
                            X_test, lengths_test = combine_sequences(test_idx, self.sequences)
                            logL = model.score(X_test, lengths_test)
                            score_list.append(logL)
                            self.X, self.lengths = temp_X, temp_lengths
                    except Exception as spl_error:
                        if (self.verbose):
                            print(spl_error)

                else:
                    model = self.base_model(n)
                    logL = model.score(self.X, self.lengths)
                    score_list.append(logL)

                score = np.mean(score_list)

                if score > best_score:
                    if (self.verbose):
                        print("In %d component,the score is %f, larger than %f" %(n, score, best_score))
                    best_score = score
                    best_model = model

            except Exception as e:
                if (self.verbose):
                    print("Exception:"+str(e))
                continue
        return best_model
                
