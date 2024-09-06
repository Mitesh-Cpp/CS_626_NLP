import streamlit as st
import joblib


from collections import defaultdict

class HMMTagger:
    def __init__(self, train_sentences, words, tags):
        self.train_sentences = train_sentences
        self.words = words
        self.tags = tags
        self.tagset = set(tags)
        self.wordset = set(words)
        self.initial_probs = {}
        self.transition_probs = {}
        self.emission_probs = {}
        self.freq = {}

    def train(self):
        self._compute_initial_probs()
        self._compute_transition_probs()
        self._compute_emission_probs()

    def _compute_initial_probs(self):
        self.freq = defaultdict(int)
        for sentence in self.train_sentences:
            for word, tag in sentence:
                w = word.lower().rstrip("'s") if word.endswith("'s") else word.lower()
                self.freq[w] += 1

        sorted_words = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)

        total_frequency = sum(self.freq.values())
        cutoff_frequency = 0.8 * total_frequency  # 80% of total frequency

        valid_words = set()
        current_sum = 0

        for word, freq in sorted_words:
            if current_sum + freq <= cutoff_frequency:
                valid_words.add(word)
                current_sum += freq
            else:
                break

        # Change words not in the top 80% to "unknown"
        unknown_count = 0
        new_freq = defaultdict(int)

        for word, freq in self.freq.items():
            if word in valid_words:
                new_freq[word] = freq
            else:
                unknown_count += freq

        new_freq["unknown"] = unknown_count

        self.freq = new_freq

        tag_counts = defaultdict(int)
        for t in self.tagset:
            tag_counts[t] = 1
        for sentence in self.train_sentences:
            if len(sentence) > 0:  # If the sentence is not empty
                first_tag = sentence[0][1]
                tag_counts[first_tag] += 1

        total_sentences = len(self.train_sentences)

        # Calculate initial probabilities
        for tag in tag_counts:
            self.initial_probs[tag] = tag_counts[tag] / (total_sentences + len(self.tagset))

    def _compute_transition_probs(self):
        bigram_counts = defaultdict(int)
        tag_counts = defaultdict(int)

        # Initialize bigram counts with smoothing (Laplace smoothing)
        for tag1 in self.tagset:
            for tag2 in self.tagset:
                bigram_counts[(tag1, tag2)] = 1  # Add 1 to each bigram count for smoothing

        # Count tag occurrences and bigrams in sentences
        for sentence in self.train_sentences:
            for i in range(len(sentence)-1):
                current_tag = sentence[i][1]
                next_tag = sentence[i+1][1]
                tag_counts[current_tag] += 1
                bigram_counts[(current_tag, next_tag)] += 1

        # Calculate transition probabilities
        for tag1 in self.tagset:
            for tag2 in self.tagset:
                self.transition_probs[(tag1, tag2)] = (bigram_counts[(tag1, tag2)]) / (len(self.tagset) + tag_counts[tag1])

    def _compute_emission_probs(self):
        word_tag_counts = defaultdict(int)
        tag_counts = defaultdict(int)

        # Initialize word-tag counts with 1 for smoothing (Laplace smoothing)
        for t in self.tagset:
            for w in self.freq:  # All words present in the frequency dictionary, including "unknown"
                word_tag_counts[(w, t)] = 1

        # Calculate the frequency of (word, tag) pairs and individual tags
        for w, t in zip(self.words, self.tags):
            if w in self.freq:
                word_tag_counts[(w, t)] += 1
            else:
                word_tag_counts[("unknown", t)] += 1

            tag_counts[t] += 1

        # Calculate emission probabilities with smoothing
        for (w, t) in word_tag_counts:
            self.emission_probs[(t, w)] = word_tag_counts[(w, t)] / (tag_counts[t] + len(self.freq))


    def _viterbi(self, phrase):
        phrase = [word.lower().rstrip("'s") if word.endswith("'s") else word.lower() for word in phrase]  # Basic Preprocessing
        T = len(phrase)  # Length of the phrase (number of words)
        N = len(self.tagset)  # Number of possible tags
        tags_list = list(self.tagset)

        # Initialize the Viterbi table (T x N) with zeros
        V = [[0.0] * N for _ in range(T)]
        backpointer = [[0] * N for _ in range(T)]

        # Initialize the first row of the Viterbi table
        for j in range(N):
            tag = tags_list[j]
            word = phrase[0]

            # Use emission probability of "unknown" if the word is not in self.freq
            if word in self.freq:
                emission_prob = self.emission_probs[(tag, word)]
            else:
                emission_prob = self.emission_probs[(tag, "unknown")]

            V[0][j] = self.initial_probs[tag] * emission_prob
            backpointer[0][j] = 0  # Start state has no previous state


        # Fill in the rest of the Viterbi table
        for i in range(1, T):
            for j in range(N):
                max_prob = -float('inf')
                max_prev = 0
                for k in range(N):
                    word = phrase[i]

                    # Use emission probability of "unknown" if the word is not in self.freq
                    if word in self.freq:
                        emission_prob = self.emission_probs[(tags_list[j], word)]
                    else:
                        emission_prob = self.emission_probs[(tags_list[j], "unknown")]

                    prob = V[i - 1][k] * self.transition_probs[(tags_list[k], tags_list[j])] * emission_prob

                    if prob > max_prob:
                        max_prob = prob
                        max_prev = k

                V[i][j] = max_prob
                backpointer[i][j] = max_prev

        # Find the most likely final state
        final_state = max(range(N), key=lambda j: V[T - 1][j])
        max_prob = V[T - 1][final_state]

        # Backtrack to find the most likely sequence of tags
        result_tags = [tags_list[final_state]]
        for i in range(T - 2, -1, -1):
            final_state = backpointer[i + 1][final_state]
            result_tags.insert(0, tags_list[final_state])

        return result_tags
    
    
def load_model():
    model = joblib.load('./hmm_tagger_model.pkl')
    return model

# Define tag list
tag_list = ['DET', 'PRT', 'ADV', 'X', 'CONJ', 'ADJ', 'ADP', 'PRON', 'NOUN', '.', 'NUM', 'VERB']

# Streamlit interface
st.title("POS Tagging with HMM")

# Input text
sentence = st.text_area("Enter a sentence:")

if st.button("Predict Tags"):
    if sentence:
        # Tokenize the sentence into words
        words_in_sentence = sentence.lower().split()

        # Load model and make predictions
        model = load_model()
        predicted_tags = model._viterbi(words_in_sentence)

        # Display results
        st.write("Words:", words_in_sentence)
        st.write("Predicted Tags:", predicted_tags)
