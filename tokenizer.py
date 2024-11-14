import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from nltk.tokenize import word_tokenize
from tokenizers import SentencePieceBPETokenizer
import argparse
import nltk
from sklearn.model_selection import train_test_split

# Ensure 'punkt' resource is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Argument Parsing
# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_path", type=str, default="test_tokenizer_100")
parser.add_argument("--csv_file_path", type=str, default="output_100.csv", help="Path to the CSV file")
parser.add_argument("--column_name", type=str, default="combined_text")
parser.add_argument("--vocab_size", type=int, default=32768)
parser.add_argument("--output_name", type=str, default="test_tokenizer_100")
args = parser.parse_args()


def train_tokenizer(data_list, vocab_size=32768, model_name="test_tokenizer_100"):
    bos_tok = "<sos>"
    eos_tok = "<end_of_sen>"
    special_char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        data_list,
        vocab_size=vocab_size,
        min_frequency=5,
        special_tokens=["<pad>", "<unk>", bos_tok, eos_tok, "<user>", "<assistant>"] + special_char,
        show_progress=True,
    )

    transformer_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=bos_tok,
        eos_token=eos_tok,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        padding_side="left",
        truncation_side="right",
        additional_special_tokens=["<user>", "<assistant>"],
        clean_up_tokenization_spaces=False,
    )

    transformer_tokenizer.save_pretrained(model_name)
    return transformer_tokenizer

# Step 1: Load the specified CSV file
df = pd.read_csv(args.csv_file_path)

# Combine all columns into one, with improved performance
combined_df = pd.concat([df.astype(str)], axis=1)
combined_df['combined_text'] = combined_df.agg(' '.join, axis=1)

data_list = combined_df['combined_text'].tolist()

# Step 2: Split data for tokenizer training and fertility score calculation
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Step 3: Train Tokenizer if path is not provided
if args.tokenizer_path == "":
    tokenizer = train_tokenizer(train_data, vocab_size=args.vocab_size, model_name=args.output_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

# Step 4: Calculate Fertility Score on the test set
# Tokenize document using nltk word_tokenize and calculate word counts
df_word_count = [len(word_tokenize(text)) for text in test_data]
np_word_count = np.array(df_word_count)

# Tokenize using trained tokenizer and calculate token counts
token_list = tokenizer(test_data)["input_ids"]
np_token_count = np.array([len(tokens) for tokens in token_list])

# Calculate fertility score
f_score = np.mean(np_token_count / np_word_count)

# Step 5: Save fertility score
with open("fertility_score_100.txt", "w") as file:
    file.write(f"Fertility Score: {f_score}")
