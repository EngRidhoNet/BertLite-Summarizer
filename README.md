BERT-based Sequence-to-Sequence Model for Question Generation

This project implements a sequence-to-sequence (Seq2Seq) model using TensorFlow and the BERT model for generating questions from given text inputs. The model leverages a BERT encoder and a custom Transformer decoder to perform the task of question generation, trained on a custom dataset.

Table of Contents





Project Overview



Dataset



Requirements



Installation



Usage



Model Architecture



Training



Generating Questions



Directory Structure



Contributing



License

Project Overview

This project builds a Seq2Seq model combining a pre-trained BERT model (as the encoder) with a custom Transformer decoder for generating questions from text. The model is trained on a dataset containing text-question pairs, processes input text using the BERT tokenizer, and generates questions autoregressively. The implementation includes custom loss and accuracy metrics to handle padding tokens effectively.

Dataset

The dataset used is sourced from this GitHub repository. It contains text passages and corresponding questions, which are preprocessed, tokenized, and used for training and validation.





Columns: Text (input text), Question (target question).



Preprocessing: Text normalization to remove non-printable characters, extra spaces, and standardize formatting.

Requirements

To run this project, you need the following Python packages:





tensorflow>=2.10



pandas



transformers



scikit-learn

You can install the required packages using:

pip install tensorflow pandas transformers scikit-learn

Installation





Clone this repository:

git clone https://github.com/your-username/bertlin-seq2seq-tf.git
cd bertlin-seq2seq-tf



Install the required dependencies:

pip install -r requirements.txt



Ensure you have a stable internet connection to download the dataset and pre-trained BERT model weights during execution.

Usage





Run the Training Script: Execute the main Python script to preprocess the data, train the model, and save the weights:

python main.py





The script downloads the dataset, tokenizes the data, trains the model for 3 epochs, and saves checkpoints and final weights in the ./bertlin_seq2seq_tf/ directory.



Training uses early stopping (patience=2) and saves the best model based on validation loss.



Generate Questions: After training, you can use the generate_question function to generate questions from new text inputs. Example:

sample_text = "Your sample text here."
generated_question = generate_question(sample_text)
print("Generated question:", generated_question)

Model Architecture

The model consists of:





Encoder: Pre-trained BERT (bert-base-uncased) to encode input text.



Decoder: Custom Transformer decoder with multiple layers, including:





Self-attention with causal masking to prevent attending to future tokens.



Cross-attention with encoder outputs.



Feed-forward neural network.



Positional Encoding: Added to decoder inputs to maintain token order.



Output Layer: Dense layer projecting to the vocabulary size for token prediction.

The model is compiled with a custom loss function and accuracy metric that account for padding tokens, using the Adam optimizer with a learning rate of 2e-5.

Training





Dataset Splitting: 80% training, 20% validation.



Batch Size: 8 (configurable).



Epochs: 3 (with early stopping).



Checkpoints: Saved in ./bertlin_seq2seq_tf/checkpoints for the best model based on validation loss.



Final Weights: Saved in ./bertlin_seq2seq_tf/final_weights.

Generating Questions

The generate_question function takes a text input, tokenizes it, and generates a question autoregressively up to a maximum length of 50 tokens. It uses the trained model to predict tokens one at a time, stopping when the [SEP] token is generated or the maximum length is reached.

Example:

sample_text = df_cleaned["Text"].iloc[0]
print("Sample text:", sample_text)
print("Generated question:", generate_question(sample_text))

Directory Structure

bertlin-seq2seq-tf/
├── main.py                   # Main script for training and question generation
├── bertlin_seq2seq_tf/       # Directory for checkpoints and final weights
│   ├── checkpoints/          # Model checkpoints
│   └── final_weights/       # Final trained model weights
├── requirements.txt          # Python dependencies
└── README.md                # This file

Contributing

Contributions are welcome! Please follow these steps:





Fork the repository.



Create a new branch (git checkout -b feature/your-feature).



Commit your changes (git commit -m 'Add your feature').



Push to the branch (git push origin feature/your-feature).



Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.
