from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the models
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

# Load the tokenizers
with open('x_tokenizer.pkl', 'rb') as f:
    x_tokenizer = pickle.load(f)

with open('y_tokenizer.pkl', 'rb') as f:
    y_tokenizer = pickle.load(f)

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

# Define the maximum length of the input text and summary
max_text_len = 100
max_summary_len = 15


def preprocess_input_text(input_text):
    """Preprocess input text by tokenizing and padding."""
    input_seq = x_tokenizer.texts_to_sequences([input_text])
    input_seq = np.pad(input_seq, [(0, 0), (0, max(0, max_text_len - len(input_seq[0])))], mode='constant')

    return input_seq

def decode_sequence(input_seq):
    """Decode the sequence and generate a summary."""
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len ):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c
    print(decoded_sentence)
    return decoded_sentence.strip()

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle the form submission and predict the summary
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_seq = preprocess_input_text(input_text)
        predicted_summary = decode_sequence(input_seq)
        # print(predicted_summary)
        return jsonify({"predicted_summary": predicted_summary})

if __name__ == '__main__':
    app.run(debug=True)
