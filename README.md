# Text Summarizer

A deep learning-based text summarization application built with Flask and Keras. This project uses a sequence-to-sequence model with encoder-decoder architecture to automatically generate concise summaries from longer text inputs.

## Features

- **Automatic Text Summarization**: Generate summaries from input text using a trained LSTM-based seq2seq model
- **Web Interface**: User-friendly web application built with Flask
- **RESTful API**: JSON-based API endpoint for easy integration
- **Pre-trained Models**: Includes encoder and decoder models ready for inference

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Deep Learning**: Keras/TensorFlow
- **Model Architecture**: Sequence-to-sequence with encoder-decoder
- **Tokenization**: Keras Tokenizer with pickle serialization
- **Frontend**: HTML/CSS (rendered via Flask templates)

## Project Structure

```
.
├── app.py                  # Main Flask application
├── encoder_model.h5        # Pre-trained encoder model
├── decoder_model.h5        # Pre-trained decoder model
├── x_tokenizer.pkl         # Input text tokenizer
├── y_tokenizer.pkl         # Summary tokenizer
├── templates/
│   └── index.html         # Web interface template
└── README.md              # Project documentation
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd text-summarizer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install flask keras tensorflow numpy
   ```

4. **Ensure model files are present**
   
   Make sure the following files are in the project directory:
   - `encoder_model.h5`
   - `decoder_model.h5`
   - `x_tokenizer.pkl`
   - `y_tokenizer.pkl`

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Generate summaries**
   - Enter your text in the input field
   - Click the submit button
   - View the generated summary

### API Usage

You can also interact with the application programmatically using the API endpoint.

**Endpoint**: `POST /predict`

**Request Format**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "input_text=Your long text here that needs to be summarized..."
```

**Response Format**:
```json
{
  "predicted_summary": "Generated summary text"
}
```

**Python Example**:
```python
import requests

url = "http://localhost:5000/predict"
data = {"input_text": "Your long text here..."}

response = requests.post(url, data=data)
summary = response.json()["predicted_summary"]
print(summary)
```

## Model Details

### Architecture

- **Encoder**: LSTM-based encoder that processes input text sequences
- **Decoder**: LSTM-based decoder that generates summary sequences
- **Maximum Input Length**: 100 tokens
- **Maximum Summary Length**: 15 tokens

### Special Tokens

- `sostok`: Start of sequence token
- `eostok`: End of sequence token

### Preprocessing

Input text is:
1. Tokenized using the pre-trained tokenizer
2. Converted to sequences
3. Padded to maximum length (100 tokens)

## Configuration

Key parameters in `app.py`:

```python
max_text_len = 100      # Maximum length of input text
max_summary_len = 15    # Maximum length of generated summary
```

Adjust these values based on your model training configuration.

## Training the Model

This README focuses on using the pre-trained models. If you want to train your own model:

1. Prepare your dataset with text-summary pairs
2. Tokenize and preprocess the data
3. Build encoder-decoder architecture
4. Train the model
5. Save the trained models and tokenizers
6. Replace the model files in this project

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'keras'`
- **Solution**: Install Keras and TensorFlow: `pip install keras tensorflow`

**Issue**: Model files not found
- **Solution**: Ensure all `.h5` and `.pkl` files are in the project root directory

**Issue**: Out of memory errors
- **Solution**: Reduce batch size or input text length, or run on a machine with more RAM

**Issue**: Poor summary quality
- **Solution**: The model quality depends on training data. Consider retraining with more diverse data

## Performance Considerations

- Model loading happens once at startup for better performance
- Inference time depends on input text length
- For production deployment, consider using a production WSGI server like Gunicorn

## Future Enhancements

- [ ] Add support for batch summarization
- [ ] Implement attention mechanism for better summaries
- [ ] Add support for multiple languages
- [ ] Create Docker container for easy deployment
- [ ] Add model performance metrics
- [ ] Implement caching for frequently summarized texts
- [ ] Add user authentication
- [ ] Support for document upload (PDF, DOCX)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

**Note**: This is an educational project demonstrating text summarization using deep learning. For production use, consider implementing additional error handling, input validation, and security measures.
