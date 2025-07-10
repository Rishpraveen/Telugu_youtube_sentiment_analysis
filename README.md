# Telugu YouTube Sentiment Analysis with Llama 3.2

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.21+-yellow)

A comprehensive sentiment analysis tool for Telugu and Telugu-English code-mixed YouTube comments using Meta's Llama 3.2 model with advanced quantization and interactive dashboard capabilities.

## 🌟 Features

- **🎯 Telugu Language Support**: Native sentiment analysis for Telugu text and Telugu-English code-mixed content
- **🚀 Advanced AI Model**: Powered by Meta Llama 3.2-1B-Instruct with 4-bit quantization for efficiency
- **📺 YouTube Integration**: Direct fetching and analysis of YouTube video comments via API
- **🎮 Interactive Dashboard**: User-friendly widget-based interface with real-time processing
- **📊 Performance Metrics**: Built-in evaluation system with confusion matrices and performance charts
- **🔄 Robust Fallback**: Keyword-based sentiment analysis as backup for AI model failures
- **⚡ Memory Optimized**: Efficient model loading with BitsAndBytesConfig and Flash Attention 2 support
- **📈 Real-time Processing**: Live comment analysis with progress tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   YouTube API   │────│  Comment Fetcher │────│ Text Processor  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Results Display │────│ Sentiment Engine │────│ Llama 3.2 Model │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Fallback System  │    │ 4-bit Quantized │
                    └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Cloud Project with YouTube Data API v3 enabled
- Hugging Face account with access token

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/telugu-youtube-sentiment-analysis.git
cd telugu-youtube-sentiment-analysis
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers bitsandbytes accelerate
pip install google-api-python-client pandas scikit-learn matplotlib ipywidgets
pip install huggingface_hub
```

3. **Optional: Install Flash Attention (for better performance)**
```bash
pip install flash-attn --no-build-isolation
```

### API Setup

1. **YouTube Data API v3**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable YouTube Data API v3
   - Create API credentials (API Key)

2. **Hugging Face Token**
   - Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a new access token with read permissions

3. **Environment Setup**
   ```python
   # In Google Colab, add to secrets:
   # HF_TOKEN: your_huggingface_token
   # YOUTUBE_API_KEY: your_youtube_api_key
   
   # For local development, create .env file:
   HF_TOKEN=your_huggingface_token_here
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

## 🚀 Usage

### Interactive Dashboard (Recommended)

1. Open the Jupyter notebook: `Telugu_youtube_sentiment_analysis.ipynb`
2. Run all cells to load the model and initialize the dashboard
3. Use the interactive widget interface:
   - Paste YouTube video URL
   - Set maximum comments to analyze (10-500)
   - Configure processing options
   - Click "Start Analysis" to begin

### Programmatic Usage

```python
from sentiment_analyzer import get_telugu_sentiment, fetch_youtube_comments

# Analyze individual text
sentiment = get_telugu_sentiment("ఈ సినిమా చాలా బాగుంది!", domain="movie")
print(f"Sentiment: {sentiment}")  # Output: Positive

# Analyze YouTube video comments
video_url = "https://www.youtube.com/watch?v=your_video_id"
comments = fetch_youtube_comments(video_url, max_comments=50)

# Batch analysis
results = []
for comment in comments:
    sentiment = get_telugu_sentiment(comment['text'], domain="general")
    results.append({
        'comment': comment['text'],
        'sentiment': sentiment,
        'author': comment['author']
    })
```

### Supported Domains

- `movie`: Film reviews and entertainment content
- `product`: E-commerce and product reviews  
- `social`: Social media posts and casual conversations
- `news`: News articles and political discussions
- `general`: General purpose text analysis

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Telugu Text Accuracy | ~85% |
| Code-mixed Text Accuracy | ~80% |
| Processing Speed | ~2-3 comments/second |
| Memory Usage (4-bit) | ~1.5GB GPU RAM |

### Language Support Examples

```python
# Pure Telugu
get_telugu_sentiment("ఈ సినిమా అద్భుతం!")  # Positive

# Telugu-English Code-mixed  
get_telugu_sentiment("Movie చాలా బాగుంది bro, must watch!")  # Positive

# English with Telugu context
get_telugu_sentiment("Acting బాగాలేదు, story కూడా weak")  # Negative
```

## 🎯 Advanced Features

### Custom Domain Analysis
```python
# Movie review analysis
sentiment = get_telugu_sentiment(
    "ఈ film visual effects అద్భుతం కానీ story weak undi",
    domain="movie"
)
```

### Batch Processing with Progress Tracking
```python
from tqdm import tqdm

comments = fetch_youtube_comments(video_url, max_comments=100)
results = []

for comment in tqdm(comments, desc="Analyzing comments"):
    sentiment = get_telugu_sentiment(comment['text'])
    results.append({'text': comment['text'], 'sentiment': sentiment})
```

### Performance Evaluation
```python
# Generate confusion matrix and metrics
from evaluation import evaluate_model_performance

test_data = [
    ("ఈ సినిమా చాలా బాగుంది", "Positive"),
    ("వేస్ట్ మూవీ", "Negative"),
    # ... more test cases
]

metrics = evaluate_model_performance(test_data)
print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"F1-Score: {metrics['f1_score']:.2f}")
```

## 🔧 Configuration

### Model Configuration
```python
# Adjust model settings
MODEL_CONFIG = {
    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    "quantization": "4bit",  # Options: "4bit", "8bit", None
    "flash_attention": True,  # Enable Flash Attention 2
    "max_new_tokens": 10,
    "temperature": 0.1
}
```

### Analysis Settings
```python
# Customize analysis parameters
ANALYSIS_CONFIG = {
    "max_comments": 100,
    "truncate_long_comments": True,
    "max_comment_length": 150,
    "show_progress": True,
    "domain": "general"
}
```

## 📈 Performance Optimization

### Memory Optimization
- **4-bit Quantization**: Reduces model size by ~75%
- **Flash Attention 2**: Improves inference speed by ~40%
- **Batch Processing**: Optimal batch size of 1 for T4 GPU

### Speed Optimization
- **Model Caching**: Model loads once and stays in memory
- **Efficient Tokenization**: Optimized padding and truncation
- **Parallel Processing**: Multi-threaded comment fetching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8 jupyter

# Run tests
pytest tests/

# Format code
black src/

# Check code style
flake8 src/
```

## 🐛 Known Issues

- **Flash Attention**: May not be available on all GPU architectures
- **API Limits**: YouTube API has quota limits (10,000 requests/day)
- **Model Loading**: Initial model load takes 30-60 seconds
- **Memory**: Requires minimum 2GB GPU RAM for smooth operation

## 🔮 Roadmap

- [ ] Support for more Indian languages (Hindi, Tamil, Kannada)
- [ ] Real-time streaming comment analysis
- [ ] Emotion detection beyond sentiment
- [ ] Custom model fine-tuning scripts
- [ ] Web API deployment with FastAPI
- [ ] Mobile app integration
- [ ] Batch processing for large datasets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama 3.2 model
- **Hugging Face** for the transformers library and model hosting
- **Google** for the YouTube Data API
- **Telugu NLP Community** for language insights and testing

## 📧 Contact

- **Email**: [My Email](rishpraveen001@gmail.com)
- **LinkedIn**: [Rishpraveen](https://www.linkedin.com/in/rishpraveen?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) and [Chengalreddy Sireesha](https://www.linkedin.com/in/chengalreddy-sireesha-a61143310/)

If this project helps you, please consider giving it a ⭐ star on GitHub!
