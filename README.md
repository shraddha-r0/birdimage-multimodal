# 🐦 Bird Image Search with Multimodal AI

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0+-red)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-blueviolet)](https://gradio.app/)

A multimodal AI application that enables semantic search through bird images using natural language queries or example images. Built with state-of-the-art deep learning models, this project demonstrates the power of multimodal AI in wildlife conservation and biodiversity research.


## ✨ Features

- **Natural Language Search**: Find birds using descriptive text queries
- **Visual Similarity Search**: Upload an image to find visually similar birds
- **Species Filtering**: Narrow down results by bird species
- **Responsive Web Interface**: User-friendly Gradio interface
- **Efficient Vector Database**: Powered by LanceDB for fast similarity search

## 🛠️ Technologies Used

- **Core AI**:
  - [CLIP](https://openai.com/research/clip) (Contrastive Language-Image Pretraining) by OpenAI
  - PyTorch for deep learning

- **Vector Database**:
  - [LanceDB](https://lancedb.com/) for efficient vector similarity search

- **Web Interface**:
  - [Gradio](https://gradio.app/) for building interactive web demos
  - Custom CSS for responsive design

- **Data Processing**:
  - Hugging Face Datasets
  - NumPy for numerical operations
  - PIL/Pillow for image processing

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/birdimage-multimodal.git
   cd birdimage-multimodal
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Download and prepare the dataset**:
   ```bash
   python scripts/build_dataset.py
   ```

2. **Build the vector index**:
   ```bash
   python scripts/build_index.py
   ```

3. **Launch the web application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your web browser and navigate to `http://localhost:7860`

## 📚 Project Structure

```
birdimage-multimodal/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── scripts/
│   ├── build_dataset.py   # Script to download and prepare data
│   └── build_index.py     # Script to create the vector index
├── data/                  # Bird images (not versioned)
└── db/                    # LanceDB database (not versioned)
```

## 🌟 Example Queries

- "red bird with black wings"
- "small yellow bird"
- "blue and white water bird"
- "bird in flight"
- "crow or black bird"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [CLIP](https://openai.com/research/clip) by OpenAI for the vision-language model
- [LanceDB](https://lancedb.com/) for the open-source vector database
- [Gradio](https://gradio.app/) for the easy-to-use web interface framework
- [Hugging Face](https://huggingface.co/) for the datasets and model hub

---

<div align="center">
  Made with ❤️ by Shraddha Ramesh | [LinkedIn](https://www.linkedin.com/in/shraddha-r0/) | [GitHub](https://github.com/shraddha-r0)
</div>