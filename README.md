# 🌾 Crop Guard - Smart Agriculture System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Crop Guard is an AI-powered smart agriculture system that helps farmers detect plant diseases and weed infections using advanced computer vision and machine learning techniques. The system combines YOLO object detection, MobileNet classification, and RAG-based chatbot to provide comprehensive agricultural assistance.

## 🚀 Features

### 🩺 Disease Detection
- **Multi-stage AI Pipeline**: YOLO for leaf detection + MobileNet for disease classification
- **9 Disease Classes**: Early Blight, Late Blight, Leaf Miner, Leaf Mold, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus, and Healthy
- **Real-time Processing**: Fast inference using ONNX runtime
- **Treatment Recommendations**: AI-generated treatment suggestions for detected diseases

### 🌿 Weed Detection
- **YOLO-based Detection**: Accurate weed identification in crop images
- **Confidence Scoring**: Provides confidence levels for detections
- **Binary Classification**: Determines presence/absence of weeds

### 🤖 RAG-powered Agricultural Assistant
- **Retrieval-Augmented Generation**: Context-aware responses using agricultural knowledge base
- **ChromaDB Vector Store**: Efficient similarity search for relevant information
- **LangChain Integration**: Advanced natural language processing capabilities
- **Agricultural Expertise**: Specialized knowledge in crop management and farming practices

### 🌐 Web Interface
- **Responsive Design**: Modern, user-friendly web interface
- **Real-time Results**: Instant image analysis and results display
- **Multi-page Application**: Separate interfaces for each functionality
- **Session Management**: Track user interactions and maintain context

## 🏗️ Architecture

```
Crop Guard System
├── Frontend (HTML/CSS/JS)
│   ├── Disease Detection Interface
│   ├── Weed Detection Interface
│   └── Chat Assistant Interface
├── Backend (FastAPI)
│   ├── Disease Detection API
│   ├── Weed Detection API
│   └── RAG Chatbot API
├── AI Models
│   ├── YOLO (Object Detection)
│   ├── MobileNet (Disease Classification)
│   └── Weed Detection YOLO
└── Knowledge Base
    ├── ChromaDB Vector Store
    └── Agricultural Documents
```

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- At least 4GB RAM
- 2GB free disk space

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Verify model files**
   Ensure the following ONNX models are present in `app/models/`:
   - `yolo.onnx` (YOLO object detection)
   - `class.onnx` (MobileNet disease classification)
   - `weed.onnx` (Weed detection)

## 🚀 Usage

### Starting the Application

1. **Run the FastAPI server**
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the web interface**
   Open your browser and navigate to: `http://localhost:8000`

### API Endpoints

#### Disease Detection
```http
POST /api/disease-detection/
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG, PNG)
- X-Session-ID: Optional session identifier

Response:
{
  "session_id": "uuid",
  "image_url": "base64_encoded_image",
  "diseases": ["disease_name"],
  "treatment": "treatment_recommendations"
}
```

#### Weed Detection
```http
POST /api/weed-detection/
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG, PNG)
- X-Session-ID: Optional session identifier

Response:
{
  "session_id": "uuid",
  "image_url": "base64_encoded_image",
  "has_weed": true/false,
  "confidence": 0.95
}
```

#### Chat Assistant
```http
POST /api/chatbot
Content-Type: application/json

Body:
{
  "user_input": "How do I treat early blight?"
}

Response:
{
  "response": "AI-generated response with treatment advice"
}
```

## 📁 Project Structure

```
AI/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── api/                      # API route handlers
│   │   ├── __init__.py
│   │   ├── disease_detection.py  # Disease detection endpoints
│   │   ├── weed_detection.py     # Weed detection endpoints
│   │   └── rag_chatbot.py        # Chatbot endpoints
│   ├── models/                   # ONNX model files
│   │   ├── yolo.onnx            # YOLO object detection model
│   │   ├── class.onnx           # MobileNet classification model
│   │   └── weed.onnx            # Weed detection model
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── disease_detection_service.py  # Disease detection logic
│   │   ├── weed_service.py       # Weed detection logic
│   │   ├── doctor_service.py     # Treatment recommendation service
│   │   ├── chat1.py             # Embedding functions
│   │   └── chat2.py             # RAG chain setup
│   ├── static/                   # Static web assets
│   │   └── style.css            # Application styles
│   ├── templates/                # HTML templates
│   │   ├── index.html           # Home page
│   │   ├── disease_detection.html
│   │   ├── weed_detection.html
│   │   └── rag_chatbot.html
│   └── rag_data/                # RAG knowledge base
│       └── Agriculture_db/       # ChromaDB vector store
├── notebooks/                    # Jupyter notebooks for model training
│   ├── Create Classification Data.ipynb
│   ├── MobileNet_Disease_Classification.ipynb
│   ├── Yolo_Disease_Detection.ipynb
│   └── Yolo_Weed_Detection.ipynb
├── data/                        # Sample data and utilities
│   ├── helper_function.py
│   ├── main.py
│   ├── test.jpg
│   └── test.xml
├── Agriculture_db/              # ChromaDB persistence directory
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 🧠 AI Models

### Disease Detection Pipeline

1. **YOLO Object Detection**
   - Detects and localizes plant leaves in images
   - Crops relevant regions for classification
   - Model: Custom trained YOLO v8

2. **MobileNet Classification**
   - Classifies cropped leaf regions into disease categories
   - 9 classes: Early Blight, Late Blight, Leaf Miner, Leaf Mold, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus, Healthy
   - Model: MobileNet v2 fine-tuned on plant disease dataset

### Weed Detection
- **YOLO-based Detection**: Custom trained model for weed identification
- **Binary Classification**: Determines presence/absence of weeds
- **Confidence Scoring**: Provides detection confidence levels

### RAG Chatbot
- **Embedding Model**: HuggingFace sentence transformers
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM**: Groq API for response generation
- **Knowledge Base**: Curated agricultural documents and FAQs

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

### Model Configuration
Models are automatically loaded from the `app/models/` directory. Ensure all ONNX files are present:

- `yolo.onnx`: YOLO object detection model
- `class.onnx`: MobileNet disease classification model
- `weed.onnx`: Weed detection model

## 📊 Supported Disease Classes

| Class ID | Disease Name | Description |
|----------|--------------|-------------|
| 0 | Early Blight | Fungal disease causing dark spots on leaves |
| 1 | Healthy | No disease detected |
| 2 | Late Blight | Serious fungal disease affecting potatoes/tomatoes |
| 3 | Leaf Miner | Insect larvae creating tunnels in leaves |
| 4 | Leaf Mold | Fungal disease in humid conditions |
| 5 | Mosaic Virus | Viral infection causing mottled patterns |
| 6 | Septoria | Fungal disease with small dark spots |
| 7 | Spider Mites | Tiny pests causing stippling damage |
| 8 | Yellow Leaf Curl Virus | Viral disease causing leaf curling |

## 🧪 Development

### Running in Development Mode

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access the application
open http://localhost:8000
```

### Model Training

The `notebooks/` directory contains Jupyter notebooks for training the AI models:

1. **Data Preparation**: `Create Classification Data.ipynb`
2. **Disease Classification**: `MobileNet_Disease_Classification.ipynb`
3. **Disease Detection**: `Yolo_Disease_Detection.ipynb`
4. **Weed Detection**: `Yolo_Weed_Detection.ipynb`

### Adding New Disease Classes

1. Update the `custom_class_names` dictionary in `disease_detection_service.py`
2. Retrain the MobileNet model with new data
3. Export the updated model to ONNX format
4. Update the treatment recommendations in `doctor_service.py`

## 🔍 Testing

### Manual Testing

1. **Disease Detection**
   ```bash
   curl -X POST "http://localhost:8000/api/disease-detection/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@test_image.jpg"
   ```

2. **Weed Detection**
   ```bash
   curl -X POST "http://localhost:8000/api/weed-detection/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@test_image.jpg"
   ```

3. **Chat Assistant**
   ```bash
   curl -X POST "http://localhost:8000/api/chatbot" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{"user_input": "How do I treat early blight?"}'
   ```

### Web Interface Testing

1. Start the application: `python main.py`
2. Navigate to `http://localhost:8000`
3. Test each feature through the web interface:
   - Upload images for disease detection
   - Upload images for weed detection
   - Chat with the agricultural assistant

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Cannot load ONNX model
   Solution: Ensure all .onnx files are present in app/models/
   ```

2. **ChromaDB Errors**
   ```
   Error: Cannot connect to ChromaDB
   Solution: Check if Agriculture_db directory exists and has proper permissions
   ```

3. **API Key Errors**
   ```
   Error: Invalid Groq API key
   Solution: Set GROQ_API_KEY in .env file with valid API key
   ```

4. **Memory Issues**
   ```
   Error: Out of memory during inference
   Solution: Reduce image size or use CPU-only inference
   ```

### Performance Optimization

- **GPU Acceleration**: Install CUDA-compatible ONNX Runtime for faster inference
- **Model Optimization**: Use quantized models for reduced memory usage
- **Caching**: Implement Redis for caching frequent requests
- **Load Balancing**: Use multiple worker processes for high-traffic scenarios

## 📈 Performance Metrics

### Model Accuracy
- **Disease Classification**: ~92% accuracy on test dataset
- **Weed Detection**: ~88% accuracy with 0.85 confidence threshold
- **Response Time**: <2 seconds average for image processing

### System Requirements
- **Minimum**: 4GB RAM, 2GB storage, Python 3.8+
- **Recommended**: 8GB RAM, 4GB storage, GPU support
- **Production**: 16GB RAM, SSD storage, load balancer

## 🤝 Contributing

We welcome contributions to improve Crop Guard! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

### Areas for Contribution

- **New Disease Classes**: Add support for additional plant diseases
- **Model Improvements**: Enhance accuracy and performance
- **UI/UX**: Improve web interface design and usability
- **Documentation**: Improve guides and tutorials
- **Testing**: Add comprehensive test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**CropPilot Team** - *Graduation Project*

- Disease Detection & Classification
- Weed Detection System
- RAG-based Agricultural Assistant
- Web Interface Development

## 🙏 Acknowledgments

- **TensorFlow/Keras** for deep learning framework
- **YOLO** for object detection architecture
- **MobileNet** for efficient classification
- **LangChain** for RAG implementation
- **ChromaDB** for vector database
- **FastAPI** for web framework
- **Agricultural Research Community** for domain knowledge

## 📞 Support

For questions, issues, or suggestions:

1. **GitHub Issues**: Create an issue for bugs or feature requests
2. **Documentation**: Check this README and code comments
3. **Community**: Join discussions in the repository

---

**Made with ❤️ by the CropPilot Team**

*Empowering farmers with AI-driven agricultural solutions*