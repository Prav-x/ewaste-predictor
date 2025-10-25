# ♻️ E-Waste AI Predictor & Recycling Guide

An intelligent AI-powered application that classifies electronic waste and provides detailed recycling suggestions to promote sustainable waste management and resource recovery.

## 🎯 Features

- **AI-Powered Classification**: Upload images to automatically classify e-waste types
- **Recycling Guidance**: Get detailed recycling suggestions for each e-waste category
- **Interactive Web Interface**: User-friendly Streamlit web application
- **Model Training**: Train and evaluate the model with your own dataset
- **Visualization**: Interactive charts showing prediction confidence
- **Environmental Impact**: Learn about the environmental benefits of recycling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # If you have git installed
   git clone <repository-url>
   cd Model-Predictor
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

## 📁 Project Structure

```
Model Predictor/
├── app.py                    # Streamlit web application
├── ewaste_predictor.py      # Core AI model and prediction logic
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── Setup_variables.py      # Configuration variables
├── Test.py                 # Dataset testing script
└── ewaste_dataset/         # Training dataset
    ├── train/              # Training images (240 per class)
    ├── val/                # Validation images (30 per class)
    └── test/               # Test images (30 per class)
```

## 🔬 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for e-waste classification
- **Data Augmentation**: Rotation, zoom, shift, and flip transformations
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers to prevent overfitting

## 📊 Supported E-Waste Categories

1. **Battery** - Lithium-ion and alkaline batteries
2. **Keyboard** - Computer keyboards and input devices
3. **Microwave** - Microwave ovens and kitchen appliances
4. **Mobile** - Mobile phones and smartphones
5. **Mouse** - Computer mice and pointing devices
6. **PCB** - Printed Circuit Boards
7. **Player** - Media players and audio devices
8. **Printer** - Printers and printing devices
9. **Television** - Televisions and display devices
10. **Washing Machine** - Washing machines and large appliances

## 🎮 How to Use

### 1. Predict E-Waste Type
- Go to the "🔍 Predict E-Waste" tab
- Upload an image of electronic waste
- Get instant classification and recycling suggestions
- View confidence scores for all categories

### 2. Train the Model
- Go to the "📊 Model Training" tab
- Configure training parameters (epochs, batch size, etc.)
- Click "Start Training" to train the model
- View training progress and results

### 3. Explore Recycling Guide
- Go to the "📚 Recycling Guide" tab
- Browse comprehensive recycling information for all categories
- Learn about recycling processes and environmental impact

### 4. Learn More
- Go to the "ℹ️ About" tab
- Read about the technology stack and features
- Understand the environmental impact of recycling

## 🌱 Environmental Impact

By properly classifying and recycling e-waste, this application helps:

- **Recover Valuable Materials**: Gold, silver, copper, rare earth elements
- **Prevent Contamination**: Stop toxic substances from entering soil and water
- **Reduce Mining**: Decrease the need for extracting new materials
- **Create Jobs**: Support the growing recycling industry
- **Reduce Landfill Waste**: Keep electronic waste out of landfills

## 🔧 Technical Details

### Dependencies
- **TensorFlow**: Deep learning framework
- **Streamlit**: Web application framework
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations

### Model Performance
- **Architecture**: CNN with transfer learning
- **Input Size**: 224x224 pixels
- **Classes**: 10 e-waste categories
- **Augmentation**: Multiple data augmentation techniques
- **Optimization**: Adam optimizer with learning rate scheduling

## 🚨 Important Notes

1. **First Run**: If no pre-trained model exists, the app will use an untrained model. Train the model first for accurate predictions.

2. **Dataset**: The application expects the dataset to be organized in the following structure:
   ```
   ewaste_dataset/
   ├── train/
   │   ├── Battery/
   │   ├── Keyboard/
   │   └── ...
   ├── val/
   │   ├── Battery/
   │   ├── Keyboard/
   │   └── ...
   └── test/
       ├── Battery/
       ├── Keyboard/
       └── ...
   ```

3. **Image Formats**: Supported formats are JPG, JPEG, and PNG.

4. **Training Time**: Model training may take 30 minutes to several hours depending on your hardware.

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed correctly
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Not Found**: Train the model first using the "Model Training" tab

3. **Memory Issues**: Reduce batch size in training configuration

4. **Slow Performance**: Use GPU acceleration if available (install tensorflow-gpu)

### Getting Help

If you encounter issues:
1. Check the terminal/command prompt for error messages
2. Ensure all dependencies are installed
3. Verify the dataset structure is correct
4. Check that Python version is 3.8 or higher

## 🔮 Future Enhancements

- [ ] Real-time camera integration
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Integration with recycling centers
- [ ] Carbon footprint calculator
- [ ] Recycling center locator
- [ ] Barcode scanning for device identification

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Made with ❤️ for a sustainable future**
