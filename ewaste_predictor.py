import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EWastePredictor:
    def __init__(self, dataset_dir="ewaste_dataset", image_size=(224, 224), batch_size=16):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.recycling_suggestions = self._load_recycling_suggestions()
        
    def _load_recycling_suggestions(self):
        """Load recycling suggestions for each e-waste category"""
        return {
            'Battery': {
                'description': 'Lithium-ion and alkaline batteries',
                'recycling_products': [
                    'New batteries (lithium recovery)',
                    'Steel and aluminum components',
                    'Plastic casings for new products',
                    'Chemical compounds for industrial use'
                ],
                'recycling_process': 'Batteries are dismantled, metals extracted, and materials separated for reuse',
                'environmental_impact': 'Prevents toxic chemicals from leaching into soil and water'
            },
            'Keyboard': {
                'description': 'Computer keyboards and input devices',
                'recycling_products': [
                    'Plastic pellets for new keyboards',
                    'Metal components for electronics',
                    'Rubber keycaps for various products',
                    'Circuit boards for precious metal recovery'
                ],
                'recycling_process': 'Disassembled, plastic melted and reformed, metals extracted',
                'environmental_impact': 'Reduces plastic waste and recovers valuable materials'
            },
            'Microwave': {
                'description': 'Microwave ovens and kitchen appliances',
                'recycling_products': [
                    'Steel and aluminum for new appliances',
                    'Glass for construction materials',
                    'Copper wiring for electronics',
                    'Plastic components for various products'
                ],
                'recycling_process': 'Dismantled, hazardous materials removed, metals separated',
                'environmental_impact': 'Prevents harmful substances from entering landfills'
            },
            'Mobile': {
                'description': 'Mobile phones and smartphones',
                'recycling_products': [
                    'Precious metals (gold, silver, platinum)',
                    'Rare earth elements for new devices',
                    'Glass for new screens',
                    'Plastic for various consumer products'
                ],
                'recycling_process': 'Shredded, metals extracted through chemical processes',
                'environmental_impact': 'Recovers valuable materials and reduces mining needs'
            },
            'Mouse': {
                'description': 'Computer mice and pointing devices',
                'recycling_products': [
                    'Plastic for new computer accessories',
                    'Metal components for electronics',
                    'Rubber for various products',
                    'Circuit boards for precious metal recovery'
                ],
                'recycling_process': 'Disassembled, materials separated by type',
                'environmental_impact': 'Reduces electronic waste in landfills'
            },
            'PCB': {
                'description': 'Printed Circuit Boards',
                'recycling_products': [
                    'Precious metals (gold, silver, copper)',
                    'Fiberglass for construction materials',
                    'Plastic components for new products',
                    'Rare earth elements for electronics'
                ],
                'recycling_process': 'Shredded, metals extracted through smelting and chemical processes',
                'environmental_impact': 'Recovers valuable materials and prevents toxic leaching'
            },
            'Player': {
                'description': 'Media players and audio devices',
                'recycling_products': [
                    'Plastic for new electronic casings',
                    'Metal components for electronics',
                    'Speakers for refurbished devices',
                    'Circuit boards for precious metal recovery'
                ],
                'recycling_process': 'Disassembled, functional parts tested, materials separated',
                'environmental_impact': 'Extends product lifecycle and recovers materials'
            },
            'Printer': {
                'description': 'Printers and printing devices',
                'recycling_products': [
                    'Steel and aluminum for new appliances',
                    'Plastic for various products',
                    'Ink cartridges for refilling',
                    'Electronic components for refurbishment'
                ],
                'recycling_process': 'Dismantled, hazardous materials removed, components separated',
                'environmental_impact': 'Prevents toxic ink and toner from contaminating soil'
            },
            'Television': {
                'description': 'Televisions and display devices',
                'recycling_products': [
                    'Glass for new screens or construction',
                    'Plastic for various products',
                    'Metal components for electronics',
                    'Rare earth elements for new displays'
                ],
                'recycling_process': 'Screen glass separated, hazardous materials removed, metals extracted',
                'environmental_impact': 'Prevents lead and other toxins from entering environment'
            },
            'Washing Machine': {
                'description': 'Washing machines and large appliances',
                'recycling_products': [
                    'Steel for new appliances',
                    'Copper wiring for electronics',
                    'Plastic for various products',
                    'Glass for construction materials'
                ],
                'recycling_process': 'Dismantled, materials separated by type and composition',
                'environmental_impact': 'Reduces landfill waste and recovers valuable materials'
            }
        }
    
    def create_model(self):
        """Create a CNN model for e-waste classification"""
        # Ensure class names are loaded
        if not self.class_names:
            self._load_class_names()
        
        # Use MobileNetV2 as base model for transfer learning
        base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def _load_class_names(self):
        """Load class names from dataset directory"""
        train_dir = os.path.join(self.dataset_dir, 'train')
        if os.path.exists(train_dir):
            self.class_names = sorted(os.listdir(train_dir))
        else:
            # Default class names if dataset not found
            self.class_names = [
                'Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
                'PCB', 'Player', 'Printer', 'Television', 'Washing Machine'
            ]
    
    def prepare_data(self):
        """Prepare training, validation, and test data"""
        # Ensure class names are loaded
        if not self.class_names:
            self._load_class_names()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Get class names from directory structure
        train_dir = os.path.join(self.dataset_dir, 'train')
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'val'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'test'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def train_model(self, epochs=50, patience=10):
        """Train the model with callbacks"""
        train_gen, val_gen, test_gen = self.prepare_data()
        
        # Create callbacks
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train the model
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, test_gen
    
    def evaluate_model(self, test_generator):
        """Evaluate the model on test data"""
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Get class names
        class_names = list(test_generator.class_indices.keys())
        
        # Generate classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return report, cm, predictions
    
    def predict_ewaste(self, image_path):
        """Predict e-waste type from image"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def get_recycling_suggestions(self, predicted_class):
        """Get recycling suggestions for predicted e-waste type"""
        if predicted_class in self.recycling_suggestions:
            return self.recycling_suggestions[predicted_class]
        else:
            return {
                'description': 'Unknown e-waste type',
                'recycling_products': ['Contact local recycling center for proper disposal'],
                'recycling_process': 'Professional assessment required',
                'environmental_impact': 'Proper disposal prevents environmental contamination'
            }
    
    def save_model(self, model_path="ewaste_model.h5"):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Please train the model first.")
    
    def load_model(self, model_path="ewaste_model.h5"):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load class names from the model directory or set default
        if not self.class_names:
            self.class_names = [
                'Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
                'PCB', 'Player', 'Printer', 'Television', 'Washing Machine'
            ]

def main():
    """Main function to train and evaluate the model"""
    # Initialize the predictor
    predictor = EWastePredictor()
    
    # Create and compile model
    model = predictor.create_model()
    print("Model created successfully!")
    print(f"Model summary:\n{model.summary()}")
    
    # Train the model
    print("\nStarting training...")
    history, test_gen = predictor.train_model(epochs=30, patience=5)
    
    # Evaluate the model
    print("\nEvaluating model...")
    report, cm, predictions = predictor.evaluate_model(test_gen)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(
        test_gen.classes, 
        np.argmax(predictions, axis=1),
        target_names=list(test_gen.class_indices.keys())
    ))
    
    # Save the model
    predictor.save_model()
    
    # Test prediction with a sample image
    print("\nTesting prediction...")
    test_dir = "ewaste_dataset/test"
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            sample_image = os.path.join(class_dir, os.listdir(class_dir)[0])
            predicted_class, confidence, all_predictions = predictor.predict_ewaste(sample_image)
            suggestions = predictor.get_recycling_suggestions(predicted_class)
            
            print(f"\nSample: {sample_image}")
            print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
            print(f"Recycling products: {', '.join(suggestions['recycling_products'][:3])}")
            break

if __name__ == "__main__":
    main()
