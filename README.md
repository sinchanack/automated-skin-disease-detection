## Automated Skin Disease Detection System

An AI-based web application for identifying and understanding various skin diseases using deep learning and image processing techniques.

## Overview
Skin diseases often require expert diagnosis or invasive procedures like biopsies. This system provides a cost-effective and fast solution for early-stage detection using dermoscopic images.

## Model Details
- Dataset used: HAM10000
- Model: Custom Convolutional Neural Network (CNN)
- Accuracy: ~85%
- Performance: Precision, Recall, and F1-score evaluated

## Features
- Check Disease: Upload skin images and get top 3 predictions with confidence scores
- Know About Disease: Learn about symptoms and causes of common skin conditions
- User-Friendly Interface: Clean frontend and backend built with HTML/CSS/Flask

## Disease Categories
- Actinic Keratosis
- Basal Cell Carcinoma
- Benign Keratosis
- Dermatofibroma
- Melanoma
- Melanocytic Nevi
- Vascular Naevus
- Normal Skin

## Tech Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- ML Framework: TensorFlow/Keras
- Model Files: `.h5`, `.json`
- Database: Excel and JSON files
- Hosting: Localhost or Flask server

## Folder Structure
Skin/
├── app.py
├── model.h5 / model1.weights.h5 / model.json
├── templates/
│ ├── index.html
│ ├── about.html
│ ├── contact.html
│ ├── service.html
│ └── success.html
├── user_data.xlsx / users_db.json
├── homebg.png / skinbg.jpg
├── ImageDatasets/ | temp/ | Test_image/


## License
For educational use only.


