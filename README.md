# Book Cover Recognition for Library Automation

A simple book cover recognition system using *Greedy Search Algorithm* and *OpenCV* for library automation.

# Overview

This project implements a book cover recognition system that can:
- Store book covers in a database
- Recognize book covers from query images using feature matching
- Use greedy search algorithm for efficient matching

# Algorithm: Greedy Search

The system uses a *greedy feature matching* approach:
1. Extract ORB (Oriented FAST and Rotated BRIEF) features from book cover images
2. Use greedy matching with ratio test to find best feature correspondences
3. Calculate similarity score based on number of good matches
4. Greedily select the best matching book from database

# Requirements

- Python 3.7+
- OpenCV
- NumPy

# Installation

1. Install dependencies:
bash
pip install -r requirements.txt


# Usage

# Running the Program

bash
python main.py


# Adding Books to Database

1. Choose option 1 from the menu
2. Enter book ID, name, and image path
3. The system will extract features and store the book
**
# Recognizing a Book

1. Choose option 2 from the menu
2. Enter the path to the query book cover image
3. The system will search the database and return the best match

# Project Structure


book-cover-recognition/
├── main.py              # Main recognition system
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── database/           # Database directory (created automatically)
    ├── metadata.json   # Book metadata
    └── *.jpg          # Book cover images


# How It Works

1. *Feature Extraction*: Uses ORB detector to extract keypoints and descriptors
2. *Greedy Matching*: 
   - Finds k=2 nearest neighbors for each feature
   - Greedily selects matches with distance ratio < 0.75
3. *Similarity Calculation*: Normalizes match count by minimum features
4. *Best Match Selection*: Greedily keeps track of best matching book

# Example

python
from main import BookCoverRecognizer

# Initialize recognizer
recognizer = BookCoverRecognizer()

# Add a book
recognizer.add_book_to_database("001", "Python Programming", "cover1.jpg")

# Recognize a book
result = recognizer.recognize_book("query_cover.jpg")
if result:
    print(f"Found: {result['name']}")


# Notes

- Keep it simple: This is a basic implementation for educational purposes
- Works best with clear, high-contrast book cover images
- Adjust threshold parameter for sensitivity
- Database is stored locally in JSON format