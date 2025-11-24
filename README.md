Book Cover Recognition for Library Automation
A simple book cover recognition system using Greedy Search Algorithm and OpenCV for library automation.

Overview
This project implements a book cover recognition system that can:

Store book covers in a database
Recognize book covers from query images using feature matching
Use greedy search algorithm for efficient matching
Algorithm: Greedy Search
The system uses a greedy feature matching approach:

Extract ORB (Oriented FAST and Rotated BRIEF) features from book cover images
Use greedy matching with ratio test to find best feature correspondences
Calculate similarity score based on number of good matches
Greedily select the best matching book from database
Requirements
Python 3.7+
OpenCV
NumPy
Installation
Install dependencies:
pip install -r requirements.txt
Usage
Running the Program
python main.py
Adding Books to Database
Choose option 1 from the menu
Enter book ID, name, and image path
The system will extract features and store the book
Recognizing a Book
Choose option 2 from the menu
Enter the path to the query book cover image
The system will search the database and return the best match
Project Structure
book-cover-recognition/
├── main.py              # Main recognition system
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── database/           # Database directory (created automatically)
    ├── metadata.json   # Book metadata
    └── *.jpg          # Book cover images
How It Works
Feature Extraction: Uses ORB detector to extract keypoints and descriptors
Greedy Matching:
Finds k=2 nearest neighbors for each feature
Greedily selects matches with distance ratio < 0.75
Similarity Calculation: Normalizes match count by minimum features
Best Match Selection: Greedily keeps track of best matching book
Example
from main import BookCoverRecognizer

# Initialize recognizer
recognizer = BookCoverRecognizer()

# Add a book
recognizer.add_book_to_database("001", "Python Programming", "cover1.jpg")

# Recognize a book
result = recognizer.recognize_book("query_cover.jpg")
if result:
    print(f"Found: {result['name']}")
Notes
Keep it simple: This is a basic implementation for educational purposes
Works best with clear, high-contrast book cover images
Adjust threshold parameter for sensitivity
Database is stored locally in JSON format
