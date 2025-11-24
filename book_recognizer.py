import cv2
import numpy as np
import os
import json
from pathlib import Path

class BookCoverRecognizer:
    def __init__(self, database_path="database"):
        self.database_path = database_path
        self.books_database = {}
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Create database directory if it doesn't exist
        Path(database_path).mkdir(exist_ok=True)
        self.load_database()
    
    def extract_features(self, image_path):
        """Extract ORB features from book cover image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors
    
    def greedy_feature_matching(self, desc1, desc2, ratio_threshold=0.75):
        """
        Greedy search algorithm for feature matching
        Greedily selects best matches based on distance ratio
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Find k=2 nearest neighbors for ratio test
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Greedy selection: choose match if distance ratio is below threshold
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def calculate_similarity(self, query_image_path, database_image_path):
        """Calculate similarity score between two book covers using greedy matching"""
        kp1, desc1 = self.extract_features(query_image_path)
        kp2, desc2 = self.extract_features(database_image_path)
        
        if desc1 is None or desc2 is None:
            return 0.0
        
        matches = self.greedy_feature_matching(desc1, desc2)
        
        # Calculate similarity score based on number of good matches
        # Normalize by minimum number of features
        min_features = min(len(desc1), len(desc2))
        if min_features == 0:
            return 0.0
        
        similarity = len(matches) / min_features
        return similarity
    
    def add_book_to_database(self, book_id, book_name, cover_image_path):
        """Add a book cover to the database"""
        if not os.path.exists(cover_image_path):
            print(f"Error: Image file not found: {cover_image_path}")
            return False
        
        # Copy image to database directory
        dest_path = os.path.join(self.database_path, f"{book_id}.jpg")
        img = cv2.imread(cover_image_path)
        cv2.imwrite(dest_path, img)
        
        # Store book information
        self.books_database[book_id] = {
            "name": book_name,
            "image_path": dest_path
        }
        
        self.save_database()
        print(f"Book '{book_name}' added to database with ID: {book_id}")
        return True
    
    def recognize_book(self, query_image_path, threshold=0.15):
        """
        Recognize a book cover from query image using greedy search
        Returns best matching book or None
        """
        if not os.path.exists(query_image_path):
            print(f"Error: Query image not found: {query_image_path}")
            return None
        
        best_match = None
        best_score = 0.0
        
        print("Searching database...")
        for book_id, book_info in self.books_database.items():
            similarity = self.calculate_similarity(
                query_image_path, 
                book_info["image_path"]
            )
            
            print(f"  {book_info['name']}: {similarity:.3f}")
            
            # Greedy selection: keep track of best match so far
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "book_id": book_id,
                    "name": book_info["name"],
                    "similarity": similarity
                }
        
        if best_match and best_match["similarity"] >= threshold:
            return best_match
        else:
            print(f"No match found (best score: {best_score:.3f} < threshold: {threshold})")
            return None
    
    def save_database(self):
        """Save database metadata to JSON file"""
        metadata_path = os.path.join(self.database_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.books_database, f, indent=2)
    
    def load_database(self):
        """Load database metadata from JSON file"""
        metadata_path = os.path.join(self.database_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.books_database = json.load(f)
            print(f"Loaded {len(self.books_database)} books from database")
        else:
            print("Database is empty. Add books to get started.")


def main():
    """Main function to demonstrate book cover recognition"""
    recognizer = BookCoverRecognizer()
    
    print("\n=== Book Cover Recognition System ===")
    print("1. Add book to database")
    print("2. Recognize book from image")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            book_id = input("Enter book ID: ").strip()
            book_name = input("Enter book name: ").strip()
            image_path = input("Enter image path: ").strip()
            recognizer.add_book_to_database(book_id, book_name, image_path)
        
        elif choice == "2":
            query_path = input("Enter query image path: ").strip()
            result = recognizer.recognize_book(query_path)
            
            if result:
                print(f"\n✓ Match found!")
                print(f"  Book: {result['name']}")
                print(f"  ID: {result['book_id']}")
                print(f"  Similarity: {result['similarity']:.3f}")
            else:
                print("\n✗ No match found in database")
        
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()