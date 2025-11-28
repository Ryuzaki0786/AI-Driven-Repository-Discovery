AI-Driven Repository Discovery

A lightweight, text-based recommendation system that analyzes research data repositories from re3data.org using TF-IDF vectorization and cosine similarity.
The system fetches live repository metadata via the re3data API, processes descriptions and subject tags, and recommends similar repositories based on textual similarity.

What This Project Does

Fetches repository metadata (names, descriptions, subjects) from re3data.org’s public API.

Cleans and merges text fields into a single representation for analysis.

Converts repository text into TF-IDF vectors.

Computes cosine similarity to identify repositories with similar content.

Includes a small validation step using a train/test split.

Core Components

Data Extraction
Uses requests + XML parsing to pull repository info and detailed metadata.

Text Processing
Builds “Combined_Text” from name, description, and subject tags.

Modeling
TF-IDF vectorizer (sklearn) transforms text into sparse vectors.

Similarity Engine
Computes cosine similarity between repositories to generate top match recommendations.

Technologies Used

Python

requests

xml.etree.ElementTree

pandas

scikit-learn (TF-IDF, cosine similarity, train/test split)
