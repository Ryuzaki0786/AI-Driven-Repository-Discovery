# AI-Driven-Repository-Discovery
Text-based recommendation system for Re3Data repositories using TF-IDF and cosine similarity.
Explaining the Re3data Repository Recommendation Code
This document provides a detailed explanation of the Python code used to fetch data from re3data.org, process it using TF-IDF, and then use cosine similarity to recommend repositories.

1. Data Fetching from re3data.org
The code begins by fetching data about research data repositories from the re3data.org API.

fetch_re3data_repo_data() function
This function is responsible for retrieving the initial list of repositories and then fetching detailed information for each.

API Endpoint: The code uses the URL https://www.re3data.org/api/v1/repositories?size=100 to access the re3data.org API. The size=100 parameter limits the initial response to 100 repositories.
HTTP Request: The requests.get(url) call sends an HTTP GET request to the API endpoint.
Error Handling: response.raise_for_status() checks if the request was successful. If not, it raises an HTTPError.
XML Parsing: The response content, which is in XML format, is parsed using xml.etree.ElementTree as ET. The root element of the XML is obtained.
Extracting Repository Information: The code iterates through <repository> elements in the XML. For each repository, it extracts:
Name: The text content of the <name> element.
URL: The href attribute of the <link> element. This URL is used to fetch detailed information later.
Creating Initial DataFrame: The extracted names and URLs are stored in a list of dictionaries, which is then converted into a pandas DataFrame.
Fetching Detailed Data: The code then iterates through the rows of the initial DataFrame. For each repository URL, it calls the fetch_repo_details() function to get more information.
Combining Data and Creating Final DataFrame: The detailed information (description and subjects) fetched by fetch_repo_details() is added to the data. A "Combined_Text" column is created by concatenating the name, description, and subjects. This combined text is crucial for the TF-IDF vectorization. A final DataFrame containing "ID", "Name", "URL", "Description", "Subjects", and "Combined_Text" is created and returned.
Additional Error Handling: try...except blocks are used to catch potential requests.exceptions.RequestException (for HTTP errors), ET.ParseError (for XML parsing errors), and general Exception during both the initial fetch and the detailed fetch.
fetch_repo_details(repo_url) function
This function fetches detailed information for a single repository given its API URL.

HTTP Request: It sends an HTTP GET request to the provided repo_url.
Error Handling: response.raise_for_status() checks for HTTP errors.
XML Parsing: The XML response for the specific repository is parsed.
Extracting Details:
Description: It finds the <r3d:description> element within the <r3d:repository> element using an XML namespace (r3d). The text content of this element is extracted as the description.
Subjects: It finds all <r3d:subject> elements within <r3d:subjects>. The text content of each subject element is extracted and added to a list.
Returning Details: The extracted description and a comma-separated string of subjects are returned.
Error Handling: Similar try...except blocks are used as in fetch_re3data_repo_data() to handle potential errors during the details fetching process.
2. Data Preparation for Modeling
Before applying TF-IDF and cosine similarity, the data is prepared.

Splitting Data into Training and Testing Sets
train_test_split(df, test_size=10, random_state=42, shuffle=True): This function from sklearn.model_selection splits the main DataFrame (df) into two subsets:
corpus_df: This will be used as the "corpus" for the TF-IDF vectorizer. The vectorizer will learn the vocabulary and IDF scores from this data.
test_df: This will be used to test the recommendation system. We will use the models trained on the corpus to find similar repositories for the test repositories.
test_size=10: Specifies that 10 repositories will be put into the test set.
random_state=42: Ensures that the split is the same every time the code is run, making the results reproducible.
shuffle=True: Randomly shuffles the data before splitting.
reset_index(drop=True): Resets the index of both the corpus_df and test_df DataFrames, dropping the original index. This is important for consistent indexing in subsequent steps.
3. TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection or corpus.

Term Frequency (TF): Measures how frequently a term appears in a document.
Inverse Document Frequency (IDF): Measures how important a term is across the whole corpus. Rare terms have a higher IDF score.
The TF-IDF score is the product of TF and IDF. It gives more weight to terms that are frequent in a specific document but rare in the overall corpus.

Applying TF-IDF
TfidfVectorizer(stop_words='english'): An instance of TfidfVectorizer is created.
stop_words='english': This argument tells the vectorizer to remove common English stop words (like "the", "a", "is") from the text before calculating TF-IDF scores. This helps focus on more meaningful terms.
corpus_matrix = tfidf.fit_transform(corpus_df['Combined_Text']):
fit: The vectorizer learns the vocabulary of the Combined_Text column in the corpus_df (the training data). It also calculates the IDF scores for each term based on its frequency across all documents in the corpus.
transform: It converts the Combined_Text of each repository in the corpus into a TF-IDF vector. Each vector represents a repository as a sequence of numbers, where each number is the TF-IDF score of a term in the vocabulary. The result is a sparse matrix (corpus_matrix) where rows represent repositories and columns represent terms.
4. Cosine Similarity
Cosine similarity is a metric used to measure how similar two non-zero vectors are. It calculates the cosine of the angle between two vectors. A cosine similarity of 1 means the vectors are identical in direction (most similar), while a similarity of 0 means they are orthogonal (no similarity).

In this context, cosine similarity is used to find how similar the TF-IDF vector of one repository is to the TF-IDF vectors of other repositories. A higher cosine similarity score indicates a greater similarity between the content (name, description, subjects) of the repositories.

Calculating Cosine Similarity
cosine_sim_corpus = cosine_similarity(corpus_matrix, corpus_matrix): This calculates the cosine similarity between all pairs of repositories within the corpus. The result is a square matrix (cosine_sim_corpus) where the entry at row i and column j represents the cosine similarity between the i-th and j-th repositories in the corpus.
5. Testing and Validation
The code then prepares the test data and uses the trained TF-IDF vectorizer and cosine similarity to recommend repositories from the corpus for the test repositories.

test_matrix = tfidf.transform(test_df['Combined_Text']): The transform method of the already fitted tfidf vectorizer is used to convert the Combined_Text of the test repositories into TF-IDF vectors. It's important to use transform here, not fit_transform, because the vectorizer should use the vocabulary and IDF scores learned from the training corpus.
test_vs_corpus_sim = cosine_similarity(test_matrix, corpus_matrix): This calculates the cosine similarity between the TF-IDF vectors of the test repositories and the TF-IDF vectors of the corpus repositories. The result is a matrix where each row corresponds to a test repository and each column corresponds to a corpus repository. The entry at row i and column j is the cosine similarity between the i-th test repository and the j-th corpus repository.
validate_recommendations(test_repo_row_index, test_vs_corpus_sim) function
This function takes the index of a test repository and the test_vs_corpus_sim matrix to find and display the top recommended repositories from the corpus.

sim_scores = list(enumerate(test_vs_corpus_sim[test_repo_row_index])): This extracts the row from the test_vs_corpus_sim matrix that corresponds to the given test_repo_row_index. This row contains the similarity scores between the test repository and all corpus repositories. enumerate is used to pair each score with its original index in the corpus.
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True): The list of (index, score) tuples is sorted in descending order based on the similarity score (x[1]).
top_indices = [i[0] for i in sim_scores[:3]]: The indices of the top 3 most similar corpus repositories are extracted from the sorted list.
recommended_names = corpus_df['Name'].iloc[top_indices].tolist(): The names of the recommended repositories are retrieved from the corpus_df using the extracted top indices.
Printing Results: The function then prints the name of the test repository and the names of the top 3 recommended repositories from the corpus.
Conclusion
This code demonstrates a basic content-based recommendation system for research data repositories. By fetching data from re3data.org, vectorizing the repository descriptions and subjects using TF-IDF, and then calculating cosine similarity, the system can identify repositories that are similar in content. This approach can be extended and refined for more sophisticated recommendation engines.
