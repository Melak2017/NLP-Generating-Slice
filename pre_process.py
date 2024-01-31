import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

import wikipediaapi
import json

def fetch_wikipedia_content(wiki_api, titles, size_limit_mb):
    content_dict = {}
    total_size_mb = 0

    for title in titles:
        page = wiki_api.page(title)
        if not page.exists():
            continue  # Skip non-existent pages
        page_content = page.text
        content_size_mb = len(page_content.encode('utf-8')) / (1024 ** 2)

        if total_size_mb + content_size_mb > size_limit_mb:
            return content_dict  # Return the current content if adding more would exceed the limit

        content_dict[title] = page_content
        total_size_mb += content_size_mb

    return content_dict

def write_to_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

ai_ml_titles = [
    "Artificial Intelligence", "Machine Learning", "Deep Learning", "Neural Networks",
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
    "Natural Language Processing", "Computer Vision", "Speech Recognition",
    "Genetic Algorithms", "Support Vector Machines", "Decision Trees",
    "Random Forests", "Gradient Boosting Machines", "Convolutional Neural Networks",
    "Recurrent Neural Networks", "Transfer Learning", "Feature Engineering",
    "Bias-Variance Tradeoff", "Dimensionality Reduction", "Principal Component Analysis",
    "Clustering Algorithms", "K-Means Clustering", "Hierarchical Clustering",
    "Deep Reinforcement Learning", "Generative Adversarial Networks", "Robotics",
    "Ethics in AI", "Explainable AI", "AI in Healthcare", "AI in Finance",
    "AI in Autonomous Vehicles", "AI in Gaming", "Data Mining", "Big Data Analytics"
]



user_agent = "MyApp/1.0 (melakemekonnen100@gmail.com)"
wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI, user_agent=user_agent)

ai_ml_content = fetch_wikipedia_content(wiki_wiki, ai_ml_titles, 128)
write_to_json('ai_ml_data.json', ai_ml_content)


import json
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_file_metrics(data, file_path):
    file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
    num_pages = len(data)
    total_word_count = sum(len(content.split()) for content in data.values())
    distinct_articles = set(data.keys())

    return file_size_mb, num_pages, total_word_count, distinct_articles

def display_metrics(file_size_mb, num_pages, total_word_count, distinct_articles):
    # Header
    print("{:<30} | {:<15}".format("Metric", "Value"))
    print("-" * 47)

    # Metrics
    print("{:<30} | {:<15.2f} MB".format("File Size", file_size_mb))
    print("{:<30} | {:<15}".format("Number of Pages", num_pages))
    print("{:<30} | {:<15}".format("Total Word Count", total_word_count))
    print("{:<30} | {:<15}".format("Distinct Articles", len(distinct_articles)))

# Main code
json_file_path = 'ai_ml_data.json'
data = load_data(json_file_path)
file_size_mb, num_pages, total_word_count, distinct_articles = get_file_metrics(data, json_file_path)
display_metrics(file_size_mb, num_pages, total_word_count, distinct_articles)


long_text = ("In computer science, artificial intelligence (AI), sometimes called machine intelligence, "
             "is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans "
             "and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that "
             "perceives its environment and takes actions that maximize its chance of successfully achieving its goals. "
             "Colloquially, the term 'artificial intelligence' is often used to describe machines (or computers) that "
             "mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'. "
             "As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the "
             "definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says 'AI is whatever hasn't "
             "been done yet.' For instance, optical character recognition is frequently excluded from things considered to be AI, "
             "having become a routine technology.") * 5  

#Preprocess 
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def calculate_cosine_similarity(slice1, slice2):
    vectorizer = CountVectorizer().fit([slice1, slice2])
    vectorized_slices = vectorizer.transform([slice1, slice2])
    similarity = cosine_similarity(vectorized_slices)[0][1]
    return similarity

def slice_input(input_text, slice_size=500, overlap=100):
    slices = []
    start = 0
    end = slice_size

    while start < len(input_text):
   
        current_slice = input_text[start:end]
        slices.append(current_slice)
        print(f"Slice {len(slices)}:\n{current_slice}\n")

        start = end - overlap
        end = start + slice_size

    return slices

sliced_demo_text = slice_input(long_text)
#len(sliced_demo_text), sliced_demo_text[:3]  

