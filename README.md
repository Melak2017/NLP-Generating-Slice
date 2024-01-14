# Slicing Algorithm

This repository contains an implementation of an algorithm for generating slices of excessive context windows for ChatGPT 3.5. The goal is to handle inputs that exceed the standard size of the context window (128 MB) by dividing them into smaller, overlapping slices. The slices are generated based on a set of criteria to ensure proper coverage and distinctiveness.

## Slicing Algorithm

### Pipeline

1. **Standard Size Handling:**
   - If the input is below the standard size, it is passed as-is to the ChatGPT 3.5 model.

2. **Slicing for Large Inputs:**
   - For inputs exceeding the standard size, they are divided into a finite number of slices.
   - Each slice is of a size that fits within the context window, ensuring the sum of all slices is greater than or equal to the input length.

3. **Coverage Criteria:**
   - Two slices can overlap.
   - No slice is included in another slice.
   - Adjacent slices must be distinct enough based on cosine distance of bag-of-words representation.

### Criteria for Slice Comparison

- **Cosine Distance:**
  - Comparison of two slices is based on cosine distance of bag-of-words.
  - Bag-of-words construction involves stopword elimination, stemming/lemmatization, and count of occurrences weighted on the length of the document.

- **Threshold:**
  - A threshold for distance is set empirically; a reasonable threshold like 20% can be used.

### Implementation Details

The slicing algorithm utilizes NLTK for natural language processing tasks, including tokenization, stopword removal, and stemming. The implementation is in Python and follows the specified criteria for generating slices.

---

## Project Structure

    .
    ├── .github/workflows              # github actions               
    │
    ├── notebooks                        # Jupyter notebooks for exploration and experimentation         
    │   ├── pre_processing.ipynb      # notebook containing processing and modeling
    │
    |    
    ├── requirements.txt               # a text file lsiting the projet's dependancies
    ├── .gitignore                     # files to ignore when committing
    ├──LICENSE                         # license file
    └── README.md                      # Markdown text with explanation of the project and the structure.

## Install

```
git clone https://github.com/Melak2017/NLP-Generating-Slice
pip install -r requirements.txt
```

---


[back to top](#SlicingAlgorithm)