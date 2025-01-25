# Project Submission

## Files Overview
1. **merge.py** - Combines datasets into a single file.
2. **clean.py** - Cleans and preprocesses the data.
3. **app.py** - Runs the main(streamlit) application.
4. **biobert.py** - Implements BioBERT for feature extraction.
5. **embedding.py** - Generates and processes embeddings.
6. **usecase1merged.py** - Merges datasets specific to Use Case 1.
7. **filteredcombined.py** - Filters and combines datasets for analysis.

## How to Reproduce the Results

### Step 1: Install Dependencies
Ensure you have Python installed. Run the following command to install required libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
Use the following command to execute the main application:
```bash
python app.py
```

### Step 3: Reproducing the Functionality
The solution uses the following libraries for key functionalities:
- **BioBERT** for extracting domain-specific features.
- **NumPy and Pandas** for data preprocessing and manipulation.
- **scikit-learn** for machine learning pipelines and evaluation.
- **matplotlib** for visualizing results.

### Packaging the Solution
The final submission includes:
1. **Codebase** - All Python scripts mentioned above.
2. **Detailed Report** - Explains the methodology, results, and conclusions.
3. **requirements.txt** - Lists all dependencies for reproducibility.
