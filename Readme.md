# Machine Learning Project

Use this README as the project home page and a running index for your learning notes, experiments, and next steps. Update the sections as the project develops.

## Project overview

- **Goal:** Learn the end-to-end process of developing and deploying a machine-learning model.
- **Current focus:** Build an Iris flower classifier and serve it through a local API.
- **Started:** 2026-11-01

## Learning roadmap

Work through one stage at a time. The Iris dataset is small and included with scikit-learn, so you can focus on the workflow without first finding or downloading data.

1. **Explore the data:** Learn what each flower measurement and label represents; inspect the dataset in a notebook.
2. **Train a first model:** Split data into training and test sets, fit a scikit-learn classifier, and establish a baseline.
3. **Evaluate and improve:** Use suitable metrics, inspect mistakes, and learn why test data must stay separate from training.
4. **Package the model:** Move reusable training code into `src/ml_learning/` and save the fitted model in `models/`.
5. **Serve predictions:** Build a FastAPI endpoint that validates input and returns a prediction from the saved model.
6. **Test and deploy:** Add tests, run the API locally, then choose a hosting platform and deployment approach.

For each stage, write down the goal, the commands you ran, what you learned, and any questions in the notes section below.

### First exercise: explore Iris

Open `notebooks/01_explore_iris.ipynb` in JupyterLab and run its cells from top to bottom. It guides you through the dataset, feature and target columns, missing-value checks, summary statistics, and a plot. Answer the reflection questions at the end before moving on to model training.

## Getting started

### 1. Open the project folder

In a terminal, move into this folder:

```bash
cd "Machine-learning"
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the project dependencies

With the virtual environment activated, run:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter joblib fastapi uvicorn
python -m pip freeze > requirements.txt
```

### 4. Create the folder structure

From the project folder, run:

```bash
mkdir -p data notebooks src/ml_learning models tests
touch src/ml_learning/__init__.py .gitignore
```

### 5. Start a notebook

Launch JupyterLab:

```bash
jupyter lab
```

Create a notebook in the project folder and use it to work through lessons and experiments. To leave the virtual environment when you are finished, run `deactivate`.

## Suggested learning workflow

1. **Choose a topic.** Write down the concept or question you want to explore.
2. **Capture the basics.** Summarize definitions, formulas, and important assumptions in your own words.
3. **Work through a small example.** Use a notebook to calculate or visualize the idea.
4. **Try it on data.** Record where the data came from and any cleaning or preparation you performed.
5. **Evaluate the result.** Note the method, metric, outcome, and anything that might make the result misleading.
6. **Reflect and continue.** Record what you learned, what remains unclear, and the next small task.

## Notes and experiments

Copy this template for each topic or experiment:

### Topic or experiment: _Name_

- **Date:**
- **Question or goal:**
- **Key ideas:**
- **Data and source:**
- **Steps taken:**
- **Result:**
- **What I learned:**
- **Open questions:**
- **Next step:**

## Useful library groups

The dependencies in `requirements.txt` include:

- **Data and numerical work:** NumPy, pandas, SciPy
- **Machine learning:** scikit-learn
- **Charts:** Matplotlib
- **Interactive work:** Jupyter and IPython

## Suggested project organization

As the project grows, you can organize your work like this:

```text
Machine-learning/
├── Readme.md
├── requirements.txt
├── data/            # Datasets (avoid committing private or very large data)
├── notebooks/       # Lessons, exploration, and experiments
├── src/ml_learning/ # Reusable training and prediction code
├── models/          # Saved model artifacts
└── tests/           # Automated tests
```

## Progress log

Add a short entry whenever you complete a meaningful step:

| Date | Topic or milestone | Notes / next step |
|---|---|---|
| 2026-11-01 | Project setup | Folders and dependencies are ready; begin by exploring the Iris dataset. |
