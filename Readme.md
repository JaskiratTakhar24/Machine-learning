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
4. **Package the model:** Train with reusable code in `src/ml_learning/`, test it, and save the fitted model in `models/`.
5. **Serve predictions:** Build a FastAPI endpoint that validates input and returns a prediction from the saved model.
6. **Test and deploy:** Add tests, run the API locally, then choose a hosting platform and deployment approach.

For each stage, write down the goal, the commands you ran, what you learned, and any questions in the notes section below.

### First exercise: explore Iris

Open `notebooks/01_explore_iris.ipynb` in JupyterLab and run its cells from top to bottom. It guides you through the dataset, feature and target columns, missing-value checks, summary statistics, and a plot. Answer the reflection questions at the end before moving on to model training.

### Second exercise: train a first model

After completing the exploration notebook, open `notebooks/02_train_first_model.ipynb`. It guides you through a reproducible train/test split, a majority-class baseline, a decision tree, and evaluation with a classification report and confusion matrix. Answer the reflection questions before moving on.

### Third exercise: cross-validation and model selection

Open `notebooks/03_cross_validation_and_model_selection.ipynb` to compare decision tree depths using cross-validation on training data. It discusses overfitting and why repeatedly checking a test set while tuning can lead to misleading results. Since the test results were already viewed in exercise 2, the notebook avoids reusing that set as a final evaluation.

### Fourth exercise: package and save the model

The reusable training code is in `src/ml_learning/train.py`. From the project folder, train the selected model and save its model bundle (classifier plus feature and species names and cross-validation scores) to `models/iris_classifier.joblib`:

```bash
python src/ml_learning/train.py
```

Run the automated test from the project folder to check that training saves a model which can be loaded and used for predictions:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

On Windows PowerShell, set the source path for the test command with `$env:PYTHONPATH = "src"` first. The saved model file is generated locally and ignored by Git.

### Fifth exercise: serve predictions with a local API

Make sure the model has been trained. From the project folder, start the API with your virtual environment active:

```bash
uvicorn ml_learning.api:app --app-dir src --reload
```

The API listens on your computer at `http://127.0.0.1:8000`. The home page at `http://127.0.0.1:8000/` links to the docs and health check. Open `http://127.0.0.1:8000/docs` to try the endpoints interactively, or visit `http://127.0.0.1:8000/health` to check its health.

In another terminal, with the virtual environment active, send a sample prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm": 5.1, "sepal_width_cm": 3.5, "petal_length_cm": 1.4, "petal_width_cm": 0.2}'
```

The response contains the predicted species and the model-reported confidence. For this decision tree, that value comes from the class proportions in the matching leaf and is not guaranteed to be calibrated. The API rejects missing or non-positive measurements with a validation error. Stop the server with `Ctrl+C`.

Run all tests, including the API tests, from the project folder:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

On Windows PowerShell, set the source path with `$env:PYTHONPATH = "src"` first. This example serves only on the local machine; deployment to a public host requires additional security and configuration.

### Sixth exercise: run the API in a local Docker container

Docker packages the API and its runtime dependencies into an image. The image builds the Iris model during creation, so the generated model file on your computer is not needed.

1. Install and start Docker Desktop for your operating system, then open a terminal in the project folder.
2. Build the image:

   ```bash
   docker build -t iris-classifier-api .
   ```

3. Run the container, publishing the API only on your own computer:

   ```bash
   docker run --rm --name iris-api -p 127.0.0.1:8000:8000 iris-classifier-api
   ```

4. Visit `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/docs` and try a prediction as described above.
5. In the terminal running the container, press `Ctrl+C` to stop it. The `--rm` option removes the stopped container; the image remains available for the next run.

The container uses the minimal runtime dependencies pinned in `requirements-api.txt`, runs as a non-root user, and includes a health check. These commands are for local learning, not public deployment.

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
