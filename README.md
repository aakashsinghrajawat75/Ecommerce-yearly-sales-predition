# Ecommerce Customer Spend Prediction

A machine learning web application that estimates a customer's yearly ecommerce spending from their engagement and membership behavior.

The project includes a Jupyter notebook for exploration, a reproducible model-training script, a serialized model artifact, and a Streamlit application that can be deployed from GitHub.

## Project Demo

The Streamlit app accepts four customer measurements and returns an estimated yearly spend. It also displays a simple monthly equivalent by dividing the yearly prediction by 12.

### Intro Video

<video src="./intro.mp4" controls width="100%"></video>

If the player is not displayed in your GitHub view, [open the intro video directly](./intro.mp4).

## Features

- Exploratory data analysis in `customer.ipynb`
- Linear regression model for spend prediction
- Reproducible training script in `train_model.py`
- Serialized deployment artifact in `customer_spend_model.pkl`
- Clear input units in the Streamlit interface
- Responsive dashboard-style user interface
- Local execution and Streamlit Cloud deployment support

## Input Features

| Feature | Meaning | Unit |
| --- | --- | --- |
| Average session length | Average duration of one customer session | Minutes per session |
| Time on app | Average time the customer spends using the app | Minutes per day |
| Time on website | Average time the customer spends on the website | Minutes per day |
| Length of membership | How long the customer has been a member | Years |

## Prediction Output

The model predicts `Yearly Amount Spent`, so the primary result is a **yearly estimate**. The app also displays a **monthly equivalent**, calculated as:

```text
monthly equivalent = yearly prediction / 12
```

The monthly value is only a conversion of the yearly prediction. The model itself was trained to predict yearly spending.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Structure

```text
.
├── Ecommerce Customers.csv       # Source customer dataset
├── customer.ipynb                # Analysis and model exploration
├── train_model.py                # Rebuilds the deployment model
├── customer_spend_model.pkl      # Trained model used by the app
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files excluded from Git
└── README.md                     # Project documentation
```

## Run Locally

Create a virtual environment, install the dependencies, generate the model, and start Streamlit:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_model.py
.venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Rebuild the Model

If the CSV dataset is updated, regenerate the pickle file with:

```bash
.venv/bin/python train_model.py
```

The training script validates the required columns, removes incomplete training rows, fits the linear regression model on the available complete data, and saves the feature names with the model artifact.

## Deploy with Streamlit Cloud

1. Push this project to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io/).
3. Choose **Deploy an app**.
4. Select the GitHub repository and the `main` branch.
5. Set the main file to `app.py`.
6. Deploy the application.

Streamlit Cloud reads `requirements.txt` automatically. Keep `app.py`, `customer_spend_model.pkl`, and `requirements.txt` in the repository root.

## Model Notes

The model is trained on the included ecommerce customer dataset using linear regression. Its output is an estimate, not a guaranteed customer spend value. Predictions should be interpreted together with the quality, range, and relevance of the input data.

## Author

**Name:** Aakash Singh (kushwah/rajawat)<br>
**GitHub:** [Aakash](https://github.com/aakashsinghrajawat)<br>
**Linked in:** [Aakash](https://www.linkedin.com/in/aakashrajawat75/)<br>

**Email:** aakashrajawat75@email.com

Replace the placeholder author details above before publishing the repository.
