# Message Intelligence System

A supervised learning project that classifies digital messages as **Spam** or **Legitimate**. It compares three classifiers (KNN, SVM and Naive Bayes) and explains their behaviour using probability concepts such as conditional probability and Bayes' Theorem.

🚀 **Live App:** [Open the Streamlit app](PASTE_STREAMLIT_APP_URL_HERE)

🎥 **Project Explanation Video:** [Watch the video](PASTE_VIDEO_LINK_HERE)

## Objective

Build a message classification system for a communication security company, compare distance-based, margin-based and probabilistic classifiers, and recommend the best model for real-world deployment.

## Dataset

`Message_Intelligence_Dataset_5200.xlsx`: 5,200 messages (4,227 legitimate, 973 spam).

| Group | Features |
|---|---|
| Message | `message_length`, `word_count`, `num_urls`, `num_digits`, `num_special_chars`, `spam_keyword_score`, `legit_keyword_score` |
| Sender | `sender_activity_score`, `sender_account_age_days`, `messages_sent_last_24h` |
| Time | `hour_of_day`, `day_of_week` |
| Target | `spam_label` (0 = Legitimate, 1 = Spam) |

Missing values appear in three sender columns and are filled with the median. `message_id`, `message_text` and `timestamp` are not used as inputs.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit

## Project Structure

```
Message Intelligence System/
├── app.py                              # Streamlit application
├── requirements.txt
├── LICENSE.txt
├── README.md
├── data/
│   └── Message_Intelligence_Dataset_5200.xlsx
├── notebooks/
│   └── Message_Intelligence_System.ipynb
├── model/
│   ├── message_intelligence_pipeline.pkl
│   └── metrics.json
├── visuals/                            # Charts used in this README
├── part-a_throry.pdf                   # Part A theory
└── PR. 4.pdf                           # Project brief
```

## Methodology

| Part | Coverage |
|---|---|
| A | Probability and conceptual theory |
| B | Dataset understanding, preprocessing (median imputation, scaling) and stratified train-test split |
| C | K-Nearest Neighbors: effect of K and distance metrics, misclassification analysis |
| D | Support Vector Machine: linear, polynomial and RBF kernels, margin and support vectors |
| E | Naive Bayes: manual conditional probabilities and Bayes' Theorem compared with model output |
| F | Model comparison using accuracy, precision, recall and F1-Score |
| G | Final analysis and business recommendation |

## Results

Test set: 1,040 messages (845 legitimate, 195 spam).

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | 1.00 | 1.00 | 1.00 | 1.00 |
| SVM | 1.00 | 1.00 | 1.00 | 1.00 |
| Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 |

All three models make zero errors on the test set. This is a property of the dataset: every spam message contains at least one special character and no legitimate message does, so a single feature separates the classes perfectly. Real-world spam is messier, so the model should be re-tested on fresh messages before it is fully trusted.

**Final model: linear SVM.** It ties with the others on every metric, but it is the most compact (only 10 support vectors, about 0.2% of the training data), has a stable linear boundary and gives smooth spam probabilities. Naive Bayes is kept as a fast baseline, and KNN has to store all training messages.

## Visualizations

![Spam vs Legitimate messages](visuals/Spam%20vs%20Legitimate%20Messages.png)

*Class distribution: 81.3% legitimate and 18.7% spam, a mild imbalance.*

![Correlation heatmap](visuals/Correlation%20Heatmap.png)

*Special characters, messages sent in 24 hours and keyword scores have the strongest link with spam. Several of these features are also correlated with each other, which breaks the Naive Bayes independence assumption.*

![Effect of K on KNN](visuals/Effect%20of%20K%20on%20KNN.png)

*Cross-validated F1-Score is highest at K = 1 and drops only slightly as K grows, because the classes are well separated.*

![Linear SVM boundary, margins and support vectors](visuals/Linear%20SVM_Boundary_Margins%20and%20Support%20Vectors.png)

*Linear SVM drawn on two features: the solid line is the decision boundary, the dashed lines are the margins and the circled points are the support vectors.*

![KNN vs SVM vs Naive Bayes](visuals/KNN%20vs%20SVM%20vs%20Naive%20Bayes.png)

*All three models are tied at 1.0 on accuracy, precision, recall and F1-Score.*

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The notebook can be opened with Jupyter from the `notebooks/` folder. Running it regenerates the files in `model/`.

## Streamlit App

Enter the message and sender details, click **Predict**, and the app shows the input you gave, the predicted class with its spam probability, and a **Model Performance** dropdown with the metrics saved in `model/metrics.json`.

## License

This project is released under the MIT License. See [LICENSE.txt](LICENSE.txt).

## Author

Om Vaja: [GitHub](https://github.com/omvaja693-create) · [LinkedIn](https://www.linkedin.com/in/om-vaja-35b1103a3)
