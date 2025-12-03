Email Spam Classifier – Machine Learning Project using Python

This project is a Machine Learning–based **Email Spam Classification System** built using Python, Scikit-learn, TF-IDF vectorization, and deployed using Streamlit.
The application classifies input email text as **Spam** or **Ham (Not Spam)** with a confidence score.

1. Project Overview

This project demonstrates the end-to-end workflow of a supervised text classification system, including:

* Data preprocessing
* Feature extraction using TF-IDF
* Model training using Multinomial Naive Bayes
* Building an interactive web interface with Streamlit
* Real-time email classification
* Model deployment-ready structure

The goal is to help users easily identify spam or fraudulent messages.

2. Features

* Classifies emails as **Spam** or **Ham**
* Displays **confidence percentage**
* Light, Dark, and System theme compatibility
* Clean, responsive UI
* Works on mobile devices when connected through the same network
* Ready for cloud deployment (Streamlit Cloud)


3. Project Structure

```
Email-Spam-Classifier/
│── app.py
│── Train.py
│── spam.csv
│── model.joblib
│── vectorizer.joblib
│── requirements.txt
│── README.md
```

---

4. How It Works:

1. User inputs email text into the application.
2. Text is vectorized using the **TF-IDF Vectorizer**.
3. The trained **Multinomial Naive Bayes** model predicts whether the email is spam or not.
4. The application displays:

   * Classification result
   * Confidence score (%)
   * Styled message box based on the result

5. Installation and Setup

Step 1: Clone the Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Step 2: Install Dependencies

```
pip install -r requirements.txt
```

Step 3: (Optional) Retrain the Model

If you want to retrain the model:

```
python Train.py
```

Step 4: Run the Streamlit Application

```
streamlit run app.py
```

---

6. Requirements

All required packages are listed in `requirements.txt`.

Example:

```
streamlit
scikit-learn
pandas
joblib
```

7. Examples of SPAM Emails:
1."Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry details."
2."WINNER!! As a valued network customer you have been selected to receive a £900 prize reward."
3."Six chances to win CASH. From 100 to 20,000 pounds. Text CSH11 to 87575."
4."URGENT! You have won a 1 week FREE membership in our £100,000 prize jackpot!"
5."You have won a guaranteed cash prize. To claim, call this number now.

8. Examples of HAM Emails:
1."Even my brother is not like to speak with me. They treat me like aids patient."
2."I am in office now. Will call you later."
3."I'm gonna be home soon and I don't want to talk about this stuff anymore tonight."
4."Don’t worry, I already submitted the form."
5."Let’s plan for a movie this weekend."


9. Running on Mobile (Same WiFi)

1. Run the application using:

   ```
   streamlit run app.py
   ```
2. Note the **Network URL** shown in the terminal (e.g., `http://192.168.xx.xx:8501`).
3. Open the same URL on your phone connected to the same WiFi network.

10. Author

**Reshmitha Kollati**
Python Developer | AI/ML Enthusiast


11. Contribution

Contributions, suggestions, and improvements are welcome.
Feel free to open issues or submit pull requests.

12. License

This project can be used for learning and academic purposes.
For commercial use, please request permission.
