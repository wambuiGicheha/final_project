# README: Machine learning for Depressive Content Detection on Reddit Using NLP

## Project Overview

This project explores the use of data science and natural language processing (NLP) techniques to differentiate between depressive and non-depressive content in Reddit posts. By analyzing posts from subreddits such as *Depression* and *SuicideWatch*, and comparing them to other subreddit content, the model aims to identify linguistic markers that indicate depressive tendencies.

The primary objective is to build a predictive model that can classify posts as depressive or non-depressive based on language patterns, sentiment analysis, and other text features. This model could be applied to help social media platforms and mental health organizations offer timely support or trigger positive interventions for individuals expressing depressive symptoms.

---

## Problem Statement

The problem being addressed is: **Can data science differentiate between depressive and non-depressive content based on language patterns on Reddit?**

This project seeks to:
- Test whether distinct linguistic markers, sentiment scores, or text features can indicate depressive tendencies.
- Build a model capable of classifying posts into depressive or non-depressive categories.
- Explore the potential for the model to trigger positive messaging or outreach interventions for users flagged as at-risk for depression.

## Hypothesis

The hypothesis is that an increased frequency of negative sentiment or specific language patterns in posts could correlate with a user experiencing or expressing depression.

---

## Why This Project Matters

Mental health is a critical global issue, with depression affecting millions of individuals across demographics. With the rise of social media platforms, people often use these spaces to express their emotions, including struggles with depression. 

This project leverages the vast amounts of publicly available data on Reddit to better understand how depressive tendencies manifest in online language, creating a potential path for early detection of mental health concerns. By detecting these patterns early, we hope to contribute to timely intervention and support for those who need it.

---

## Industry Relevance

This project applies to several key industries:
- **Mental Health & Psychology**: Insights from this project could be used to better understand depressive language and behavior patterns.
- **Social Media**: Platforms like Reddit, Twitter, and Facebook could benefit by improving user safety and offering support systems for users expressing signs of mental distress.
- **Data Science & Machine Learning**: NLP techniques and machine learning models will be employed for text classification, sentiment analysis, and feature extraction.
- **Public Health**: The model could contribute to public health initiatives by identifying at-risk individuals and supporting mental health interventions on digital platforms.

---

## Target Audience

The project is aimed at the following audiences:
1. **Social Media Platforms**: Tools for improving user safety, content moderation, and mental health outreach.
2. **Investors in AI & Social Impact**: AI-driven tools that address mental health concerns may attract venture capital firms and impact investors.
3. **Academic and Research Institutions**: Collaborators in psychology, psychiatry, data science, and social science research could leverage the model for studies on language patterns and mental health trends.

---

## Data Sources

- Data is sourced from Reddit posts, specifically focusing on subreddits related to depression (*Depression*, *SuicideWatch*) and other non-depression-related subreddits.
- The dataset includes post titles, body content, and metadata (e.g., upvotes, comments), which are analyzed for sentiment and linguistic patterns.

---

## Methodology

### Data Preprocessing
1. **Text Cleaning**: Tokenization, lemmatization, removal of stopwords, and punctuation.
2. **Sentiment Analysis**: Using tools like VADER sentiment analysis to quantify sentiment polarity (depressive, non depressive) for each post.
3. **Feature Engineering**: Extracting relevant linguistic markers, such as the frequency of depressive words, tone, and sentiment scores.
   
### Model Development
- **Initial Models**: Logistic Regression (as baseline),Naive Bayes, SVM (Support Vector Machine) for text classification to distinguish between depressive and non-depressive posts. Advanced modelling attempt using BERT. 
- **Comparison Models**: Baseline and advanced machine learning models, including hyperparameter tuning through RandomSearch, to optimize performance.
  
### Evaluation Metrics
- Accuracy, Precision, Recall, and F1 Score will be used to evaluate the model's performance in identifying depressive posts.

---

## Responsible Use and Ethical Considerations

- **Building a Culture of Support, Not Surveillance**: Our goal is to use this model as part of a broader effort to foster a **culture of support**, where users who are going through tough times receive help—not as a tool to police or monitor discussions, but to provide real and immediate support to those in need.


### Ethical Considerations
Given the sensitive nature of the data, ethical and responsible use is at the forefront of our approach. These are the key ethical principles guiding our model:
1. **Non-Alarmist Approach**: The model will be used as a supportive tool to flag potential depressive posts, not as a definitive diagnostic tool.
2. **Privacy Protection**: All analysis will respect user anonymity, with no personal identification attached to the posts.
3. **Non-Stigmatization**: The model will avoid attaching stigmatizing terms to users and instead provide pathways to helpful resources.

---

## Model Usage

If implemented, the model could:
- Help social media platforms provide real-time mental health support.
- Assist moderators in identifying users who may need intervention.
- Provide academic and healthcare institutions with insights into depressive trends on social media.

---

## Future Directions

1. **Integrating Positive Messaging**: Explore automated systems that trigger positive or supportive messaging to users showing depressive tendencies.
2. **Broader Deployment**: Expanding the model's use beyond Reddit to other social platforms and adapting it for diverse mental health applications.

---

## Conclusion

This project aims to demonstrate the responsible use of AI and NLP for mental health detection on social media platforms. By partnering with mental health organizations like **Mental360** in Kenya, and leveraging insights from the model, we can support users who are struggling with depression and provide timely interventions. This work has the potential to transform how we identify and address mental health concerns in digital spaces.

---

## Deployment
The live application is hosted [here](https://moodlens.streamlit.app) 

---  

## Getting Started  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.8 or above  
- Jupyter Notebook  
- Necessary libraries specified in the `requirements.txt`  

### Installation  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/moodlens.git  
   ```  
2. Navigate to the project directory:  
   ```bash  
   cd moodlens  
   ```  
3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

### Running Locally  
1. Open the Jupyter Notebook:  
   ```bash  
   jupyter notebook index.ipynb  
   ```  
2. Follow the notebook instructions to load your dataset and perform sentiment analysis.  

### Running the Web App  
1. Run the Streamlit app locally:  
   ```bash  
   streamlit run app.py  
   ```  
2. Access the app in your browser at `http://localhost:8501`.  

## Project Structure  
```  
moodlens/  
├── app.py                  # Streamlit app script  
├── index.ipynb             # Jupyter Notebook for local analysis  
├── requirements.txt        # List of dependencies  
├── README.md               # Documentation  
└── data/                   # Sample datasets  
```  

## Contributing  
We welcome contributions! Please follow these steps:  
1. Fork this repository.  
2. Create a new branch:  
   ```bash  
   git checkout -b feature-branch  
   ```  
3. Commit your changes:  
   ```bash  
   git commit -m "Feature description"  
   ```  
4. Push to the branch:  
   ```bash  
   git push origin feature-branch  
   ```  
5. Open a pull request.  


## Contact & Collaboration

The project team looks forward to collaborating to improve this model. We appreciate your insights. 

