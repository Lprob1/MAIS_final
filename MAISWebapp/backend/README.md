# MAIS Fall 2020 final project
To run the web app, make sure that the following python libraries are installed:
- sklean
- numpy
- pandas
- pickle
- os
- flask
- string
- nltk

# Running the application
Run the **app.py** file from the **root** directory. 
`python app.py` 
Go to **localhost:5000** to access the application from the browser of your choice.

# Project Architecture

## Structure of the folder

```
├── README.md
├── app.py                      # Main code to run the flask app
├── model
│   ├── main_wine_class.py      # contains the code for the SVM and other models (RFC, Naive Bayes)
│   └── results
│       └── svm.pickle          # Weights of the trained RFC
│       └── vectorizer.pickle   # Weights of the trained TFIDF vectorizer
├── requirements.txt            # File containing packages needed to run the code
├── static
│   ├── css
│   │   └── main.css            # Style sheet to make the front-end prettier
│   └── RIOJAWINE.jpg           # Background image

└── templates
    └── index.html              # HTML file that Flask renders
```

# Credits
The layout of this README, as well as some code for the app.py file, is taken from Youssef Bou and William Zhang. Thank you guys!