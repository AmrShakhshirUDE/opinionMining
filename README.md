# Opinion Mining
This project is designed to predict texts' subjectivity [Subjective / Objective] based on real tweets that have been labeled by our team.
> Master Project: Our team is a group of students majoring in Computer Engineering and Komedia - "Applied cognitive and media science" at University of Duisburg-Essen.

# Features
## Users can do the following:
1. Check wheather a sentence is subjective or objective.

# Try on Heroku
[Opinion-Mining](https://opinion-mining-ude.herokuapp.com/)

<!-- # Live Demo
[Live demo and screenshots](https://www.youtube.com/watch?v=USe6Ot3qFys) -->

# Screenshots
>Logo

![logo](https://github.com/AmrShakhshirUDE/opinionMining/blob/main/images/logo.png)

>Landing page

![LandingPage](https://github.com/AmrShakhshirUDE/opinionMining/blob/main/images/LandingPage.png)

>Prediction Results

![Objective Result](https://github.com/AmrShakhshirUDE/opinionMining/blob/main/images/objectiveResult.png)

![Subjective Result](https://github.com/AmrShakhshirUDE/opinionMining/blob/main/images/subjectiveResult.png)


# Technical architecture
The application consists of two main parts:
* Backend: responsible for: <br />Machine Learning <br />Server-side web application logic, consists of a server and an application.
* Frontend: the part that users interact with.

# Technologies/ libraries used
![Technologies](https://github.com/AmrShakhshirUDE/opinionMining/blob/main/images/technologies.png)
## Backend technologies
* Scikit-learn for machine learning
* Tensorflow keras for CNN model
* Flask
## Frontend technologies
* React
* Bootstrap
## Connecting frontend to backend
* Axios
* Postman: to test connectivity especially for 'POST' method
## Deploy technologies
* Gunicorn as a web server gateway interface "WSGI"
* Github
* Heroku

# How to run the project
> Using VScode editor is highly recommended

## To run locally
> Make sure to have the file `numValuesComb.csv` in the same folder that contains `scikit.py`

1. Open terminal and go to the path of `deployMasterProject.py` then type the following:
```
pip install -r requirements.txt
deployMasterProject.py
```

* First command will install all required python packages to run this app <br />
* Second command will run the backend part of the app

The backend part should be running now.

2. Download fornt-end repository from `https://github.com/AmrShakhshirUDE/masterFron`

3. Go to `src\Components\Header.js`

comment line.14

`'serverUrl': 'https://opinion-backend.herokuapp.com/',`

uncomment line.13

`'serverUrl': 'http://127.0.0.1:5000/'`

```
npm install
npm start
```

> If `npm start` doesn't work, do the following:
```
npm install rimraf -g
rimraf node_modules
npm start -- --reset-cache
```
then repeat step number 6



# Group members
> <ul><li>Amr Shakhshir</li> <li>Sophia Abel</li> <li>Lena Greiner-Hiero</li> <li>Alina Krüger</li> <li>Chiara Loverso</li></ul>

# Dataset source
Data has been collected from twitter using [twint] (https://github.com/twintproject/twint) tool and manually labeled by our team member with reliability score of 92.8% according to Krippendorff's alpha measurement.