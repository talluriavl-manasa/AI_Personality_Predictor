from flask import Flask, render_template, request
import pickle
import random

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Questions list
questions = [

"How do you handle difficult challenges?",

"What role do you usually take in a team?",

"What motivates you to achieve your goals?",

"How do you react when you fail at something?",

"What type of work environment do you prefer?",

"How do you solve complex problems?",

"What are your strengths when working with others?",

"How do you manage responsibilities?",

"What do you do when a team member disagrees with you?",

"How do you plan your long-term goals?",

"What inspires you to work harder?",

"How do you deal with stress during work or study?",

"What do you enjoy more: leading or supporting others?",

"How do you make important decisions?",

"What do you do when facing unexpected problems?",

"How do you stay motivated when tasks become difficult?",

"What is your approach when working on a new project?",

"How do you organize your daily tasks?",

"How do you contribute in group discussions?",

"What do you do when multiple tasks need to be completed?",

"How do you react when someone criticizes your work?",

"What type of tasks do you enjoy the most?",

"How do you handle conflicts within a team?",

"What helps you stay focused on your goals?",

"How do you balance teamwork and individual work?"

]

# Suggestions for each personality
suggestions = {
"Leader": "Improve listening skills and encourage teamwork.",
"Creative": "Focus on planning and time management.",
"Analyst": "Improve communication and collaboration.",
"Team Player": "Develop leadership and decision-making skills.",
"Independent": "Try participating more in teamwork."
}

@app.route("/")
def home():

    selected_questions = random.sample(questions, 7)

    return render_template("index.html", questions=selected_questions)


@app.route("/predict", methods=["POST"])
def predict():

    answers = request.form.getlist("answers")

    text = " ".join(answers)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]
    labels = model.classes_

    scores = dict(zip(labels, probabilities))

    suggestion = suggestions[prediction]

    return render_template(
        "result.html",
        personality=prediction,
        suggestion=suggestion,
        scores=scores
    )


if __name__ == "__main__":
    app.run(debug=True)