from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None
    probability = None
    risk = None
    performance_score = None
    recommendation = []
    study_plan = []

    if request.method == "POST":

        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        assignments = float(request.form["assignments"])

        data = pd.DataFrame(
            [[study_hours, attendance, assignments]],
            columns=["study_hours","attendance","assignments"]
        )

        result = model.predict(data)

        prob = model.predict_proba(data)[0][1] * 100
        probability = round(prob,2)

        if result[0] == 1:
            prediction = "Student likely to PASS"
        else:
            prediction = "Student likely to FAIL"

        if probability >= 70:
            risk = "Low Risk"
        elif probability >= 40:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        performance_score = round((study_hours*10 + attendance*0.5 + assignments*5)/2,2)

        if study_hours < 4:
            recommendation.append("Increase study hours to at least 5 hours/day")

        if attendance < 75:
            recommendation.append("Improve attendance above 80%")

        if assignments < 5:
            recommendation.append("Complete more assignments")

        if not recommendation:
            recommendation.append("Excellent! Keep maintaining your performance.")

        if risk == "High Risk":
            study_plan = [
                "Morning: 2 hours concept revision",
                "Afternoon: 1 hour assignment practice",
                "Evening: 2 hours problem solving"
            ]

        elif risk == "Medium Risk":
            study_plan = [
                "Morning: 1 hour revision",
                "Evening: 2 hours practice questions"
            ]

        else:
            study_plan = [
                "Maintain your current study routine",
                "Practice 1 hour daily to stay consistent"
            ]

        labels = ['Study Hours','Attendance','Assignments']
        values = [study_hours, attendance, assignments]

        plt.figure()
        plt.bar(labels, values)
        plt.ylim(0,100)
        plt.title("Student Performance Indicators")

        graph_path = "static/graph.png"
        os.makedirs("static", exist_ok=True)
        plt.savefig(graph_path)
        plt.close()

    else:
        graph_path = None

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk=risk,
        performance_score=performance_score,
        recommendation=recommendation,
        study_plan=study_plan,
        graph=graph_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)