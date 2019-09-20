from flask import Flask, request, render_template

from .recommender import Recommender

app = Flask(__name__)

@app.route("/")
def index():    
    return render_template("input.html", categories = engine.contexts)
  
@app.route("/result/", methods = ["POST", "GET"])
def result():
    if request.method == "POST":
        result = request.form
        
        try:
            user_id = int(result["UserID"])
        except ValueError:
            user_id = None
            
        if user_id not in engine.user_id_map:
            user_id = None
        
        # Compute recommendations for the given user and context/category
        rec_items = engine.recommend(user_id, result["ctx"], result["cat"])
        tracks = engine.tracks[engine.tracks["id"].isin([ri[1] for ri in rec_items])][[" artist", " title"]].values.tolist()
        
        return render_template("output.html", user_id = user_id, result = result, tracks = tracks)

@app.route("/metrics/")
def metrics():
    return render_template("metrics.html", metrics = engine.get_accuracy_metrics())

# Initialize the recommender engine
# NOTE: use the below call to re-train model:
# engine = Recommender(data_file = "../data/Data_InCarMusic.xlsx", train_epochs=25, method="adam")

# NOTE: use the below call if you have a pre-trained model:
engine = Recommender(data_file = None, model_file = "model.pickle")

