from flask import Flask,request,jsonify,render_template
from src.pipelines.prediction_pipeline import CustomData , PredictionPipeline

application = Flask(__name__)

app = application

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    
    else :
        carat = float(request.form.get("carat"))
        depth = float(request.form.get("depth"))
        table = float(request.form.get("table"))
        x = float(request.form.get("x"))
        y = float(request.form.get("y"))
        z = float(request.form.get("z"))
        cut = str(request.form.get("cut"))
        color = str(request.form.get("color"))
        clarity = str(request.form.get("clarity"))

        custom = CustomData(carat,cut,color,clarity,depth,table,x,y,z)
        dataframe = custom.get_data_as_dataframe()

        model = PredictionPipeline()
        pred = model.predict(dataframe)
        final_result = round(pred[0],2)
        return render_template("form.html",final_result = final_result)
    

if __name__ == "__main__" :
    app.run(host="0.0.0.0",debug=True)
