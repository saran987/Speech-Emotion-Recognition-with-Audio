# Package import
from re import M
from flask import *  
from werkzeug.utils import secure_filename
from Main import output_predictions
import os

# Creating the Application to run the project
app = Flask(__name__)

# File upload location
UPLOAD_FOLDER = 'Speech-Emotion-Recognition-with-Audio/upload'
filename = ''

# Configuring Folder to application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Methods to take input and display the output
@app.route("/", methods=['POST', 'GET'])

def index():

    if request.method == "POST":

        f1 = request.files['audio_data']

        with open('Speech-Emotion-Recognition-with-Audio/upload/audio_sample.wav', 'wb') as audio:

            f1.save(audio)

        return render_template('Index.html', request="POST", visibility1="none")

    else:

        return render_template("Index.html", visibility1="none")
    
@app.route("/success1", methods=['POST'])

def success1():

        if request.method == 'POST':  

            f2 = request.files['file']  

            f2.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f2.filename)))

            a,b,c,d,e,f,g,h,i,j,k,l = output_predictions()

            return render_template("Index.html", visibility1 = "block",  pred1 = a, pred2 = b, pred3 =  c, pred4 = d, pred5 = e, pred6 = f, pred7 = g, pred8 = h, pred9 = i, pred10 = j,  pred11 = k,  pred12 = l,  request="POST")

@app.route("/success3", methods=['POST'])

def success3():

    a,b,c,d,e,f,g,h,i,j,k,l = output_predictions()


    return render_template("Index.html", visibility1 = "block",  pred1 = a, pred2 = b, pred3 =  c, pred4 = d, pred5 = e, pred6 = f, pred7 = g, pred8 = h, pred9 = i, pred10 = j,  pred11 = k,  pred12 = l,  request="POST")
                
if __name__ == "__main__":
    app.run(debug=True)
