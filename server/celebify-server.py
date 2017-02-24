import time
import json
import os
import os.path
from flask import Flask, url_for, request, jsonify, render_template, flash, session, redirect, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../watch/in'
OUTPUT_FOLDER = '../watch/out'

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print 'no file part'
            return redirect('/')
        file = request.files['file']
        # if user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            print 'No selected file'
            return redirect('/')
        if file and allowed_file(file.filename):
            securedname = secure_filename(file.filename)
            print securedname
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], securedname))
            print 'file was saved'
            # change file extension for securedname to .json
            securedname = os.path.splitext(securedname)[0]+".json"
            print securedname
            return redirect(url_for('return_json', json_file=securedname))
    return render_template('index.html')

@app.route('/ranking', methods=['GET'])
def return_json():
    json_file = request.args.get('json_file')
    my_file = os.path.join(app.config['OUTPUT_FOLDER'], json_file)
    while(True):
        if os.path.isfile(my_file):
            data = json.load(open(my_file))
            # delete the json file from directory
            os.remove(my_file)
            return jsonify(data)
        else:
            time.sleep(0.01)

app.run(debug=True)
