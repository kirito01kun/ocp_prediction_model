from flask import Flask, render_template, request, redirect, url_for, session
from processing import *
from visualization import *
from werkzeug.utils import secure_filename
import atexit
import shutil
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/koch')
def koch():
    return render_template('koch.html')


@app.route('/krupp')
def krupp():
    return render_template('krupp.html')

@app.route('/krupp_processing', methods=['POST'])
def krupp_processing():
    if 'upload' in request.form and 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file:
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            print(uploaded_file_path)
            
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            
            uploaded_file.save(uploaded_file_path)
            data = krupp_pre_pro(uploaded_file_path)
            display_data(app, data, 'models/krupp.h5')
            ts_decomp_graph(app, data, 'models/krupp.h5')
            rolling_stats(app, data, 'models/krupp.h5')
            train, valid = preforcast(app, data, 'models/krupp.h5')
            model_test_graph(app, train, valid, 'models/krupp.h5')
            future_dates, future_predictions = forcast_peocessing(app, data, 'models/krupp.h5')
            
            forcast_graph(app, train, valid, future_dates, future_predictions, 'models/krupp.h5')



    session['graph_displayed'] = True  # Set the session value to indicate the graph has been displayed
    return render_template('krupp.html')


@app.route('/koch_processing', methods=['POST'])
def koch_processing():
    if 'upload' in request.form and 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file:
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            print(uploaded_file_path)
            
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            
            uploaded_file.save(uploaded_file_path)
            data = koch_pre_pro(uploaded_file_path)
            display_data(app, data, 'models/koch.h5')
            ts_decomp_graph(app, data, 'models/koch.h5')
            rolling_stats(app, data, 'models/koch.h5')
            train, valid = preforcast(app, data, 'models/koch.h5')
            model_test_graph(app, train, valid, 'models/koch.h5')
            future_dates, future_predictions = forcast_peocessing(app, data, 'models/koch.h5')
            
            forcast_graph(app, train, valid, future_dates, future_predictions, 'models/koch.h5')



    session['graph_displayed'] = True  # Set the session value to indicate the graph has been displayed
    return render_template('koch.html')



@app.route('/krupp_2023')
def krupp_2023():
    file_path = os.path.join(os.path.dirname(__file__), 'R0_KRUPP_2023.xlsx')
    data = krupp_pre_pro(file_path)
    display_data(app, data, 'models/krupp.h5')
    ts_decomp_graph(app, data, 'models/krupp.h5')
    rolling_stats(app, data, 'models/krupp.h5')
    train, valid = preforcast(app, data, 'models/krupp.h5')
    model_test_graph(app, train, valid, 'models/krupp.h5')
    future_dates, future_predictions = forcast_peocessing(app, data, 'models/krupp.h5')
    forcast_graph(app, train, valid, future_dates, future_predictions, 'models/krupp.h5')
    session['graph_displayed'] = True  # Set the session value to indicate the graph has been displayed
    return render_template('krupp.html')


@app.route('/koch_2023')
def koch_2023():
    file_path = os.path.join(os.path.dirname(__file__), 'R0_koch_2023.xlsm')
    data = koch_pre_pro(file_path)
    display_data(app, data, 'models/koch.h5')
    ts_decomp_graph(app, data, 'models/koch.h5')
    rolling_stats(app, data, 'models/koch.h5')
    train, valid = preforcast(app, data, 'models/koch.h5')
    model_test_graph(app, train, valid, 'models/koch.h5')
    future_dates, future_predictions = forcast_peocessing(app, data, 'models/koch.h5')
    forcast_graph(app, train, valid, future_dates, future_predictions, 'models/koch.h5')
    session['graph_displayed'] = True  # Set the session value to indicate the graph has been displayed
    return render_template('koch.html')



def cleanup_images():
    graphs_dir = os.path.join(app.static_folder, 'graphs')
    if os.path.exists(graphs_dir):
        for item in os.listdir(graphs_dir):
            item_path = os.path.join(graphs_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

atexit.register(cleanup_images)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)

     