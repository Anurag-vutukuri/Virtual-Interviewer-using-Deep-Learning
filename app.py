from flask import Flask, render_template, Response, make_response, send_file
import audio_recognizer
import json
import video_recognizer
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__, static_folder='static')

global audio_emotion
global multi_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_recognizer.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/multimodal1')
def multimodal1():
    return render_template('question.html')

@app.route('/multimodal')
def multimodal():
    return render_template('multimodal.html')

@app.route('/question2html')
def question2html():
    return render_template('question2.html')

@app.route('/questio2')
def questio2():
    return render_template('multimodal2.html')

@app.route('/live-data_multi')
def live_data_multi():
    """ This function is called from graph_multi.js that updates the multimodal diagram.
    It fetches predictions, updates global variables and returns a multimodal prediction. """
    global multi_emotion
    f = open('video_prediction.json')
    video_data = json.load(f)
    audio_data = audio_recognizer.analyze_audio()
    newdict = [{"name": "Video",
                "data": [{"name": "Angry", "value": round(video_data[0] * 100, 2)},
                         {"name": "Fearful", "value": round(video_data[1] * 100, 2)},
                         {"name": "Happy", "value": round(video_data[2] * 100, 2)},
                         {"name": "Sad", "value": round(video_data[3] * 100, 2)}]},
               {"name": "Audio",
                "data": [{"name": "Angry", "value": round(audio_data[0] * 100, 2)},
                         {"name": "Fearful", "value": round(audio_data[1] * 100, 2)},
                         {"name": "Happy", "value": round(audio_data[2] * 100, 2)},
                         {"name": "Sad", "value": round(audio_data[3] * 100, 2)}]}]
    multimodal_data = []
    video_data = np.array(video_data)
    for i in range(len(audio_data)):
        multimodal_data.append(0.7 * video_data[i] + 0.3 * audio_data[i])
    multi_emotion = multimodal_data

    response = make_response(json.dumps(newdict))
    response.content_type = 'application/json'
    return response
  
@app.route('/multi_emotion')
def multi_emotion():
    try:
        data = multi_emotion
    except NameError:
        data = [0.0, 0.0, 0.0, 0.0]
    print('multi array: ', data)
    if np.argmax(data) == 0:
        return send_file('static\\images\\angry.jpg', mimetype='image/jpg')
    elif np.argmax(data) == 1:
        return send_file('static\\images\\fear.jpg', mimetype='image/jpg')
    elif np.argmax(data) == 2:
        return send_file('static\\images\\happy.jpg', mimetype='image/jpg')
    elif np.argmax(data) == 3:
        return send_file('static\\images\\sad.jpg', mimetype='image/jpg')
    else:
        return send_file('static\\images\\neutral.jpg', mimetype='image/jpg')
    
if __name__ == '__main__':
    app.run(debug=False)
