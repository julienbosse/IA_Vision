from flask import render_template, request, abort, redirect, url_for, Flask, Response
from app import app
from app import models

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(models.gen_frames_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')