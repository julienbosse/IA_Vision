from flask import render_template, request, abort, redirect, url_for, Flask, Response
from app import app
from app import models


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/inversion', methods=['GET', 'POST'])
def inversion():
    return render_template('inversion.html')


@app.route('/video_feed')
def video_feed():
    return Response(models.gen_frames_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/inversion_feed')
def inversion_feed():
    return Response(models.gen_frames_inversion(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/inversion-swapp', methods=['GET', 'POST'])
def inversion_swapp():
    return render_template('inversion_swapp.html')


@app.route('/inversion_swapping')
def inversion_swapping():
    return Response(models.gen_video_inversion_swapping(), mimetype='multipart/x-mixed-replace; boundary=frame')
