from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import cloudProcessing
import logging, json, os, time, sys

api = Blueprint("api", __name__, url_prefix="/api")
cloudDatasetPath = os.path.join(os.path.dirname(__file__), "cloudDataset")

"""
Input: json data
Objective: receives data from edge node
"""
@api.route('/edgearch/cloud', methods = ['POST'])
def edgearch_cloud():

	data = request.json
	
	result = cloudProcessing.uploadImgData(data)

	if(result['status'] == 'ok'):
		return jsonify(result), 200
	else:
		return jsonify(result), 500
		

