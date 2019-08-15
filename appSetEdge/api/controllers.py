from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import edgeProcessing
import logging, json, os, time, sys

api = Blueprint("api", __name__, url_prefix="/api")
edgeDatasetPath = os.path.join(os.path.dirname(__file__), "edgeDataset")


@api.route('/edgearch/edgesetcheck', methods = ['POST'])
def edgesetcheck():
	fileJson = request.json
	result = edgeProcessing.checkCache(fileJson)
	if(result['status'] == 'ok'):
		return jsonify(result), 200 
	else:
		return jsonify(result), 500


@api.route('/edgearch/edgesetcache', methods = ['POST'])
def edgesetcache():
	fileImg = request.files["file"]
	result = edgeProcessing.setCache(fileImg)
	if(result['status'] == 'ok'):
		return jsonify(result), 200 
	else:
		return jsonify(result), 500
