from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import edgeProcessing
import logging, json, os, time, sys

api = Blueprint("api", __name__, url_prefix="/api")

@api.route('/edgearch/edge', methods = ['POST'])
def edge():
	fileImg = request.files["file"]
	result = edgeProcessing.receiveData(fileImg)

	if(result['status'] == 'ok'):
		return jsonify(result), 200
	else:
		return jsonify(result), 500


@api.route('/edgearch/setedgecache', methods = ['POST'])
def setedgecache():

	fileImg = request.json
	result = edgeProcessing.setCache(fileImg)

	if(result['status'] == 'ok'):
		return jsonify(result), 200
	else:
		return jsonify(result), 500
