from flask import Blueprint, render_template, request
import tensorflow as tf
import re
import base64
import numpy as np 
import os
from werkzeug.utils import secure_filename

upload_api = Blueprint('upload_api', __name__)

