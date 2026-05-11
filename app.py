import re
import threading
import subprocess
import uuid
import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import config

app = Flask(__name__)

# Basic storage setup
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# In-memory session tracking for progress
tasks = {}

def process_video(task_id, filename):
    # Use absolute paths for robust resolution in background threads
    input_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    out_dir = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], task_id))
    os.makedirs(out_dir, exist_ok=True)
    
    tasks[task_id] = {
        'status': 'processing', 
        'progress': 0, 
        'events': 0, 
        'clips': [], 
        'log': [], 
        'error': None
    }
    
    def my_callback(info):
        if 'progress' in info:
            tasks[task_id]['progress'] = info['progress']
        if 'events' in info:
            tasks[task_id]['events'] = info['events']
        if 'log' in info:
            tasks[task_id]['log'].append(info['log'])
            if len(tasks[task_id]['log']) > 50:
                tasks[task_id]['log'].pop(0)

    try:
        from main import run as run_main
        run_main(
            input_path=input_path,
            out_folder=out_dir,
            device=config.YOLO_DEVICE,
            progress_callback=my_callback
        )
        tasks[task_id]['progress'] = 100
        tasks[task_id]['status'] = 'completed'
        
        # Read event_log.txt for stats
        log_path = os.path.join(out_dir, "event_log.txt")
        stats = {}
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
                processed_match = re.search(r'Processed\s*:\s*([\d:]+)', content)
                if processed_match:
                    stats['original_length'] = processed_match.group(1)
                
                extracted_match = re.search(r'Extracted\s*:\s*([\d:]+)', content)
                if extracted_match:
                    stats['extracted_length'] = extracted_match.group(1)
                
                input_mb_match = re.search(r'Input video\s*:\s*([\d\.]+\s*MB)', content)
                if input_mb_match:
                    stats['input_size'] = input_mb_match.group(1)
                    
                clips_mb_match = re.search(r'All clips\s*:\s*([\d\.]+\s*MB)', content)
                if clips_mb_match:
                    stats['output_size'] = clips_mb_match.group(1)
                    
                reduction_match = re.search(r'Size reduction\s*:\s*([\d\.]+%)', content)
                if reduction_match:
                    stats['reduction'] = reduction_match.group(1)

        tasks[task_id]['stats'] = stats

        # Look for clips
        clips = []
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith('.mp4'):
                    clips.append(f)
        tasks[task_id]['clips'] = clips
            
    except BaseException as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = error_msg
        print("ERROR IN THREAD:", error_msg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        task_id = str(uuid.uuid4())
        
        # Use a unique filename
        ext = os.path.splitext(filename)[1]
        unique_filename = f"{task_id}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Start background processing
        
        # Save metadata for history
        out_dir = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], task_id))
        os.makedirs(out_dir, exist_ok=True)
        import json
        from datetime import datetime
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'original_filename': file.filename,
                'timestamp': datetime.now().isoformat()
            }, f)
            
        thread = threading.Thread(target=process_video, args=(task_id, unique_filename))
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Upload successful', 'task_id': task_id})

@app.route('/progress/<task_id>')
def progress(task_id):
    if task_id in tasks:
        return jsonify(tasks[task_id])
    return jsonify({'error': 'Task not found'}), 404

import json
from datetime import datetime

@app.route('/history')
def get_history():
    history = []
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        return jsonify([])
        
    for foldername in os.listdir(app.config['OUTPUT_FOLDER']):
        out_dir = os.path.join(app.config['OUTPUT_FOLDER'], foldername)
        if os.path.isdir(out_dir):
            item = {'task_id': foldername, 'clips': [], 'stats': {}, 'metadata': {}}
            
            # Read metadata
            meta_path = os.path.join(out_dir, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        item['metadata'] = json.load(f)
                except Exception:
                    pass
            
            # Read stats
            log_path = os.path.join(out_dir, "event_log.txt")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    import re
                    m = re.search(r'Processed\s*:\s*([\d:]+)', content)
                    if m: item['stats']['original_length'] = m.group(1)
                    
                    m = re.search(r'Extracted\s*:\s*([\d:]+)', content)
                    if m: item['stats']['extracted_length'] = m.group(1)
                    
                    m = re.search(r'Input video\s*:\s*([\d\.]+\s*MB)', content)
                    if m: item['stats']['input_size'] = m.group(1)
                    
                    m = re.search(r'All clips\s*:\s*([\d\.]+\s*MB)', content)
                    if m: item['stats']['output_size'] = m.group(1)
                    
                    m = re.search(r'Size reduction\s*:\s*([\d\.]+%)', content)
                    if m: item['stats']['reduction'] = m.group(1)

            # Read clips
            for f in os.listdir(out_dir):
                if f.endswith('.mp4'):
                    item['clips'].append(f)
                    
            if item['clips'] or item['stats']:
                history.append(item)
                
    # Sort history latest first
    history.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
    return jsonify(history)

import shutil

@app.route('/download/zip/<task_id>')
def download_zip(task_id):
    out_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    if not os.path.exists(out_dir):
        return "Not found", 404
        
    zip_filename = f"{task_id}_clips"
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)
    shutil.make_archive(zip_path, 'zip', out_dir)
    return send_from_directory(app.config['OUTPUT_FOLDER'], f"{zip_filename}.zip", as_attachment=True)

@app.route('/download/log/<task_id>')
def download_log(task_id):
    out_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    if not os.path.exists(os.path.join(out_dir, "event_log.txt")):
        return "Log not found", 404
    return send_from_directory(out_dir, "event_log.txt", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
