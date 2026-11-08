document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadProgressBar = document.getElementById('upload-progress-bar');
    const uploadPercentage = document.getElementById('upload-percentage');
    
    const processingContainer = document.getElementById('processing-progress-container');
    const processingProgressBar = document.getElementById('processing-progress-bar');
    const processingPercentage = document.getElementById('processing-percentage');
    const eventsDetected = document.getElementById('events-detected');
    const videoStreamInfo = document.getElementById('video-stream-info');
    const logOutput = document.getElementById('log-output');
    
    const resultsContainer = document.getElementById('results-container');
    const clipsGrid = document.getElementById('clips-grid');

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if(files.length) handleFile(files[0]);
    });

    fileInput.addEventListener('change', function() {
        if(this.files.length) handleFile(this.files[0]);
    });

    function resetStats() {
        document.getElementById('stat-orig-len').textContent = '--';
        document.getElementById('stat-extracted-len').textContent = '--';
        document.getElementById('stat-input-size').textContent = '--';
        document.getElementById('stat-output-size').textContent = '--';
        document.getElementById('stat-reduction').textContent = '--';
        document.getElementById('stat-input-res').textContent = '--';
        document.getElementById('stat-output-res').textContent = '--';
        document.getElementById('stat-input-fps').textContent = '--';
        document.getElementById('stat-output-fps').textContent = '--';
    }

    function populateStats(stats) {
        if (!stats) {
            resetStats();
            return;
        }
        document.getElementById('stat-orig-len').textContent = stats.original_length || '--';
        document.getElementById('stat-extracted-len').textContent = stats.extracted_length || '--';
        document.getElementById('stat-input-size').textContent = stats.input_size || '--';
        document.getElementById('stat-output-size').textContent = stats.output_size || '--';
        document.getElementById('stat-reduction').textContent = stats.reduction || '--';
        document.getElementById('stat-input-res').textContent = stats.input_resolution || '--';
        document.getElementById('stat-output-res').textContent = stats.output_resolution || '--';
        document.getElementById('stat-input-fps').textContent = stats.input_fps || '--';
        document.getElementById('stat-output-fps').textContent = stats.output_fps || '--';
    }

    function handleFile(file) {
        // Reset UI
        dropZone.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        uploadProgressContainer.classList.remove('hidden');
        processingContainer.classList.add('hidden');
        if (videoStreamInfo) {
            videoStreamInfo.classList.add('hidden');
            videoStreamInfo.textContent = '--';
        }
        logOutput.innerHTML = "Waiting for output...";
        resetStats();
        
        uploadFile(file);
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                uploadProgressBar.style.width = percentComplete + '%';
                uploadPercentage.textContent = percentComplete + '%';
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                uploadProgressContainer.classList.add('hidden');
                startPollingProgress(response.task_id);
            } else {
                alert('Upload failed. ' + xhr.responseText);
                dropZone.classList.remove('hidden');
                uploadProgressContainer.classList.add('hidden');
            }
        };

        xhr.onerror = () => {
            alert('Error during upload.');
            dropZone.classList.remove('hidden');
            uploadProgressContainer.classList.add('hidden');
        };

        xhr.send(formData);
    }

    function startPollingProgress(taskId) {
        processingContainer.classList.remove('hidden');
        
        const pollInterval = setInterval(() => {
            fetch(`/progress/${taskId}`)
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        clearInterval(pollInterval);
                        alert("Error: " + data.error);
                        dropZone.classList.remove('hidden');
                        processingContainer.classList.add('hidden');
                        return;
                    }

                    // Update UI
                    processingProgressBar.style.width = data.progress + '%';
                    processingPercentage.textContent = data.progress.toFixed(1) + '%';
                    eventsDetected.textContent = `${data.events} Events Detected`;

                    if (data.input_res && data.input_fps && videoStreamInfo) {
                        videoStreamInfo.textContent = `${data.input_res} @ ${data.input_fps} → Output: ${data.output_res || data.input_res} @ ${data.output_fps || data.input_fps}`;
                        videoStreamInfo.classList.remove('hidden');
                    }
                    
                    if (data.log && data.log.length > 0) {
                        logOutput.innerHTML = data.log.join('<br>');
                        logOutput.scrollTop = logOutput.scrollHeight;
                    }

                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        
                        // Populate stats
                        populateStats(data.stats);
                        
                        // Set download links
                        document.getElementById('btn-download-zip').href = `/download/zip/${taskId}`;
                        document.getElementById('btn-download-log').href = `/download/log/${taskId}`;

                        showResults(data.clips, taskId, data.stats);
                        loadHistory(); // refresh history list
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        alert("Processing Error: " + data.error);
                        dropZone.classList.remove('hidden');
                        processingContainer.classList.add('hidden');
                    }
                })
                .catch(err => console.error(err));
        }, 1000);
    }

    function showResults(clips, taskId, stats) {
        processingContainer.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        dropZone.classList.remove('hidden'); // allow another upload
        
        clipsGrid.innerHTML = '';
        
        if (!clips || clips.length === 0) {
            clipsGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted); font-size: 1.1rem; padding: 2rem;">No key events detected in the footage.</p>';
            return;
        }

        const outRes = stats?.output_resolution || '';
        const outFps = stats?.output_fps || '';

        clips.forEach(clip => {
            const clipUrl = `/output/${taskId}/${clip}`;
            
            // Extract info from filename: event_001__00h00m10s__person_car.mp4
            let label = "Event";
            let timeStr = "";
            let components = [];
            
            if(clip.includes('__')) {
                const parts = clip.split('__');
                if (parts.length >= 3) {
                    timeStr = parts[1].replace(/h/g, ':').replace(/m/g, ':').replace(/s/g, '');
                    const labelPart = parts[2].replace('.mp4', '');
                    components = labelPart.split('_');
                }
            }

            const badgesHtml = components.map(c => `<span class="badge">${c.toUpperCase()}</span>`).join('');
            const resBadgeHtml = outRes ? `<span class="badge badge-tech">${outRes}</span>` : '';
            const fpsBadgeHtml = outFps ? `<span class="badge badge-tech">${outFps}</span>` : '';

            const card = document.createElement('div');
            card.className = 'clip-card';
            card.innerHTML = `
                <video class="clip-video" controls preload="metadata">
                    <source src="${clipUrl}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="clip-info">
                    <div class="clip-title">${clip.replace(/__/g, ' - ').replace(/_/g, ' ')}</div>
                    <div class="badges-row">
                        ${badgesHtml || '<span class="badge">EVENT</span>'}
                        ${resBadgeHtml}
                        ${fpsBadgeHtml}
                        <a href="${clipUrl}" download class="badge badge-download" style="margin-left: auto;">Download</a>
                    </div>
                </div>
            `;
            clipsGrid.appendChild(card);
        });
    }

    // --- History Appending ---
    const historyList = document.getElementById('history-list');

    function loadHistory() {
        fetch('/history')
            .then(res => res.json())
            .then(data => {
                if(data.length === 0) {
                    historyList.innerHTML = '<p style="text-align:center; color:var(--text-muted);">No previously processed videos yet.</p>';
                    return;
                }
                
                historyList.innerHTML = '';
                data.forEach(item => {
                    const originalName = item.metadata.original_filename || item.task_id;
                    const dateStr = item.metadata.timestamp ? new Date(item.metadata.timestamp).toLocaleString() : 'Unknown Date';
                    const numClips = item.clips.length;
                    const resBadge = item.stats?.output_resolution ? `<div class="hist-stat-badge">${item.stats.output_resolution}</div>` : '';
                    const fpsBadge = item.stats?.output_fps ? `<div class="hist-stat-badge">${item.stats.output_fps}</div>` : '';
                    
                    const card = document.createElement('div');
                    card.className = 'history-card';
                    card.innerHTML = `
                        <div class="hist-info">
                            <h4>${originalName}</h4>
                            <div class="hist-sub">${dateStr} &nbsp;|&nbsp; Task: ${item.task_id.substring(0,8)}...</div>
                        </div>
                        <div class="hist-stats">
                            <div class="hist-stat-badge">${numClips} Clips</div>
                            ${resBadge}
                            ${fpsBadge}
                            <div class="hist-stat-badge">${item.stats?.output_size || '-- MB'}</div>
                            <div class="hist-stat-badge" style="color:var(--success); font-weight: bold;">${item.stats?.reduction || '--%'} Reduced</div>
                            <button class="btn-secondary" style="padding: 0.4rem 1rem; font-size: 0.8rem;">View</button>
                        </div>
                    `;
                    
                    card.addEventListener('click', () => {
                        populateStats(item.stats);
                        
                        // Set download links
                        document.getElementById('btn-download-zip').href = `/download/zip/${item.task_id}`;
                        document.getElementById('btn-download-log').href = `/download/log/${item.task_id}`;
                        
                        showResults(item.clips, item.task_id, item.stats);
                        
                        // Scroll to results
                        document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
                    });
                    
                    historyList.appendChild(card);
                });
            })
            .catch(err => console.error(err));
    }

    loadHistory();
});
