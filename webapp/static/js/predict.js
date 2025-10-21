// File Upload Based Prediction System - Real ML Feature Extraction

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Store extracted features
let extractedFeatures = {
    speech: null,
    handwriting: null,
    gait: null
};

$(document).ready(function() {
    // Voice recording
    $('#recordBtn').click(function() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
    
    // File upload buttons
    $('#uploadAudioBtn').click(function() {
        uploadAudioFile();
    });
    
    $('#uploadHandwritingBtn').click(function() {
        uploadHandwritingFile();
    });
    
    $('#uploadGaitBtn').click(function() {
        uploadGaitFile();
    });
    
    // Example buttons
    $('#useAudioExample').click(function() {
        useExampleAudio();
    });
    
    $('#useHandwritingExample').click(function() {
        useExampleHandwriting();
    });
    
    $('#useGaitExample').click(function() {
        useExampleGait();
    });
    
    // Image preview
    $('#handwritingFileInput').change(function() {
        previewHandwritingImage(this);
    });
    
    // Predict button
    $('#predictBtn').click(function() {
        makePrediction();
    });
    
    // Reset button
    $('#resetBtn').click(function() {
        resetForm();
    });
});

// Voice Recording Functions
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            uploadRecordedAudio(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        $('#recordBtn').removeClass('btn-danger').addClass('btn-warning');
        $('#recordBtnText').text('Stop Recording');
        $('#recordingStatus').show();
        
        showNotification('🎤 Recording started! Say "Aaaaahhh" for 3-5 seconds', 'info');
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showNotification('❌ Could not access microphone. Please check permissions.', 'danger');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        $('#recordBtn').removeClass('btn-warning').addClass('btn-danger');
        $('#recordBtnText').text('Start Recording');
        $('#recordingStatus').hide();
        
        showNotification('✅ Recording stopped. Processing audio...', 'success');
    }
}

// Upload recorded audio
function uploadRecordedAudio(audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');
    
    uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus');
}

// Upload audio file
function uploadAudioFile() {
    const fileInput = document.getElementById('audioFileInput');
    if (!fileInput.files.length) {
        showNotification('⚠️ Please select an audio file first', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    $('#audioUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    
    uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus');
}

// Upload handwriting image
function uploadHandwritingFile() {
    const fileInput = document.getElementById('handwritingFileInput');
    if (!fileInput.files.length) {
        showNotification('⚠️ Please select an image file first', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    $('#handwritingUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    
    uploadFile('/api/upload/handwriting', formData, 'handwriting', 'handwritingFeatureStatus');
}

// Upload gait video
function uploadGaitFile() {
    const fileInput = document.getElementById('gaitFileInput');
    if (!fileInput.files.length) {
        showNotification('⚠️ Please select a video file first', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    $('#gaitUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    
    uploadFile('/api/upload/gait', formData, 'gait', 'gaitFeatureStatus');
}

// Generic file upload function
function uploadFile(endpoint, formData, modality, statusElementId) {
    $.ajax({
        url: endpoint,
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.success) {
                // Store extracted features
                extractedFeatures[modality] = response.features;
                
                // Update hidden input
                $('#' + modality + 'Features').val(response.features.join(','));
                
                // Show success message
                const modalityIcon = {
                    'speech': '🎤',
                    'handwriting': '✍️',
                    'gait': '🚶'
                };
                
                $('#' + statusElementId).html(
                    '<div class="alert alert-success">' +
                    '<i class="fas fa-check-circle"></i> <strong>Success!</strong><br>' +
                    modalityIcon[modality] + ' Extracted ' + response.feature_count + ' features<br>' +
                    '<small>' + response.message + '</small>' +
                    (response.note ? '<br><small class="text-muted">' + response.note + '</small>' : '') +
                    '</div>'
                );
                
                // Clear upload status
                const uploadStatusId = modality === 'speech' ? 'audioUploadStatus' :
                                      modality === 'handwriting' ? 'handwritingUploadStatus' :
                                      'gaitUploadStatus';
                $('#' + uploadStatusId).html('');
                
                // Enable predict button
                updatePredictButton();
                
                showNotification('✅ Features extracted successfully!', 'success');
            } else {
                showNotification('❌ ' + response.error, 'danger');
                clearUploadStatus(modality);
            }
        },
        error: function(xhr, status, error) {
            const errorMsg = xhr.responseJSON && xhr.responseJSON.error ? 
                            xhr.responseJSON.error : 'Upload failed';
            showNotification('❌ ' + errorMsg, 'danger');
            clearUploadStatus(modality);
        }
    });
}

// Clear upload status
function clearUploadStatus(modality) {
    const uploadStatusId = modality === 'speech' ? 'audioUploadStatus' :
                          modality === 'handwriting' ? 'handwritingUploadStatus' :
                          'gaitUploadStatus';
    $('#' + uploadStatusId).html('');
}

// Preview handwriting image
function previewHandwritingImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            $('#handwritingImg').attr('src', e.target.result);
            $('#handwritingPreview').show();
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// Update predict button state
function updatePredictButton() {
    const hasAnyFeatures = extractedFeatures.speech !== null || 
                          extractedFeatures.handwriting !== null || 
                          extractedFeatures.gait !== null;
    
    $('#predictBtn').prop('disabled', !hasAnyFeatures);
}

// Make prediction
function makePrediction() {
    // Build request with extracted features
    const requestData = {};
    const modalitiesUsed = [];
    let totalFeatures = 0;
    
    if (extractedFeatures.speech) {
        requestData.speech_features = extractedFeatures.speech;
        modalitiesUsed.push('<span class="badge bg-primary"><i class="fas fa-microphone"></i> Speech (22)</span>');
        totalFeatures += 22;
    }
    
    if (extractedFeatures.handwriting) {
        requestData.handwriting_features = extractedFeatures.handwriting;
        modalitiesUsed.push('<span class="badge bg-success"><i class="fas fa-pen"></i> Handwriting (10)</span>');
        totalFeatures += 10;
    }
    
    if (extractedFeatures.gait) {
        requestData.gait_features = extractedFeatures.gait;
        modalitiesUsed.push('<span class="badge bg-warning text-dark"><i class="fas fa-walking"></i> Gait (10)</span>');
        totalFeatures += 10;
    }
    
    if (Object.keys(requestData).length === 0) {
        showNotification('⚠️ Please upload at least one file first!', 'warning');
        return;
    }
    
    console.log('Making prediction with extracted features:', Object.keys(requestData));
    console.log('Total features:', totalFeatures);
    
    // Show loading
    $('#placeholderSection').hide();
    $('#resultsSection').hide();
    $('#loadingSection').show();
    $('#loadingText').html(
        'Analyzing <strong>' + totalFeatures + '</strong> features from <strong>' + 
        modalitiesUsed.length + '</strong> modality/modalities<br>' +
        modalitiesUsed.join(' ')
    );
    
    const startTime = Date.now();
    
    // Make API request
    $.ajax({
        url: '/api/predict',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function(response) {
            const elapsedTime = Date.now() - startTime;
            const remainingTime = Math.max(0, 1000 - elapsedTime);
            
            setTimeout(function() {
                if (response.success) {
                    displayResults(response, modalitiesUsed, totalFeatures);
                    showNotification('✅ Prediction complete!', 'success');
                } else {
                    showNotification('❌ Prediction failed: ' + response.error, 'danger');
                    $('#loadingSection').hide();
                    $('#placeholderSection').show();
                }
            }, remainingTime);
        },
        error: function(xhr, status, error) {
            setTimeout(function() {
                const errorMsg = xhr.responseJSON && xhr.responseJSON.error ? 
                                xhr.responseJSON.error : 'Prediction failed';
                showNotification('❌ ' + errorMsg, 'danger');
                $('#loadingSection').hide();
                $('#placeholderSection').show();
            }, 500);
        }
    });
}

// Display results
function displayResults(response, modalitiesUsed, totalFeatures) {
    $('#loadingSection').hide();
    
    const prediction = response.prediction;
    const confidence = response.confidence;
    
    // Update prediction result
    if (prediction === 1) {
        $('#predictionAlert').removeClass('alert-success').addClass('alert-warning');
        $('#predictionLabel').html('<i class="fas fa-exclamation-triangle"></i> Parkinson\'s Disease Detected');
        $('#predictionText').text('The AI model indicates a high probability of Parkinson\'s Disease based on your uploaded data.');
    } else {
        $('#predictionAlert').removeClass('alert-warning').addClass('alert-success');
        $('#predictionLabel').html('<i class="fas fa-check-circle"></i> Healthy');
        $('#predictionText').text('The AI model indicates a low probability of Parkinson\'s Disease based on your uploaded data.');
    }
    
    // Show modalities
    $('#modalitiesUsed').html(
        modalitiesUsed.join(' ') + 
        '<br><small class="text-muted">Total: ' + totalFeatures + ' features automatically extracted</small>'
    );
    
    // Update confidence bar
    const confidencePercent = (confidence * 100).toFixed(2);
    $('#confidenceBar').css('width', confidencePercent + '%');
    $('#confidenceText').text(confidencePercent + '%');
    
    if (confidence >= 0.8) {
        $('#confidenceBar').removeClass('bg-warning bg-info').addClass('bg-success');
    } else if (confidence >= 0.6) {
        $('#confidenceBar').removeClass('bg-success bg-danger').addClass('bg-info');
    } else {
        $('#confidenceBar').removeClass('bg-success bg-info').addClass('bg-warning');
    }
    
    // Update probabilities
    $('#healthyProb').text((response.probabilities.healthy * 100).toFixed(2) + '%');
    $('#parkinsonsProb').text((response.probabilities.parkinsons * 100).toFixed(2) + '%');
    
    // Show results
    $('#resultsSection').fadeIn('slow');
}

// Reset form
function resetForm() {
    // Clear file inputs
    $('#audioFileInput').val('');
    $('#handwritingFileInput').val('');
    $('#gaitFileInput').val('');
    
    // Clear hidden inputs
    $('#speechFeatures').val('');
    $('#handwritingFeatures').val('');
    $('#gaitFeatures').val('');
    
    // Clear extracted features
    extractedFeatures = {
        speech: null,
        handwriting: null,
        gait: null
    };
    
    // Clear status displays
    $('#speechFeatureStatus').html('');
    $('#handwritingFeatureStatus').html('');
    $('#gaitFeatureStatus').html('');
    $('#audioUploadStatus').html('');
    $('#handwritingUploadStatus').html('');
    $('#gaitUploadStatus').html('');
    
    // Hide handwriting preview
    $('#handwritingPreview').hide();
    
    // Stop recording if active
    if (isRecording) {
        stopRecording();
    }
    
    // Disable predict button
    $('#predictBtn').prop('disabled', true);
    
    // Hide results
    $('#resultsSection').hide();
    $('#loadingSection').hide();
    $('#placeholderSection').show();
    
    // Switch to first tab
    $('#speech-tab').tab('show');
    
    showNotification('🔄 Form reset - ready for new files', 'info');
}

// ===== EXAMPLE FUNCTIONS =====

// Use Audio Example
function useExampleAudio() {
    showNotification('📥 Loading audio example...', 'info');
    $('#audioUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Loading example...</div>');
    
    fetch('/static/examples/example_audio.wav')
        .then(response => response.blob())
        .then(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'example_audio.wav');
            uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus');
        })
        .catch(error => {
            showNotification('❌ Error loading example: ' + error, 'danger');
            $('#audioUploadStatus').html('');
        });
}

// Use Handwriting Example
function useExampleHandwriting() {
    showNotification('📥 Loading handwriting example...', 'info');
    $('#handwritingUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Loading example...</div>');
    
    fetch('/static/examples/example_handwriting.jpg')
        .then(response => response.blob())
        .then(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'example_handwriting.jpg');
            uploadFile('/api/upload/handwriting', formData, 'handwriting', 'handwritingFeatureStatus');
        })
        .catch(error => {
            showNotification('❌ Error loading example: ' + error, 'danger');
            $('#handwritingUploadStatus').html('');
        });
}

// Use Gait Example
function useExampleGait() {
    showNotification('📥 Loading gait example...', 'info');
    $('#gaitUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Loading example...</div>');
    
    fetch('/static/examples/example_gait.mp4')
        .then(response => response.blob())
        .then(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'example_gait.mp4');
            uploadFile('/api/upload/gait', formData, 'gait', 'gaitFeatureStatus');
        })
        .catch(error => {
            showNotification('❌ Error loading example: ' + error, 'danger');
            $('#gaitUploadStatus').html('');
        });
}

// View Handwriting Example in Modal
function viewExampleHandwriting() {
    $('#exampleModalTitle').text('Handwriting Example - Spiral Drawing');
    $('#examplePreviewContent').html(
        '<img src="/static/examples/example_handwriting.jpg" class="img-fluid" alt="Handwriting example">' +
        '<p class="mt-3 text-muted">This is an Archimedes spiral, commonly used in Parkinson\'s disease assessment.</p>'
    );
    const modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

// View Gait Example in Modal
function viewExampleGait() {
    $('#exampleModalTitle').text('Gait Example - Walking Analysis');
    $('#examplePreviewContent').html(
        '<video controls class="w-100"><source src="/static/examples/example_gait.mp4" type="video/mp4"></video>' +
        '<p class="mt-3 text-muted">This animation demonstrates a side-view walking pattern for gait analysis.</p>'
    );
    const modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

console.log('✅ File Upload Prediction System loaded - Real ML feature extraction enabled!');
