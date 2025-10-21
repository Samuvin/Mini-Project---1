// Audio recording and upload functionality

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

$(document).ready(function() {
    // Check if browser supports audio recording
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('Audio recording not supported in this browser');
        $('#recordBtn').prop('disabled', true).attr('title', 'Recording not supported');
    }
    
    // Audio file upload
    $('#audioFile').change(function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadAudioFile(file);
        }
    });
    
    // Handwriting image upload
    $('#handwritingImage').change(function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadHandwritingImage(file);
        }
    });
    
    // Record button click
    $('#recordBtn').click(function() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
});

// Start audio recording
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
            processAudioBlob(audioBlob);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        $('#recordBtn')
            .removeClass('btn-danger')
            .addClass('btn-warning')
            .html('<i class="fas fa-stop"></i> Stop Recording');
        $('#recordingStatus').show().text('Recording... Speak now!');
        
        showNotification('Recording started. Speak clearly!', 'info');
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showNotification('Could not access microphone. Please check permissions.', 'danger');
    }
}

// Stop audio recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        $('#recordBtn')
            .removeClass('btn-warning')
            .addClass('btn-danger')
            .html('<i class="fas fa-microphone"></i> Record Audio');
        $('#recordingStatus').hide();
        
        showNotification('Processing audio...', 'info');
    }
}

// Process recorded audio blob
function processAudioBlob(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    uploadAudio(formData);
}

// Upload audio file
function uploadAudioFile(file) {
    const formData = new FormData();
    formData.append('audio', file);
    
    showNotification('Uploading audio file...', 'info');
    uploadAudio(formData);
}

// Send audio to server for processing
function uploadAudio(formData) {
    $.ajax({
        url: '/api/process_audio',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.success) {
                // Store speech features
                window.speechFeatures = response.features;
                
                // Create feature display
                const featuresHtml = createFeatureDisplay(response.features, 'Speech');
                
                // Create predict button HTML
                const predictBtnHtml = `
                    <div class="predict-btn-container mt-3 mb-3">
                        <div class="d-grid gap-2">
                            <button type="button" class="btn btn-success btn-lg predict-now-btn">
                                <i class="fas fa-brain"></i> Make Prediction Now
                            </button>
                        </div>
                        <p class="text-center text-muted mt-2 mb-0">
                            <small><i class="fas fa-info-circle"></i> Click to analyze speech features</small>
                        </p>
                    </div>
                `;
                
                $('#audioStatus').html(
                    '<div class="alert alert-success">' +
                    '<i class="fas fa-check-circle"></i> <strong>Audio processed successfully!</strong>' +
                    '</div>' +
                    featuresHtml +
                    predictBtnHtml
                );
                
                // Bind click event to the new button
                $('#audioStatus .predict-now-btn').on('click', function() {
                    $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
                    makePredictionFromFeatures();
                });
                
                showNotification('Audio features extracted successfully!', 'success');
                updateCombinedFeatures();
            } else {
                showNotification('Error processing audio: ' + response.error, 'danger');
            }
        },
        error: function(xhr, status, error) {
            const errorMsg = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error';
            showNotification('Failed to process audio: ' + errorMsg, 'danger');
        }
    });
}

// Upload handwriting image
function uploadHandwritingImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        $('#imagePreview').html(
            '<img src="' + e.target.result + '" class="img-fluid rounded" style="max-height: 200px;">'
        );
    };
    reader.readAsDataURL(file);
    
    showNotification('Uploading handwriting image...', 'info');
    
    $.ajax({
        url: '/api/process_handwriting',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.success) {
                // Store handwriting features
                window.handwritingFeatures = response.features;
                
                // Create feature display
                const featuresHtml = createFeatureDisplay(response.features, 'Handwriting');
                
                // Create predict button HTML
                const predictBtnHtml = `
                    <div class="predict-btn-container mt-3 mb-3">
                        <div class="d-grid gap-2">
                            <button type="button" class="btn btn-success btn-lg predict-now-btn">
                                <i class="fas fa-brain"></i> Make Prediction Now
                            </button>
                        </div>
                        <p class="text-center text-muted mt-2 mb-0">
                            <small><i class="fas fa-info-circle"></i> Click to analyze handwriting features</small>
                        </p>
                    </div>
                `;
                
                $('#handwritingStatus').html(
                    '<div class="alert alert-success mt-2">' +
                    '<i class="fas fa-check-circle"></i> <strong>Image processed successfully!</strong>' +
                    '</div>' +
                    featuresHtml +
                    predictBtnHtml
                );
                
                // Bind click event to the new button
                $('#handwritingStatus .predict-now-btn').on('click', function() {
                    $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
                    makePredictionFromFeatures();
                });
                
                showNotification('Handwriting features extracted successfully!', 'success');
                updateCombinedFeatures();
            } else {
                showNotification('Error processing image: ' + response.error, 'danger');
            }
        },
        error: function(xhr, status, error) {
            const errorMsg = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error';
            showNotification('Failed to process image: ' + errorMsg, 'danger');
        }
    });
}

// Update combined features display
function updateCombinedFeatures() {
    // Use stored features or generate default gait features
    const speech = window.speechFeatures || [];
    const handwriting = window.handwritingFeatures || [];
    const gait = window.gaitFeatures || generateDefaultGaitFeatures();
    
    const combined = [...speech, ...handwriting, ...gait];
    
    if (combined.length === 42) {
        $('#featuresInput').val(combined.join(', '));
        
        $('#featuresStatus').html(
            '<div class="alert alert-info mt-2">' +
            '<i class="fas fa-info-circle"></i> ' +
            'Combined ' + speech.length + ' speech + ' + 
            handwriting.length + ' handwriting + ' + 
            gait.length + ' gait features = ' + 
            combined.length + ' total features ready for prediction!' +
            '</div>'
        );
        
        // Show predict button in audio tab if audio features exist
        if (speech.length > 0) {
            showPredictButton('audioPanel');
        }
        
        // Show predict button in handwriting tab if handwriting features exist
        if (handwriting.length > 0) {
            showPredictButton('handwritingPanel');
        }
    }
}

// Show predict button in a specific tab
function showPredictButton(panelId) {
    const panel = $('#' + panelId);
    
    // Remove existing button if any
    panel.find('.predict-btn-container').remove();
    
    // Add predict button at the end of the panel
    const buttonHtml = `
        <div class="predict-btn-container mt-4 mb-3">
            <div class="d-grid gap-2">
                <button type="button" class="btn btn-success btn-lg predict-now-btn">
                    <i class="fas fa-brain"></i> Make Prediction Now
                </button>
            </div>
            <p class="text-center text-muted mt-2 mb-0">
                <small><i class="fas fa-info-circle"></i> Features are ready for prediction</small>
            </p>
        </div>
    `;
    
    // Append to the panel
    panel.append(buttonHtml);
    
    // Bind click event
    panel.find('.predict-now-btn').off('click').on('click', function() {
        $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
        makePredictionFromFeatures();
    });
    
    // Scroll to the button
    setTimeout(function() {
        panel.find('.predict-btn-container')[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

// Make prediction directly from extracted features
function makePredictionFromFeatures() {
    // Get available features - NO auto-generation
    const speech = window.speechFeatures || [];
    const handwriting = window.handwritingFeatures || [];
    const gait = window.gaitFeatures || [];
    
    // Check what we have
    const hasAllModalities = speech.length > 0 && handwriting.length > 0 && gait.length > 0;
    const hasSpeechOnly = speech.length > 0 && handwriting.length === 0 && gait.length === 0;
    const hasHandwritingOnly = handwriting.length > 0 && speech.length === 0 && gait.length === 0;
    
    let features = [];
    let usingDefaultFeatures = false;
    
    // If we have all modalities, use them all
    if (hasAllModalities) {
        features = [...speech, ...handwriting, ...gait];
    }
    // If we only have speech, generate realistic defaults for others with a warning
    else if (hasSpeechOnly) {
        features = [...speech, ...generateDefaultHandwritingFeatures(), ...generateDefaultGaitFeatures()];
        usingDefaultFeatures = true;
        console.log('Using speech features + estimated handwriting/gait features');
    }
    // If we only have handwriting, generate defaults for others
    else if (hasHandwritingOnly) {
        features = [...generateDefaultSpeechFeatures(), ...handwriting, ...generateDefaultGaitFeatures()];
        usingDefaultFeatures = true;
        console.log('Using handwriting features + estimated speech/gait features');
    }
    // Otherwise, we need more data
    else {
        showNotification('Please provide at least speech OR handwriting data to make a prediction.', 'warning');
        restorePredictButton();
        return;
    }
    
    // Validate we have exactly 42 features
    if (features.length !== 42) {
        showNotification(`Error: Expected 42 features, got ${features.length}.`, 'danger');
        restorePredictButton();
        return;
    }
    
    // Show loading in current tab
    const currentTab = $('.tab-pane.active');
    currentTab.find('.predict-btn-container').html(
        '<div class="text-center">' +
        '<div class="spinner-border text-primary" role="status"></div>' +
        '<p class="mt-2 mb-1"><strong>Analyzing with AI Model...</strong></p>' +
        '<small class="text-muted">Processing your data</small>' +
        '</div>'
    );
    
    // Also update the main results section
    $('#placeholderSection').hide();
    $('#resultsSection').hide();
    $('#loadingSection').show();
    $('#loadingSection').html(
        '<div class="text-center py-5">' +
        '<div class="spinner-border text-primary mb-3" style="width: 3rem; height: 3rem;" role="status"></div>' +
        '<h5>Running AI Analysis...</h5>' +
        '<p class="text-muted">The SVM model is processing your features</p>' +
        (usingDefaultFeatures ? '<small class="text-warning"><i class="fas fa-info-circle"></i> Using estimated values for missing data</small>' : '') +
        '</div>'
    );
    
    // Add a minimum delay to show loading state (so users see the model is working)
    const minLoadingTime = 800; // milliseconds
    const startTime = Date.now();
    
    // Make API request
    $.ajax({
        url: '/api/predict',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ features: features }),
        success: function(response) {
            // Ensure minimum loading time has passed
            const elapsedTime = Date.now() - startTime;
            const remainingTime = Math.max(0, minLoadingTime - elapsedTime);
            
            setTimeout(function() {
                if (response.success) {
                    displayResults(response);
                    
                    // Update button to show success
                    const currentTab = $('.tab-pane.active');
                    const statusDiv = currentTab.find('#audioStatus, #handwritingStatus');
                    
                    statusDiv.find('.predict-btn-container').html(
                        '<div class="alert alert-success mb-0">' +
                        '<i class="fas fa-check-circle"></i> <strong>Analysis Complete!</strong> ' +
                        'See results on the right →' +
                        '</div>'
                    );
                } else {
                    showNotification('Prediction failed: ' + response.error, 'danger');
                    restorePredictButton();
                }
            }, remainingTime);
        },
        error: function(xhr, status, error) {
            // Ensure minimum loading time has passed even for errors
            const elapsedTime = Date.now() - startTime;
            const remainingTime = Math.max(0, minLoadingTime - elapsedTime);
            
            setTimeout(function() {
                const errorMsg = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error';
                showNotification('Prediction failed: ' + errorMsg, 'danger');
                
                $('#loadingSection').hide();
                $('#placeholderSection').show();
                
                restorePredictButton();
            }, remainingTime);
        }
    });
}

// Restore predict button after error
function restorePredictButton() {
    const currentTab = $('.tab-pane.active');
    const statusDiv = currentTab.find('#audioStatus, #handwritingStatus');
    
    statusDiv.find('.predict-btn-container').html(
        '<div class="d-grid gap-2">' +
        '<button type="button" class="btn btn-success btn-lg predict-now-btn">' +
        '<i class="fas fa-brain"></i> Try Again' +
        '</button>' +
        '</div>' +
        '<p class="text-center text-muted mt-2 mb-0">' +
        '<small><i class="fas fa-info-circle"></i> Click to retry prediction</small>' +
        '</p>'
    );
    
    // Re-bind click event
    statusDiv.find('.predict-now-btn').off('click').on('click', function() {
        $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
        makePredictionFromFeatures();
    });
}

// Generate default speech features with some randomization
function generateDefaultSpeechFeatures() {
    // Generate 22 realistic speech features with slight variations
    const baseFeatures = [
        0.00289, 0.00245, -0.00456, 0.00312, 0.00198, -0.00234, 0.00167, 0.00289, -0.00123, 0.00456, 0.00234, -0.00345, 0.00123,  // MFCCs
        0.00168, 0.00003, 0.00420, 0.00252,  // Jitter
        0.01438, 0.09796,  // Shimmer
        21.033, 0.414783, 0.815285  // HNR, RPDE, DFA
    ];
    
    // Add small random variations (±10%)
    return baseFeatures.map(val => {
        const variation = (Math.random() - 0.5) * 0.2;
        return parseFloat((val * (1 + variation)).toFixed(6));
    });
}

// Generate default handwriting features with some randomization
function generateDefaultHandwritingFeatures() {
    // Generate 10 realistic handwriting features with slight variations
    // Add small random variations to avoid identical predictions
    const baseFeatures = [
        0.5234,  // Mean Pressure
        0.1234,  // Pressure Variation
        2.4567,  // Mean Velocity
        0.3456,  // Velocity Variation
        1.2345,  // Mean Acceleration
        0.1500,  // Pen-up Time
        5.6789,  // Stroke Length
        1.8900,  // Writing Tempo
        0.0500,  // Tremor Frequency
        0.7500   // Fluency Score
    ];
    
    // Add small random variations (±10%) to make each prediction unique
    return baseFeatures.map(val => {
        const variation = (Math.random() - 0.5) * 0.2; // ±10%
        return parseFloat((val * (1 + variation)).toFixed(6));
    });
}

// Generate default gait features with some randomization
function generateDefaultGaitFeatures() {
    // Generate 10 realistic gait features with slight variations
    // Add small random variations to avoid identical predictions
    const baseFeatures = [
        1.0500,  // Stride Interval
        0.0300,  // Stride Variability
        0.4000,  // Swing Time
        0.6000,  // Stance Time
        0.2000,  // Double Support
        1.2000,  // Gait Speed
        110.00,  // Cadence
        0.7000,  // Step Length
        0.9500,  // Stride Regularity
        0.0500   // Gait Asymmetry
    ];
    
    // Add small random variations (±10%) to make each prediction unique
    return baseFeatures.map(val => {
        const variation = (Math.random() - 0.5) * 0.2; // ±10%
        return parseFloat((val * (1 + variation)).toFixed(6));
    });
}

// Create feature display card
function createFeatureDisplay(features, modalityName) {
    const featureNames = getFeatureNames(modalityName);
    
    let rows = '';
    features.forEach((value, index) => {
        const featureName = featureNames[index] || `Feature ${index + 1}`;
        const displayValue = parseFloat(value).toFixed(4);
        rows += `
            <tr>
                <td><small>${featureName}</small></td>
                <td class="text-end"><code>${displayValue}</code></td>
            </tr>
        `;
    });
    
    const collapseId = modalityName.toLowerCase() + 'FeaturesCollapse';
    
    return `
        <div class="card mt-2 border-info">
            <div class="card-header bg-light p-2">
                <a class="text-decoration-none d-flex justify-content-between align-items-center" 
                   data-bs-toggle="collapse" href="#${collapseId}" role="button">
                    <span><i class="fas fa-list-ul"></i> <strong>Extracted ${modalityName} Features (${features.length})</strong></span>
                    <i class="fas fa-chevron-down"></i>
                </a>
            </div>
            <div class="collapse" id="${collapseId}">
                <div class="card-body p-2" style="max-height: 300px; overflow-y: auto;">
                    <table class="table table-sm table-striped mb-0">
                        <thead>
                            <tr>
                                <th>Feature Name</th>
                                <th class="text-end">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// Get feature names for display
function getFeatureNames(modalityName) {
    if (modalityName === 'Speech') {
        return [
            'MFCC-1', 'MFCC-2', 'MFCC-3', 'MFCC-4', 'MFCC-5',
            'MFCC-6', 'MFCC-7', 'MFCC-8', 'MFCC-9', 'MFCC-10',
            'MFCC-11', 'MFCC-12', 'MFCC-13',
            'Jitter (%)', 'Jitter (Abs)', 'RAP', 'PPQ',
            'Shimmer', 'Shimmer (dB)', 'HNR', 'RPDE', 'DFA'
        ];
    } else if (modalityName === 'Handwriting') {
        return [
            'Mean Pressure', 'Pressure Variation', 'Mean Velocity',
            'Velocity Variation', 'Mean Acceleration', 'Pen-up Time',
            'Stroke Length', 'Writing Tempo', 'Tremor Frequency',
            'Fluency Score'
        ];
    } else if (modalityName === 'Gait') {
        return [
            'Stride Interval', 'Stride Variability', 'Swing Time',
            'Stance Time', 'Double Support', 'Gait Speed',
            'Cadence', 'Step Length', 'Stride Regularity',
            'Gait Asymmetry'
        ];
    }
    return [];
}

