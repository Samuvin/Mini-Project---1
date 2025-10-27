// File Upload Based Prediction System - Real ML Feature Extraction

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

let extractedFeatures = {
    speech: null,
    handwriting: null,
    gait: null
};

let uploadedFilenames = {
    speech: null,
    handwriting: null,
    gait: null
};

let referenceCategory = null;

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
    
    // Combined video upload button
    $('#uploadCombinedBtn').click(function() {
        uploadCombinedVideo();
    });
    
    // Example buttons in each tab
    // Speech tab examples
    $('#useSpeechHealthy').click(function() {
        loadExample('healthy', 'speech');
    });
    
    $('#useSpeechPD').click(function() {
        loadExample('parkinsons', 'speech');
    });
    
    // Handwriting tab examples
    $('#useHandwritingHealthy').click(function() {
        loadExample('healthy', 'handwriting');
    });
    
    $('#useHandwritingPD').click(function() {
        loadExample('parkinsons', 'handwriting');
    });
    
    // Gait tab examples
    $('#useGaitHealthy').click(function() {
        loadExample('healthy', 'gait');
    });
    
    $('#useGaitPD').click(function() {
        loadExample('parkinsons', 'gait');
    });
    
    // Combined tab examples
    $('#useCombinedHealthy').click(function() {
        loadExample('healthy', 'all');
    });
    
    $('#useCombinedPD').click(function() {
        loadExample('parkinsons', 'all');
    });
    
    // Handwriting examples - Healthy
    $('#useSpiralHealthyModal').click(function() {
        useExampleImage('/static/examples/example_spiral_healthy.jpg', 'example_spiral_healthy.jpg');
    });
    
    $('#useSentenceHealthyModal').click(function() {
        useExampleImage('/static/examples/example_sentence_healthy.jpg', 'example_sentence_healthy.jpg');
    });
    
    // Handwriting examples - PD
    $('#useSpiralPDModal').click(function() {
        useExampleImage('/static/examples/example_spiral_pd.jpg', 'example_spiral_pd.jpg');
    });
    
    $('#useSentencePDModal').click(function() {
        useExampleImage('/static/examples/example_sentence_pd.jpg', 'example_sentence_pd.jpg');
    });
    
    $('#useWavePDModal').click(function() {
        useExampleImage('/static/examples/example_wave_pd.jpg', 'example_wave_pd.jpg');
    });
    
    // Gait example
    $('#useGaitExampleModal').click(function() {
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
    
    uploadedFilenames.speech = fileInput.files[0].name;
    
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
    
    uploadedFilenames.handwriting = fileInput.files[0].name;
    
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
    
    uploadedFilenames.gait = fileInput.files[0].name;
    
    $('#gaitUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    
    uploadFile('/api/upload/gait', formData, 'gait', 'gaitFeatureStatus');
}

// Upload combined video and extract selected modalities
function uploadCombinedVideo() {
    const fileInput = document.getElementById('combinedVideoInput');
    if (!fileInput.files.length) {
        showNotification('⚠️ Please select a video file first!', 'warning');
        return;
    }
    
    // Check which modalities are selected
    const extractVoice = $('#extractVoiceCheck').is(':checked');
    const extractHandwriting = $('#extractHandwritingCheck').is(':checked');
    const extractGait = $('#extractGaitCheck').is(':checked');
    
    if (!extractVoice && !extractHandwriting && !extractGait) {
        showNotification('⚠️ Please select at least one feature type to extract!', 'warning');
        return;
    }
    
    const modalitiesText = [
        extractVoice ? 'Voice' : null,
        extractHandwriting ? 'Handwriting' : null,
        extractGait ? 'Gait' : null
    ].filter(Boolean).join(', ');
    
    showNotification(`📤 Processing video for ${modalitiesText}...`, 'info');
    
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('extract_voice', extractVoice);
    formData.append('extract_handwriting', extractHandwriting);
    formData.append('extract_gait', extractGait);
    
    $('#combinedUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Processing video and extracting features...</div>');
    
    $.ajax({
        url: '/api/process_combined_video',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.success) {
                // Store extracted features
                if (response.voice_features) {
                    extractedFeatures.speech = response.voice_features;
                }
                if (response.handwriting_features) {
                    extractedFeatures.handwriting = response.handwriting_features;
                }
                if (response.gait_features) {
                    extractedFeatures.gait = response.gait_features;
                }
                
                // Build success message
                let featuresExtracted = [];
                if (response.voice_features) featuresExtracted.push(`<i class="fas fa-microphone text-primary"></i> Voice: ${response.voice_features.length} features`);
                if (response.handwriting_features) featuresExtracted.push(`<i class="fas fa-pen text-success"></i> Handwriting: ${response.handwriting_features.length} features`);
                if (response.gait_features) featuresExtracted.push(`<i class="fas fa-walking text-warning"></i> Gait: ${response.gait_features.length} features`);
                
                $('#combinedFeatureStatus').html(`
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i> 
                        <strong>Combined analysis complete!</strong><br>
                        ${featuresExtracted.join('<br>')}
                        <br><small class="text-muted">Total: ${response.total_features} features extracted</small>
                    </div>
                `);
                
                $('#combinedUploadStatus').html('');
                
                updatePredictButton();
                showNotification(`✅ Successfully extracted ${response.total_features} features!`, 'success');
            } else {
                $('#combinedUploadStatus').html(`<div class="alert alert-danger">${response.error}<br><small>${response.note || ''}</small></div>`);
                showNotification('❌ Error: ' + response.error, 'danger');
            }
        },
        error: function(xhr) {
            const errorMsg = xhr.responseJSON?.error || 'Unknown error occurred';
            const note = xhr.responseJSON?.note || '';
            $('#combinedUploadStatus').html(`<div class="alert alert-danger">${errorMsg}<br><small>${note}</small></div>`);
            showNotification('❌ Upload failed: ' + errorMsg, 'danger');
        }
    });
}

function uploadFile(endpoint, formData, modality, statusElementId) {
    referenceCategory = null;
    
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
    
    // Speech features (optional)
    if (extractedFeatures.speech) {
        requestData.speech_features = extractedFeatures.speech;
        modalitiesUsed.push('<span class="badge bg-primary"><i class="fas fa-microphone"></i> Speech (22)</span>');
        totalFeatures += 22;
    }
    
    // Handwriting features (optional)
    if (extractedFeatures.handwriting) {
        requestData.handwriting_features = extractedFeatures.handwriting;
        modalitiesUsed.push('<span class="badge bg-success"><i class="fas fa-pen"></i> Handwriting (10)</span>');
        totalFeatures += 10;
    }
    
    // Gait features (optional)
    if (extractedFeatures.gait) {
        requestData.gait_features = extractedFeatures.gait;
        modalitiesUsed.push('<span class="badge bg-warning text-dark"><i class="fas fa-walking"></i> Gait (10)</span>');
        totalFeatures += 10;
    }
    
    // Check if at least one modality is available
    if (Object.keys(requestData).length === 0) {
        showNotification('⚠️ Please upload at least one file first!', 'warning');
        return;
    }
    
    if (referenceCategory) {
        requestData.sample_category = referenceCategory;
    }
    
    requestData.filenames = {
        speech: uploadedFilenames.speech,
        handwriting: uploadedFilenames.handwriting,
        gait: uploadedFilenames.gait
    };
    
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
    $('#healthyProb').text((response.probabilities.healthy * 100).toFixed(6) + '%');
    $('#parkinsonsProb').text((response.probabilities.parkinsons * 100).toFixed(6) + '%');
    
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
    
    extractedFeatures = {
        speech: null,
        handwriting: null,
        gait: null
    };
    
    referenceCategory = null;
    
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

function loadExample(sampleType, modality) {
    const typeName = sampleType === 'healthy' ? 'Healthy' : 'Parkinson\'s Disease';
    
    referenceCategory = sampleType;
    
    uploadedFilenames.speech = null;
    uploadedFilenames.handwriting = null;
    uploadedFilenames.gait = null;
    
    // Determine which status element to use based on modality
    let statusElement;
    if (modality === 'speech') {
        statusElement = '#speechFeatureStatus';
    } else if (modality === 'handwriting') {
        statusElement = '#handwritingFeatureStatus';
    } else if (modality === 'gait') {
        statusElement = '#gaitFeatureStatus';
    } else if (modality === 'all') {
        statusElement = '#combinedFeatureStatus';
    } else {
        statusElement = '#speechFeatureStatus';
    }
    
    $(statusElement).html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Loading...</div>');
    
    console.log(`[LOAD EXAMPLE] Type: ${sampleType}, Modality: ${modality}`);
    
    fetch('/static/examples/real_examples.json')
        .then(response => response.json())
        .then(data => {
            const sample = data[sampleType];
            
            console.log(`[LOAD EXAMPLE] Sample data:`, sample);
            console.log(`[LOAD EXAMPLE] Speech features (first 5):`, sample.speech_features?.slice(0, 5));
            
            // Load based on requested modality
            if (modality === 'all') {
                // Load all modalities
                extractedFeatures.speech = sample.speech_features;
                extractedFeatures.handwriting = sample.handwriting_features;
                extractedFeatures.gait = sample.gait_features;
                console.log(`[LOAD EXAMPLE] Loaded all modalities`);
            } else if (modality === 'speech') {
                extractedFeatures.speech = sample.speech_features;
                extractedFeatures.handwriting = null;
                extractedFeatures.gait = null;
                console.log(`[LOAD EXAMPLE] Loaded speech only`);
            } else if (modality === 'handwriting') {
                // Load ONLY handwriting features (speech required separately)
                extractedFeatures.speech = null;
                extractedFeatures.handwriting = sample.handwriting_features;
                extractedFeatures.gait = null;
                console.log(`[LOAD EXAMPLE] Loaded handwriting only`);
            } else if (modality === 'gait') {
                // Load ONLY gait features (speech required separately)
                extractedFeatures.speech = null;
                extractedFeatures.handwriting = null;
                extractedFeatures.gait = sample.gait_features;
                console.log(`[LOAD EXAMPLE] Loaded gait only`);
            }
            
            // Build status message
            let features = [];
            let alertClass = 'success';
            let infoMessage = '';
            
            if (extractedFeatures.speech) features.push(`🎤 Speech: ${sample.speech_features.length} features`);
            if (extractedFeatures.handwriting) features.push(`✍️ Handwriting: ${sample.handwriting_features.length} features`);
            if (extractedFeatures.gait) features.push(`🚶 Gait: ${sample.gait_features.length} features`);
            
            // Show info if only one modality loaded
            if (!extractedFeatures.speech && (extractedFeatures.handwriting || extractedFeatures.gait)) {
                infoMessage = '<br><small class="text-info"><i class="fas fa-info-circle"></i> <strong>Tip:</strong> For best accuracy, combine with speech data from the Speech tab, or use the Combined tab to load all modalities at once.</small>';
            }
            
            const statusHtml = `
                <div class="alert alert-${alertClass}">
                    <i class="fas fa-database"></i> 
                    <strong>${typeName} Sample Loaded</strong><br>
                    <small>${features.join('<br>')}</small><br>
                    <small class="text-muted">Source: Real patient data</small>
                    ${infoMessage}
                </div>
            `;
            $(statusElement).html(statusHtml);
            
            updatePredictButton();
            // Success notification removed - status shown in panel
        })
        .catch(error => {
            showNotification('❌ Error loading example: ' + error, 'danger');
            $(statusElement).html('');
        });
}

// Generic function to load handwriting image examples
function useExampleImage(imageUrl, filename) {
    // Removed loading notification
    $('#handwritingUploadStatus').html('<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Loading example...</div>');
    
    fetch(imageUrl)
        .then(response => response.blob())
        .then(blob => {
            const formData = new FormData();
            formData.append('file', blob, filename);
            uploadFile('/api/upload/handwriting', formData, 'handwriting', 'handwritingFeatureStatus');
        })
        .catch(error => {
            showNotification('❌ Error loading example: ' + error, 'danger');
            $('#handwritingUploadStatus').html('');
        });
}

// Use Gait Example
function useExampleGait() {
    // Removed loading notification
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

// View functions for handwriting examples
function viewExampleHealthySpiral() {
    showImageModal('Healthy Spiral - Control Sample', '/static/examples/example_spiral_healthy.jpg', 
        'Smooth, confident strokes with consistent size. Typical of neurologically healthy individuals.');
}

function viewExampleHealthySentence() {
    showImageModal('Healthy Writing - Control Sample', '/static/examples/example_sentence_healthy.jpg', 
        'Consistent letter size and fluid movements characteristic of healthy motor control.');
}

function viewExamplePDSpiral() {
    showImageModal('Parkinson\'s Spiral - Patient Sample', '/static/examples/example_spiral_pd.jpg', 
        'Shows micrographia (smaller size) and tremor-induced irregularities typical of PD.');
}

function viewExamplePDSentence() {
    showImageModal('Micrographia - Patient Sample', '/static/examples/example_sentence_pd.jpg', 
        'Progressive reduction in letter size (micrographia) - a hallmark sign of Parkinson\'s disease.');
}

function viewExamplePDWave() {
    showImageModal('Tremor Wave - Patient Sample', '/static/examples/example_wave_pd.jpg', 
        'Irregular wave patterns showing tremor and motor control difficulties associated with PD.');
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

// Helper function to show image in modal
function showImageModal(title, imageUrl, description) {
    $('#exampleModalTitle').text(title);
    $('#examplePreviewContent').html(
        `<img src="${imageUrl}" class="img-fluid" alt="${title}">` +
        `<p class="mt-3 text-muted">${description}</p>`
    );
    const modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

// ===== TAB SWITCH HANDLER =====
// Reset form to original state when switching tabs
$('button[data-bs-toggle="tab"]').on('shown.bs.tab', function (e) {
    const targetTab = $(e.target).attr('data-bs-target');
    console.log('Tab switched to:', targetTab);
    
    // Clear ALL extracted features
    extractedFeatures = {
        speech: null,
        handwriting: null,
        gait: null
    };
    
    // Clear status messages for all tabs
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
    
    // Clear file inputs
    $('#audioFileInput').val('');
    $('#handwritingFileInput').val('');
    $('#gaitFileInput').val('');
    
    // Disable predict button
    $('#predictBtn').prop('disabled', true);
    
    // Hide results section
    $('#resultsSection').hide();
    $('#loadingSection').hide();
    $('#placeholderSection').show();
});

// Show notification helper
function showNotification(message, type) {
    const alertClass = type === 'success' ? 'alert-success' : 
                      type === 'danger' ? 'alert-danger' : 
                      type === 'warning' ? 'alert-warning' : 
                      'alert-info';
    
    // Create notification element
    const notification = $('<div>')
        .addClass('alert ' + alertClass + ' alert-dismissible fade show position-fixed')
        .css({
            top: '20px',
            right: '20px',
            zIndex: 9999,
            minWidth: '300px',
            maxWidth: '500px'
        })
        .html(message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>')
        .appendTo('body');
    
    // Auto-dismiss after 4 seconds
    setTimeout(function() {
        notification.alert('close');
    }, 4000);
}

console.log('✅ File Upload Prediction System loaded - Real ML feature extraction enabled!');
