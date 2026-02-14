/**
 * predict.js — Prediction page logic.
 *
 * Fixes from plan:
 *  1. Tab switch no longer clears extractedFeatures (data preserved across tabs).
 *  2. Orphaned handler bindings removed.
 *  3. resetForm() also clears combined status/inputs/checkboxes.
 *  4. showNotification uses Bootstrap 5 Toast API (via main.js global).
 *  5. Upload buttons show spinner & disable during upload.
 *  6. Predict button tooltip when disabled.
 *  7. Unified showNotification (falls through to main.js).
 */

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

/* ------------------------------------------------------------------ */
/*  Bootstrap Ready                                                    */
/* ------------------------------------------------------------------ */

$(document).ready(function () {

    // ---- Event Bindings ----

    // Voice recording
    $('#recordBtn').click(function () {
        if (!isRecording) startRecording(); else stopRecording();
    });

    // Upload buttons
    $('#uploadAudioBtn').click(function () { uploadAudioFile(); });
    $('#uploadHandwritingBtn').click(function () { uploadHandwritingFile(); });
    $('#uploadGaitBtn').click(function () { uploadGaitFile(); });
    $('#uploadCombinedBtn').click(function () { uploadCombinedVideo(); });

    // Example buttons – Speech
    $('#useSpeechHealthy').click(function () { loadExample('healthy', 'speech'); });
    $('#useSpeechPD').click(function () { loadExample('parkinsons', 'speech'); });

    // Example buttons – Handwriting
    $('#useHandwritingHealthy').click(function () { loadExample('healthy', 'handwriting'); });
    $('#useHandwritingPD').click(function () { loadExample('parkinsons', 'handwriting'); });

    // Example buttons – Gait
    $('#useGaitHealthy').click(function () { loadExample('healthy', 'gait'); });
    $('#useGaitPD').click(function () { loadExample('parkinsons', 'gait'); });

    // Example buttons – Combined
    $('#useCombinedHealthy').click(function () { loadExample('healthy', 'all'); });
    $('#useCombinedPD').click(function () { loadExample('parkinsons', 'all'); });

    // Image preview
    $('#handwritingFileInput').change(function () { previewHandwritingImage(this); });

    // Track file selections
    $('#audioFileInput').change(function () {
        if (this.files && this.files[0]) uploadedFilenames.speech = this.files[0].name;
    });
    $('#handwritingFileInput').change(function () {
        if (this.files && this.files[0]) uploadedFilenames.handwriting = this.files[0].name;
    });
    $('#gaitFileInput').change(function () {
        if (this.files && this.files[0]) uploadedFilenames.gait = this.files[0].name;
    });
    $('#combinedVideoInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.speech = this.files[0].name;
            uploadedFilenames.handwriting = this.files[0].name;
            uploadedFilenames.gait = this.files[0].name;
        }
    });

    // Predict & Reset
    $('#predictBtn').click(function () { makePrediction(); });
    $('#resetBtn').click(function () { resetForm(); });

    // Initialize predict button tooltip (Bootstrap 5)
    var predictBtn = document.getElementById('predictBtn');
    if (predictBtn) new bootstrap.Tooltip(predictBtn);

    // DL accordion toggles (event delegation)
    $(document).on('click', '.dl-section-toggle', function () {
        var targetId = $(this).data('target');
        var $content = $('#' + targetId);
        if ($content.is(':visible')) {
            $content.slideUp(200);
            $(this).addClass('collapsed');
        } else {
            $content.slideDown(200);
            $(this).removeClass('collapsed');
        }
    });

    // Drag-and-drop zones
    initDropZone('audioDropZone', 'audioFileInput');
    initDropZone('handwritingDropZone', 'handwritingFileInput');
    initDropZone('gaitDropZone', 'gaitFileInput');
    initDropZone('combinedDropZone', 'combinedVideoInput');
});

/* ------------------------------------------------------------------ */
/*  Drag & Drop Helper                                                 */
/* ------------------------------------------------------------------ */

function initDropZone(zoneId, inputId) {
    var zone = document.getElementById(zoneId);
    var input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', function () { input.click(); });

    zone.addEventListener('dragover', function (e) {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', function () {
        zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            $(input).trigger('change');
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Step Indicator                                                     */
/* ------------------------------------------------------------------ */

function updateSteps(current) {
    // current: 1 = select, 2 = provide, 3 = results
    for (var i = 1; i <= 3; i++) {
        var step = document.getElementById('step' + i);
        if (!step) continue;
        step.classList.remove('active', 'completed');
        if (i < current) step.classList.add('completed');
        else if (i === current) step.classList.add('active');
    }
    for (var j = 1; j <= 2; j++) {
        var line = document.getElementById('stepLine' + j);
        if (!line) continue;
        line.classList.remove('active', 'completed');
        if (j < current) line.classList.add('completed');
        else if (j === current) line.classList.add('active');
    }
}

/* ------------------------------------------------------------------ */
/*  Tab Switch -- FIX #1: Do NOT clear extractedFeatures               */
/* ------------------------------------------------------------------ */

$('button[data-bs-toggle="tab"]').on('shown.bs.tab', function () {
    // Only clear UI status for the CURRENT tab's displays.
    // Do NOT reset extractedFeatures — data is preserved across tabs.
    updateSteps(1);
});

/* ------------------------------------------------------------------ */
/*  Voice Recording                                                    */
/* ------------------------------------------------------------------ */

async function startRecording() {
    try {
        var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = function (event) { audioChunks.push(event.data); };
        mediaRecorder.onstop = function () {
            var audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            uploadRecordedAudio(audioBlob);
            stream.getTracks().forEach(function (t) { t.stop(); });
        };

        mediaRecorder.start();
        isRecording = true;
        $('#recordBtn').removeClass('btn-danger').addClass('btn-warning');
        $('#recordBtnText').text('Stop Recording');
        $('#recordingStatus').show();
    } catch (error) {
        console.error('Microphone error:', error);
        showNotification('Could not access microphone. Check permissions.', 'danger');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        $('#recordBtn').removeClass('btn-warning').addClass('btn-danger');
        $('#recordBtnText').text('Start Recording');
        $('#recordingStatus').hide();
        showNotification('Recording stopped. Processing audio...', 'success');
    }
}

/* ------------------------------------------------------------------ */
/*  Upload Functions                                                   */
/* ------------------------------------------------------------------ */

function uploadRecordedAudio(audioBlob) {
    var formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');
    uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus', '#uploadAudioBtn');
}

function uploadAudioFile() {
    var fileInput = document.getElementById('audioFileInput');
    if (!fileInput.files.length) { showNotification('Please select an audio file first', 'warning'); return; }
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadedFilenames.speech = fileInput.files[0].name;
    $('#audioUploadStatus').html('<div class="alert alert-info small"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus', '#uploadAudioBtn');
}

function uploadHandwritingFile() {
    var fileInput = document.getElementById('handwritingFileInput');
    if (!fileInput.files.length) { showNotification('Please select an image first', 'warning'); return; }
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadedFilenames.handwriting = fileInput.files[0].name;
    $('#handwritingUploadStatus').html('<div class="alert alert-info small"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    uploadFile('/api/upload/handwriting', formData, 'handwriting', 'handwritingFeatureStatus', '#uploadHandwritingBtn');
}

function uploadGaitFile() {
    var fileInput = document.getElementById('gaitFileInput');
    if (!fileInput.files.length) { showNotification('Please select a video first', 'warning'); return; }
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadedFilenames.gait = fileInput.files[0].name;
    $('#gaitUploadStatus').html('<div class="alert alert-info small"><i class="fas fa-spinner fa-spin"></i> Uploading and extracting features...</div>');
    uploadFile('/api/upload/gait', formData, 'gait', 'gaitFeatureStatus', '#uploadGaitBtn');
}

/* FIX #5 — spinner + disable on upload button */
function uploadFile(endpoint, formData, modality, statusElementId, btnSelector) {
    referenceCategory = null;
    var $btn = btnSelector ? $(btnSelector) : null;
    var btnOrigHtml = $btn ? $btn.html() : '';

    if ($btn) {
        $btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
    }

    updateSteps(2);

    $.ajax({
        url: endpoint,
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            if ($btn) $btn.prop('disabled', false).html(btnOrigHtml);

            if (response.success) {
                extractedFeatures[modality] = response.features;
                $('#' + modality + 'Features').val(response.features.join(','));

                var modalityIcon = { speech: '🎤', handwriting: '✍️', gait: '🚶' };
                $('#' + statusElementId).html(
                    '<div class="alert alert-success small">' +
                    '<i class="fas fa-check-circle"></i> <strong>Success!</strong><br>' +
                    modalityIcon[modality] + ' Extracted ' + response.feature_count + ' features<br>' +
                    '<small>' + response.message + '</small>' +
                    (response.note ? '<br><small class="text-muted">' + response.note + '</small>' : '') +
                    '</div>'
                );

                clearUploadStatus(modality);
                updatePredictButton();
            } else {
                showNotification(response.error, 'danger');
                clearUploadStatus(modality);
            }
        },
        error: function (xhr) {
            if ($btn) $btn.prop('disabled', false).html(btnOrigHtml);
            var errorMsg = xhr.responseJSON && xhr.responseJSON.error ? xhr.responseJSON.error : 'Upload failed';
            showNotification(errorMsg, 'danger');
            clearUploadStatus(modality);
        }
    });
}

function clearUploadStatus(modality) {
    var id = modality === 'speech' ? 'audioUploadStatus' :
             modality === 'handwriting' ? 'handwritingUploadStatus' :
             'gaitUploadStatus';
    $('#' + id).html('');
}

/* ------------------------------------------------------------------ */
/*  Combined Video Upload                                              */
/* ------------------------------------------------------------------ */

function uploadCombinedVideo() {
    var fileInput = document.getElementById('combinedVideoInput');
    if (!fileInput.files.length) { showNotification('Please select a video file first!', 'warning'); return; }

    var extractVoice = $('#extractVoiceCheck').is(':checked');
    var extractHandwriting = $('#extractHandwritingCheck').is(':checked');
    var extractGait = $('#extractGaitCheck').is(':checked');

    if (!extractVoice && !extractHandwriting && !extractGait) {
        showNotification('Select at least one feature type to extract!', 'warning'); return;
    }

    var modalitiesText = [extractVoice ? 'Voice' : null, extractHandwriting ? 'Handwriting' : null, extractGait ? 'Gait' : null].filter(Boolean).join(', ');
    showNotification('Processing video for ' + modalitiesText + '...', 'info');

    var formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('extract_voice', extractVoice);
    formData.append('extract_handwriting', extractHandwriting);
    formData.append('extract_gait', extractGait);

    var videoFilename = fileInput.files[0].name;
    if (extractVoice) uploadedFilenames.speech = videoFilename;
    if (extractHandwriting) uploadedFilenames.handwriting = videoFilename;
    if (extractGait) uploadedFilenames.gait = videoFilename;

    var $btn = $('#uploadCombinedBtn');
    var btnOrig = $btn.html();
    $btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');

    $('#combinedUploadStatus').html('<div class="alert alert-info small"><i class="fas fa-spinner fa-spin"></i> Processing video and extracting features...</div>');
    updateSteps(2);

    $.ajax({
        url: '/api/process_combined_video',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            $btn.prop('disabled', false).html(btnOrig);
            if (response.success) {
                if (response.voice_features) extractedFeatures.speech = response.voice_features;
                if (response.handwriting_features) extractedFeatures.handwriting = response.handwriting_features;
                if (response.gait_features) extractedFeatures.gait = response.gait_features;

                var fe = [];
                if (response.voice_features) fe.push('<i class="fas fa-microphone" style="color:var(--accent)"></i> Voice: ' + response.voice_features.length + ' features');
                if (response.handwriting_features) fe.push('<i class="fas fa-pen" style="color:var(--success)"></i> Handwriting: ' + response.handwriting_features.length + ' features');
                if (response.gait_features) fe.push('<i class="fas fa-walking" style="color:var(--warning)"></i> Gait: ' + response.gait_features.length + ' features');

                $('#combinedFeatureStatus').html(
                    '<div class="alert alert-success small">' +
                    '<i class="fas fa-check-circle"></i> <strong>Combined analysis complete!</strong><br>' +
                    fe.join('<br>') +
                    '<br><small class="text-muted">Total: ' + response.total_features + ' features extracted</small></div>'
                );
                $('#combinedUploadStatus').html('');
                updatePredictButton();
                showNotification('Successfully extracted ' + response.total_features + ' features!', 'success');
            } else {
                $('#combinedUploadStatus').html('<div class="alert alert-danger small">' + response.error + '<br><small>' + (response.note || '') + '</small></div>');
                showNotification('Error: ' + response.error, 'danger');
            }
        },
        error: function (xhr) {
            $btn.prop('disabled', false).html(btnOrig);
            var errorMsg = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error';
            var note = xhr.responseJSON ? (xhr.responseJSON.note || '') : '';
            $('#combinedUploadStatus').html('<div class="alert alert-danger small">' + errorMsg + '<br><small>' + note + '</small></div>');
            showNotification('Upload failed: ' + errorMsg, 'danger');
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Preview & Button State                                             */
/* ------------------------------------------------------------------ */

function previewHandwritingImage(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#handwritingImg').attr('src', e.target.result);
            $('#handwritingPreview').show();
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function updatePredictButton() {
    var hasAny = extractedFeatures.speech !== null ||
                 extractedFeatures.handwriting !== null ||
                 extractedFeatures.gait !== null;

    var btn = document.getElementById('predictBtn');
    if (btn) {
        btn.disabled = !hasAny;
        var tip = bootstrap.Tooltip.getInstance(btn);
        if (tip) {
            btn.setAttribute('data-bs-original-title', hasAny ? 'Run AI prediction' : 'Load example data or upload a file first');
            tip.hide();
        }
    }
    if (hasAny) updateSteps(2);
}

/* ------------------------------------------------------------------ */
/*  Prediction                                                         */
/* ------------------------------------------------------------------ */

function makePrediction() {
    var requestData = {};
    var modalitiesUsed = [];
    var totalFeatures = 0;

    if (extractedFeatures.speech) {
        requestData.speech_features = extractedFeatures.speech;
        modalitiesUsed.push('<span class="chip chip-accent"><i class="fas fa-microphone"></i> Speech (22)</span>');
        totalFeatures += 22;
    }
    if (extractedFeatures.handwriting) {
        requestData.handwriting_features = extractedFeatures.handwriting;
        modalitiesUsed.push('<span class="chip chip-success"><i class="fas fa-pen"></i> Handwriting (10)</span>');
        totalFeatures += 10;
    }
    if (extractedFeatures.gait) {
        requestData.gait_features = extractedFeatures.gait;
        modalitiesUsed.push('<span class="chip chip-warning"><i class="fas fa-walking"></i> Gait (10)</span>');
        totalFeatures += 10;
    }

    if (Object.keys(requestData).length === 0) {
        showNotification('Please upload at least one file first!', 'warning');
        return;
    }

    if (referenceCategory) requestData.sample_category = referenceCategory;
    requestData.filenames = { speech: uploadedFilenames.speech, handwriting: uploadedFilenames.handwriting, gait: uploadedFilenames.gait };

    $('#loadingSection').show();
    $('#loadingText').html('Analyzing <strong>' + totalFeatures + '</strong> features from <strong>' + modalitiesUsed.length + '</strong> modality/modalities<br>' + modalitiesUsed.join(' '));

    updateSteps(3);

    var startTime = Date.now();

    $.ajax({
        url: '/api/predict',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function (response) {
            var elapsed = Date.now() - startTime;
            var remaining = Math.max(0, 1000 - elapsed);
            setTimeout(function () {
                if (response.success) {
                    displayResults(response, modalitiesUsed, totalFeatures);
                } else {
                    showNotification('Prediction failed: ' + response.error, 'danger');
                    $('#loadingSection').hide();
                    updateSteps(2);
                }
            }, remaining);
        },
        error: function (xhr) {
            setTimeout(function () {
                var errorMsg = xhr.responseJSON && xhr.responseJSON.error ? xhr.responseJSON.error : 'Prediction failed';
                showNotification(errorMsg, 'danger');
                $('#loadingSection').hide();
                updateSteps(2);
            }, 500);
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Display Results                                                    */
/* ------------------------------------------------------------------ */

function displayResults(response, modalitiesUsed, totalFeatures) {
    $('#loadingSection').hide();

    var prediction = response.prediction;
    var confidence = response.confidence;
    var isDL = response.model_type === 'deep_learning';

    // Model type badge
    if (isDL) {
        $('#modelTypeBadgeContainer').html('<span class="model-type-badge dl"><i class="fas fa-brain"></i> SE-ResNet + Attention Fusion</span>');
    } else {
        $('#modelTypeBadgeContainer').html('<span class="model-type-badge sklearn"><i class="fas fa-cogs"></i> sklearn Ensemble</span>');
    }

    // Prediction icon and label
    var $icon = $('#predictionIcon');
    var $label = $('#predictionLabel');
    var $text = $('#predictionText');

    if (prediction === 1) {
        $icon.html('<i class="fas fa-exclamation-triangle" style="color:var(--warning)"></i>');
        $label.html("Parkinson's Disease Detected").css('color', 'var(--warning)');
        $text.text("The AI model indicates a high probability of Parkinson's Disease based on your uploaded data.");
    } else {
        $icon.html('<i class="fas fa-shield-alt" style="color:var(--success)"></i>');
        $label.html('Healthy').css('color', 'var(--success)');
        $text.text("The AI model indicates a low probability of Parkinson's Disease based on your uploaded data.");
    }

    // Modalities used
    $('#modalitiesUsed').html(
        modalitiesUsed.join(' ') +
        '<br><small style="color:var(--text-3)">Total: ' + totalFeatures + ' features extracted</small>'
    );

    // Confidence ring (SVG) — r=60 in modal
    var pct = (confidence * 100).toFixed(1);
    var circumference = 2 * Math.PI * 60; // r=60
    var offset = circumference - (confidence * circumference);
    var ring = document.getElementById('confidenceRing');
    if (ring) {
        var ringColor = confidence >= 0.7 ? 'var(--success)' : confidence >= 0.5 ? 'var(--warning)' : 'var(--danger)';
        ring.style.stroke = ringColor;
        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = offset;
    }
    $('#confidenceText').text(pct + '%');

    // Probability bars
    var hp = (response.probabilities.healthy * 100).toFixed(2);
    var pp = (response.probabilities.parkinsons * 100).toFixed(2);
    $('#healthyBar').css('width', hp + '%').css('background', 'var(--success)');
    $('#pdBar').css('width', pp + '%').css('background', 'var(--warning)');
    $('#healthyProb').text(hp + '%');
    $('#parkinsonsProb').text(pp + '%');

    // DL Explainability
    if (isDL) {
        renderAttentionWeights(response.attention_weights);
        renderFeatureImportance(response.feature_importance, response.feature_names);
        renderSEWeights(response.se_weights);
    } else {
        $('#dlAttentionSection').hide();
        $('#dlFeatureImportanceSection').hide();
        $('#dlSEWeightsSection').hide();
    }

    // Open results modal
    var resultsModal = new bootstrap.Modal(document.getElementById('resultsModal'));
    resultsModal.show();
    updateSteps(3);
}

/* ------------------------------------------------------------------ */
/*  DL Visualization                                                   */
/* ------------------------------------------------------------------ */

function renderAttentionWeights(weights) {
    if (!weights) { $('#dlAttentionSection').hide(); return; }

    var mods = [
        { key: 'speech', label: 'Speech', cls: 'speech', icon: 'fa-microphone' },
        { key: 'handwriting', label: 'Handwriting', cls: 'handwriting', icon: 'fa-pen' },
        { key: 'gait', label: 'Gait', cls: 'gait', icon: 'fa-walking' }
    ];

    var html = '';
    mods.forEach(function (m) {
        var w = weights[m.key] || 0;
        var p = (w * 100).toFixed(1);
        html += '<div class="attention-bar-row">' +
            '<div class="attention-bar-label"><i class="fas ' + m.icon + '"></i> ' + m.label + '</div>' +
            '<div class="attention-bar-track">' +
            '<div class="attention-bar-fill ' + m.cls + '" style="width:' + p + '%;">' + p + '%</div>' +
            '</div></div>';
    });
    $('#attentionContent').html(html);
    $('#dlAttentionSection').show();
}

function renderFeatureImportance(importance, featureNames) {
    if (!importance) { $('#dlFeatureImportanceSection').hide(); return; }

    var colorMap = { speech: 'var(--accent)', handwriting: 'var(--success)', gait: 'var(--warning)' };
    var html = '';

    ['speech', 'handwriting', 'gait'].forEach(function (mod) {
        var scores = importance[mod];
        var names = (featureNames && featureNames[mod]) ? featureNames[mod] : [];
        if (!scores || scores.length === 0) return;

        var color = colorMap[mod];
        var maxScore = Math.max.apply(null, scores);
        if (maxScore === 0) maxScore = 1;

        html += '<div class="fi-modality-title" style="color:' + color + ';">' +
            '<i class="fas ' + (mod === 'speech' ? 'fa-microphone' : mod === 'handwriting' ? 'fa-pen' : 'fa-walking') + '"></i> ' +
            mod.charAt(0).toUpperCase() + mod.slice(1) + ' Features</div>';

        var indexed = scores.map(function (s, i) { return { score: s, idx: i }; });
        indexed.sort(function (a, b) { return b.score - a.score; });
        var top = indexed.slice(0, 8);

        top.forEach(function (item) {
            var pct = ((item.score / maxScore) * 100).toFixed(0);
            var name = names[item.idx] || ('Feature ' + item.idx);
            if (name.length > 18) name = name.substring(0, 16) + '..';
            html += '<div class="fi-bar-row">' +
                '<div class="fi-bar-name" title="' + (names[item.idx] || '') + '">' + name + '</div>' +
                '<div class="fi-bar-track"><div class="fi-bar-fill" style="width:' + pct + '%;background:' + color + ';opacity:.8;"></div></div>' +
                '<div class="fi-bar-value">' + (item.score * 100).toFixed(1) + '%</div></div>';
        });
        html += '<div style="margin-bottom:.8rem;"></div>';
    });
    $('#featureImportanceContent').html(html);
    $('#dlFeatureImportanceSection').show();
}

function renderSEWeights(seWeights) {
    if (!seWeights) { $('#dlSEWeightsSection').hide(); return; }

    var colorMap = { speech: 'var(--accent)', handwriting: 'var(--success)', gait: 'var(--warning)' };
    var html = '';

    ['speech', 'handwriting', 'gait'].forEach(function (mod) {
        var w = seWeights[mod];
        if (!w || w.length === 0) return;

        var maxW = Math.max.apply(null, w);
        if (maxW === 0) maxW = 1;
        var color = colorMap[mod];

        html += '<div class="fi-modality-title" style="color:' + color + ';">' +
            '<i class="fas ' + (mod === 'speech' ? 'fa-microphone' : mod === 'handwriting' ? 'fa-pen' : 'fa-walking') + '"></i> ' +
            mod.charAt(0).toUpperCase() + mod.slice(1) + ' (' + w.length + ' channels)</div>';

        html += '<div class="se-weights-container">';
        w.forEach(function (v) {
            var h = Math.max(2, (v / maxW) * 28);
            html += '<div class="se-weight-bar" style="height:' + h.toFixed(0) + 'px;background:' + color + ';opacity:' + (0.3 + 0.7 * v / maxW).toFixed(2) + ';"></div>';
        });
        html += '</div><div style="margin-bottom:.8rem;"></div>';
    });
    $('#seWeightsContent').html(html);
    $('#dlSEWeightsSection').show();
}

/* ------------------------------------------------------------------ */
/*  Reset — FIX #3: Also clear combined inputs/status/checkboxes       */
/* ------------------------------------------------------------------ */

function resetForm() {
    // Clear file inputs
    $('#audioFileInput').val('');
    $('#handwritingFileInput').val('');
    $('#gaitFileInput').val('');
    $('#combinedVideoInput').val('');

    // Clear hidden inputs
    $('#speechFeatures').val('');
    $('#handwritingFeatures').val('');
    $('#gaitFeatures').val('');

    extractedFeatures = { speech: null, handwriting: null, gait: null };
    uploadedFilenames = { speech: null, handwriting: null, gait: null };
    referenceCategory = null;

    // Clear ALL status displays
    $('#speechFeatureStatus, #handwritingFeatureStatus, #gaitFeatureStatus').html('');
    $('#audioUploadStatus, #handwritingUploadStatus, #gaitUploadStatus').html('');
    $('#combinedUploadStatus, #combinedFeatureStatus').html('');

    // Reset combined checkboxes
    $('#extractVoiceCheck').prop('checked', true);
    $('#extractHandwritingCheck').prop('checked', false);
    $('#extractGaitCheck').prop('checked', false);

    // Hide previews
    $('#handwritingPreview').hide();

    // Stop recording
    if (isRecording) stopRecording();

    // Disable predict
    $('#predictBtn').prop('disabled', true);
    var tip = bootstrap.Tooltip.getInstance(document.getElementById('predictBtn'));
    if (tip) {
        document.getElementById('predictBtn').setAttribute('data-bs-original-title', 'Load example data or upload a file first');
    }

    // Hide loading and dismiss results modal if open
    $('#loadingSection').hide();
    var modalEl = document.getElementById('resultsModal');
    var modalInstance = bootstrap.Modal.getInstance(modalEl);
    if (modalInstance) modalInstance.hide();

    // Reset DL sections
    $('#dlAttentionSection, #dlFeatureImportanceSection, #dlSEWeightsSection').hide();
    $('#modelTypeBadgeContainer').empty();

    // Reset to first tab
    $('#speech-tab').tab('show');

    // Reset steps
    updateSteps(1);

    showNotification('Form reset — ready for new data', 'info');
}

/* ------------------------------------------------------------------ */
/*  Example Functions                                                  */
/* ------------------------------------------------------------------ */

function loadExample(sampleType, modality) {
    var typeName = sampleType === 'healthy' ? 'Healthy' : "Parkinson's Disease";
    referenceCategory = sampleType;

    var statusElement;
    if (modality === 'speech') statusElement = '#speechFeatureStatus';
    else if (modality === 'handwriting') statusElement = '#handwritingFeatureStatus';
    else if (modality === 'gait') statusElement = '#gaitFeatureStatus';
    else if (modality === 'all') statusElement = '#combinedFeatureStatus';
    else statusElement = '#speechFeatureStatus';

    $(statusElement).html('<div class="alert alert-info small"><i class="fas fa-spinner fa-spin"></i> Loading example data...</div>');
    updateSteps(2);

    fetch('/static/examples/real_examples.json')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var sample = data[sampleType];

            if (modality === 'all') {
                extractedFeatures.speech = sample.speech_features;
                extractedFeatures.handwriting = sample.handwriting_features;
                extractedFeatures.gait = sample.gait_features;
            } else if (modality === 'speech') {
                extractedFeatures.speech = sample.speech_features;
            } else if (modality === 'handwriting') {
                extractedFeatures.handwriting = sample.handwriting_features;
            } else if (modality === 'gait') {
                extractedFeatures.gait = sample.gait_features;
            }

            var features = [];
            if (extractedFeatures.speech) features.push('🎤 Speech: ' + sample.speech_features.length + ' features');
            if (extractedFeatures.handwriting) features.push('✍️ Handwriting: ' + sample.handwriting_features.length + ' features');
            if (extractedFeatures.gait) features.push('🚶 Gait: ' + sample.gait_features.length + ' features');

            var infoMessage = '';
            if (!extractedFeatures.speech && (extractedFeatures.handwriting || extractedFeatures.gait)) {
                infoMessage = '<br><small style="color:var(--info)"><i class="fas fa-info-circle"></i> Tip: Combine with speech for better accuracy.</small>';
            }

            $(statusElement).html(
                '<div class="alert alert-success small">' +
                '<i class="fas fa-database"></i> <strong>' + typeName + ' Sample Loaded</strong><br>' +
                '<small>' + features.join('<br>') + '</small><br>' +
                '<small class="text-muted">Source: Real patient data</small>' +
                infoMessage + '</div>'
            );

            updatePredictButton();
        })
        .catch(function (error) {
            showNotification('Error loading example: ' + error, 'danger');
            $(statusElement).html('');
        });
}

/* ------------------------------------------------------------------ */
/*  View Example Functions (for modals)                                */
/* ------------------------------------------------------------------ */

function viewExampleHealthySpiral() {
    showImageModal('Healthy Spiral - Control Sample', '/static/examples/example_spiral_healthy.jpg',
        'Smooth, confident strokes with consistent size.');
}
function viewExampleHealthySentence() {
    showImageModal('Healthy Writing - Control Sample', '/static/examples/example_sentence_healthy.jpg',
        'Consistent letter size and fluid movements.');
}
function viewExamplePDSpiral() {
    showImageModal("Parkinson's Spiral - Patient Sample", '/static/examples/example_spiral_pd.jpg',
        'Shows micrographia and tremor-induced irregularities.');
}
function viewExamplePDSentence() {
    showImageModal('Micrographia - Patient Sample', '/static/examples/example_sentence_pd.jpg',
        'Progressive reduction in letter size (micrographia).');
}
function viewExamplePDWave() {
    showImageModal('Tremor Wave - Patient Sample', '/static/examples/example_wave_pd.jpg',
        'Irregular wave patterns showing tremor and motor control difficulties.');
}

function viewExampleGait() {
    $('#exampleModalTitle').text('Gait Example - Walking Analysis');
    $('#examplePreviewContent').html(
        '<video controls class="w-100"><source src="/static/examples/example_gait.mp4" type="video/mp4"></video>' +
        '<p class="mt-3 text-muted small">Side-view walking pattern for gait analysis.</p>'
    );
    var modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

function showImageModal(title, imageUrl, description) {
    $('#exampleModalTitle').text(title);
    $('#examplePreviewContent').html(
        '<img src="' + imageUrl + '" class="img-fluid rounded" alt="' + title + '">' +
        '<p class="mt-3 text-muted small">' + description + '</p>'
    );
    var modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

/* ------------------------------------------------------------------ */
/*  Notification — FIX #4 & #7: Use global showNotification from main.js  */
/*  If main.js loaded, it overrides window.showNotification.            */
/*  This is a local fallback in case main.js isn't ready yet.           */
/* ------------------------------------------------------------------ */

if (typeof window.showNotification !== 'function') {
    window.showNotification = function (message, type) {
        type = type || 'info';
        // Use Bootstrap Toast API directly
        var colors = {
            success: { bg: 'rgba(16,185,129,.15)', border: 'rgba(16,185,129,.25)', text: '#10b981' },
            danger:  { bg: 'rgba(244,63,94,.15)',  border: 'rgba(244,63,94,.25)',  text: '#f43f5e' },
            warning: { bg: 'rgba(245,158,11,.15)', border: 'rgba(245,158,11,.25)', text: '#f59e0b' },
            info:    { bg: 'rgba(6,182,212,.12)',  border: 'rgba(6,182,212,.2)',   text: '#06b6d4' }
        };
        var c = colors[type] || colors.info;

        var toastEl = document.createElement('div');
        toastEl.className = 'toast align-items-center border-0';
        toastEl.setAttribute('role', 'alert');
        toastEl.style.cssText = 'background:' + c.bg + ';border:1px solid ' + c.border + ' !important;backdrop-filter:blur(12px);border-radius:12px;color:' + c.text + ';';
        toastEl.innerHTML = '<div class="d-flex"><div class="toast-body" style="font-weight:500;font-size:.88rem;">' + message + '</div>' +
            '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>';

        var container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        container.appendChild(toastEl);

        var toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 4000 });
        toast.show();
        toastEl.addEventListener('hidden.bs.toast', function () { toastEl.remove(); });
    };
}
