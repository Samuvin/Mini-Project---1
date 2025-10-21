// Main JavaScript for Parkinson's Disease Detection System

$(document).ready(function() {
    // Smooth scrolling for anchor links
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        
        var target = this.hash;
        var $target = $(target);
        
        if ($target.length) {
            $('html, body').animate({
                'scrollTop': $target.offset().top - 70
            }, 500);
        }
    });
    
    // Add fade-in animation to cards
    $('.card').addClass('fade-in');
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Check API health on page load
    checkAPIHealth();
});

// Check API health status
function checkAPIHealth() {
    $.ajax({
        url: '/api/health',
        type: 'GET',
        success: function(response) {
            console.log('API Health Status:', response);
            // Only show warning if model is not loaded
            if (response.status === 'healthy' && !response.model_loaded) {
                showNotification('Model not loaded. Please train a model first.', 'warning');
            }
        },
        error: function(xhr, status, error) {
            console.error('API Health Check Failed:', error);
            showNotification('Cannot connect to API', 'danger');
        }
    });
}

// Show notification (toast)
function showNotification(message, type = 'info') {
    // Create toast element
    var toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    // Add to page
    if (!$('#toast-container').length) {
        $('body').append('<div id="toast-container" class="toast-container position-fixed top-0 end-0 p-3"></div>');
    }
    
    var $toast = $(toastHtml);
    $('#toast-container').append($toast);
    
    // Show toast
    var toast = new bootstrap.Toast($toast[0], {
        autohide: true,
        delay: 3000
    });
    toast.show();
    
    // Remove after hiding
    $toast.on('hidden.bs.toast', function() {
        $(this).remove();
    });
}

// Format number as percentage
function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

// Format number to fixed decimals
function formatNumber(value, decimals = 4) {
    return parseFloat(value).toFixed(decimals);
}

// Copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard', 'success');
    }, function(err) {
        console.error('Could not copy text: ', err);
        showNotification('Failed to copy', 'danger');
    });
}

// Parse CSV data
function parseCSV(csvText) {
    var lines = csvText.split('\n');
    var result = [];
    
    // Skip header if present
    var startIndex = 0;
    if (lines.length > 0 && isNaN(lines[0].split(',')[0])) {
        startIndex = 1;
    }
    
    for (var i = startIndex; i < lines.length; i++) {
        var line = lines[i].trim();
        if (line) {
            var values = line.split(',').map(function(val) {
                return parseFloat(val.trim());
            });
            result.push(values);
        }
    }
    
    return result;
}

// Validate feature array
function validateFeatures(features) {
    if (!Array.isArray(features)) {
        return false;
    }
    
    for (var i = 0; i < features.length; i++) {
        if (isNaN(features[i])) {
            return false;
        }
    }
    
    return features.length > 0;
}

// Show loading state
function showLoading(element) {
    $(element).html('<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>');
}

// Hide loading state
function hideLoading(element) {
    $(element).empty();
}

// Example data for testing (42 multimodal features: 22 speech + 10 handwriting + 10 gait)
var exampleSpeechFeatures = [
    119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554,
    0.01109, 0.04374, 0.426, 0.02182, 0.03130, 0.02971, 0.06545,
    0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482,
    2.301442, 0.284654
];

var exampleHandwritingFeatures = [
    0.45, 0.18, 2.2, 0.9, 1.0, 0.35, 9.5, 1.3, 7.2, 0.55
];

var exampleGaitFeatures = [
    1.15, 0.08, 0.38, 0.72, 0.28, 1.0, 100, 0.60, 0.72, 0.15
];

// Combined multimodal features
var exampleFeatures = exampleSpeechFeatures.concat(exampleHandwritingFeatures).concat(exampleGaitFeatures);

// Log initialization
console.log('Main.js loaded successfully');

