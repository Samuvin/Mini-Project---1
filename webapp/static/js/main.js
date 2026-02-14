// Main JavaScript for Parkinson's Disease Detection System

$(document).ready(function() {
    // Smooth scrolling
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        var target = this.hash;
        var $target = $(target);
        if ($target.length) {
            $('html, body').animate({ 'scrollTop': $target.offset().top - 70 }, 500);
        }
    });

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (el) { return new bootstrap.Tooltip(el); });

    // Check API health
    checkAPIHealth();
});

// Check API health status
function checkAPIHealth() {
    $.ajax({
        url: '/api/health',
        type: 'GET',
        success: function(response) {
            if (response.status === 'healthy' && response.models_loaded && response.models_loaded.length === 0) {
                showNotification('No models loaded. Please train models first.', 'warning');
            }
        },
        error: function() {
            showNotification('Cannot connect to API', 'danger');
        }
    });
}

// Toast notification (dark glass style)
function showNotification(message, type) {
    type = type || 'info';

    // Color map for dark theme
    var colors = {
        success: { bg: 'rgba(25,135,84,.25)', border: 'rgba(25,135,84,.35)', text: '#75d9a0' },
        danger:  { bg: 'rgba(220,53,69,.25)', border: 'rgba(220,53,69,.35)', text: '#ff6b7a' },
        warning: { bg: 'rgba(255,193,7,.2)',  border: 'rgba(255,193,7,.3)',  text: '#ffe066' },
        info:    { bg: 'rgba(13,202,240,.15)', border: 'rgba(13,202,240,.25)', text: '#80e5f5' }
    };
    var c = colors[type] || colors.info;

    var toastHtml =
        '<div class="toast align-items-center border-0" role="alert" ' +
        'style="background:' + c.bg + ';border:1px solid ' + c.border + ' !important;' +
        'backdrop-filter:blur(12px);border-radius:12px;color:' + c.text + ';">' +
        '<div class="d-flex">' +
        '<div class="toast-body" style="font-weight:500;font-size:.9rem;">' + message + '</div>' +
        '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>' +
        '</div></div>';

    if (!$('#toast-container').length) {
        $('body').append('<div id="toast-container" class="toast-container position-fixed top-0 end-0 p-3" style="z-index:9999;"></div>');
    }

    var $toast = $(toastHtml);
    $('#toast-container').append($toast);

    var toast = new bootstrap.Toast($toast[0], { autohide: true, delay: 4000 });
    toast.show();
    $toast.on('hidden.bs.toast', function() { $(this).remove(); });
}

// Format helpers
function formatPercentage(value) { return (value * 100).toFixed(2) + '%'; }
function formatNumber(value, decimals) { return parseFloat(value).toFixed(decimals || 4); }

// Loading helpers (dot loader for dark theme)
function showLoading(element) {
    $(element).html(
        '<div class="text-center py-2">' +
        '<div class="loader-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>' +
        '</div>'
    );
}
function hideLoading(element) { $(element).empty(); }

