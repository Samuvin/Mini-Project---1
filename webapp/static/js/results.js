/**
 * results.js - Prediction results page functionality
 */

(function () {
    'use strict';

    var currentPage = 0;
    var pageSize = 50;
    var currentFilters = {};
    var totalResults = 0;

    /* ------------------------------------------------------------------ */
    /*  Initialize                                                          */
    /* ------------------------------------------------------------------ */

    document.addEventListener('DOMContentLoaded', function () {
        // Check if user is logged in
        if (!window.AuthHelper || !window.AuthHelper.isLoggedIn()) {
            showError('Please log in to view your prediction results.');
            return;
        }

        // Load initial results
        loadResults();

        // Setup form handlers
        document.getElementById('searchForm').addEventListener('submit', function (e) {
            e.preventDefault();
            currentPage = 0;
            loadResults();
        });

        document.getElementById('clearFilters').addEventListener('click', function () {
            clearFilters();
            currentPage = 0;
            loadResults();
        });

        // Pagination handlers
        document.getElementById('prevPage').addEventListener('click', function () {
            if (currentPage > 0) {
                currentPage--;
                loadResults();
            }
        });

        document.getElementById('nextPage').addEventListener('click', function () {
            var maxPage = Math.ceil(totalResults / pageSize) - 1;
            if (currentPage < maxPage) {
                currentPage++;
                loadResults();
            }
        });
    });

    /* ------------------------------------------------------------------ */
    /*  Load Results                                                        */
    /* ------------------------------------------------------------------ */

    function loadResults() {
        var filters = buildFilters();
        currentFilters = filters;

        // Show loading
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('errorState').style.display = 'none';

        // Build query string
        var params = new URLSearchParams();
        params.append('limit', pageSize);
        params.append('skip', currentPage * pageSize);

        if (filters.date_from) params.append('date_from', filters.date_from);
        if (filters.date_to) params.append('date_to', filters.date_to);
        if (filters.result) params.append('result', filters.result);
        if (filters.min_confidence !== null) params.append('min_confidence', filters.min_confidence);
        if (filters.max_confidence !== null) params.append('max_confidence', filters.max_confidence);

        // Fetch results
        fetch('/api/results?' + params.toString(), {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(function (response) {
            if (response.status === 401) {
                window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
                return null;
            }
            return response.json();
        })
        .then(function (data) {
            document.getElementById('loadingSection').style.display = 'none';

            if (!data) return;

            if (data.success) {
                displayResults(data.data.results, data.data.total);
                totalResults = data.data.total;
                updatePagination();
            } else {
                showError(data.error || 'Failed to load results');
            }
        })
        .catch(function (error) {
            document.getElementById('loadingSection').style.display = 'none';
            showError('Error loading results: ' + error.message);
        });
    }

    /* ------------------------------------------------------------------ */
    /*  Build Filters                                                       */
    /* ------------------------------------------------------------------ */

    function buildFilters() {
        var form = document.getElementById('searchForm');
        return {
            date_from: form.dateFrom.value || null,
            date_to: form.dateTo.value || null,
            result: form.resultFilter.value || null,
            min_confidence: form.minConfidence.value ? parseFloat(form.minConfidence.value) : null,
            max_confidence: form.maxConfidence.value ? parseFloat(form.maxConfidence.value) : null
        };
    }

    /* ------------------------------------------------------------------ */
    /*  Clear Filters                                                      */
    /* ------------------------------------------------------------------ */

    function clearFilters() {
        document.getElementById('dateFrom').value = '';
        document.getElementById('dateTo').value = '';
        document.getElementById('resultFilter').value = '';
        document.getElementById('minConfidence').value = '';
        document.getElementById('maxConfidence').value = '';
    }

    /* ------------------------------------------------------------------ */
    /*  Display Results                                                     */
    /* ------------------------------------------------------------------ */

    function displayResults(results, total) {
        var tbody = document.getElementById('resultsTableBody');
        tbody.innerHTML = '';

        if (results.length === 0) {
            document.getElementById('emptyState').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            return;
        }

        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'block';

        results.forEach(function (result) {
            var row = document.createElement('tr');
            
            // Format date to IST
            var dateStr = formatDateToIST(result.created_at);

            // Result badge
            var resultClass = result.prediction_label === 'Healthy' ? 'success' : 'danger';
            var resultBadge = '<span class="badge bg-' + resultClass + '">' + 
                             escapeHtml(result.prediction_label) + '</span>';

            // Confidence percentage
            var confidencePercent = (result.confidence * 100).toFixed(1) + '%';

            // Modalities badges
            var modalitiesHtml = '';
            if (result.modalities_used && result.modalities_used.length > 0) {
                modalitiesHtml = result.modalities_used.map(function (m) {
                    return '<span class="badge bg-secondary me-1">' + escapeHtml(m) + '</span>';
                }).join('');
            } else {
                modalitiesHtml = '<span class="text-muted small">N/A</span>';
            }

            // Model type badge
            var modelBadge = result.model_type === 'dl' ? 
                '<span class="badge bg-primary">Deep Learning</span>' :
                '<span class="badge bg-info">Machine Learning</span>';

            row.innerHTML = 
                '<td><span style="color:var(--text-1);font-weight:500;font-family:var(--mono);font-size:0.9rem">' + escapeHtml(dateStr) + '</span></td>' +
                '<td>' + resultBadge + '</td>' +
                '<td><strong>' + confidencePercent + '</strong></td>' +
                '<td>' + modalitiesHtml + '</td>' +
                '<td>' + modelBadge + '</td>' +
                '<td>' +
                    '<button class="btn btn-sm btn-outline-glass" onclick="viewResultDetails(\'' + 
                    result._id + '\')">' +
                    '<i class="fas fa-eye"></i> View</button>' +
                '</td>';

            tbody.appendChild(row);
        });

        // Update results info
        var start = currentPage * pageSize + 1;
        var end = Math.min((currentPage + 1) * pageSize, total);
        document.getElementById('resultsInfo').textContent = 
            'Showing ' + start + '-' + end + ' of ' + total + ' results';
    }

    /* ------------------------------------------------------------------ */
    /*  Update Pagination                                                   */
    /* ------------------------------------------------------------------ */

    function updatePagination() {
        var maxPage = Math.ceil(totalResults / pageSize) - 1;
        var pageInfo = 'Page ' + (currentPage + 1) + ' of ' + (maxPage + 1);

        document.getElementById('pageInfo').textContent = pageInfo;
        document.getElementById('prevPage').disabled = currentPage === 0;
        document.getElementById('nextPage').disabled = currentPage >= maxPage;
    }

    /* ------------------------------------------------------------------ */
    /*  View Result Details                                                 */
    /* ------------------------------------------------------------------ */

    window.viewResultDetails = function (resultId) {
        fetch('/api/results/' + resultId, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(function (response) {
            if (response.status === 401) {
                window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
                return null;
            }
            return response.json();
        })
        .then(function (data) {
            if (!data || !data.success) {
                alert('Failed to load result details');
                return;
            }

            var result = data.data;
            var modalBody = document.getElementById('resultModalBody');
            
            // Format date to IST
            var dateStr = formatDateToIST(result.created_at);

            // Probabilities
            var probHtml = '';
            if (result.probabilities) {
                for (var key in result.probabilities) {
                    var prob = (result.probabilities[key] * 100).toFixed(2);
                    probHtml += '<div class="mb-2">' +
                               '<strong>' + escapeHtml(key) + ':</strong> ' + prob + '%' +
                               '</div>';
                }
            }

            modalBody.innerHTML = 
                '<div class="row">' +
                    '<div class="col-md-6">' +
                        '<h6>Date</h6>' +
                        '<p style="color:var(--text-1);font-weight:500;font-family:var(--mono);font-size:1rem">' + escapeHtml(dateStr) + '</p>' +
                    '</div>' +
                    '<div class="col-md-6">' +
                        '<h6>Result</h6>' +
                        '<p><span class="badge bg-' + 
                        (result.prediction_label === 'Healthy' ? 'success' : 'danger') + '">' +
                        escapeHtml(result.prediction_label) + '</span></p>' +
                    '</div>' +
                '</div>' +
                '<div class="row mt-3">' +
                    '<div class="col-md-6">' +
                        '<h6>Confidence</h6>' +
                        '<p><strong>' + (result.confidence * 100).toFixed(1) + '%</strong></p>' +
                    '</div>' +
                    '<div class="col-md-6">' +
                        '<h6>Model Type</h6>' +
                        '<p>' + (result.model_type === 'dl' ? 
                            '<span class="badge bg-primary">Deep Learning</span>' :
                            '<span class="badge bg-info">Machine Learning</span>') + '</p>' +
                    '</div>' +
                '</div>' +
                '<div class="row mt-3">' +
                    '<div class="col-12">' +
                        '<h6>Probabilities</h6>' +
                        probHtml +
                    '</div>' +
                '</div>' +
                '<div class="row mt-3">' +
                    '<div class="col-12">' +
                        '<h6>Modalities Used</h6>' +
                        '<p>' + (result.modalities_used && result.modalities_used.length > 0 ?
                            result.modalities_used.map(function (m) {
                                return '<span class="badge bg-secondary me-1">' + escapeHtml(m) + '</span>';
                            }).join('') :
                            '<span class="text-muted">N/A</span>') + '</p>' +
                    '</div>' +
                '</div>';

            var modal = new bootstrap.Modal(document.getElementById('resultModal'));
            modal.show();
        })
        .catch(function (error) {
            alert('Error loading result details: ' + error.message);
        });
    };

    /* ------------------------------------------------------------------ */
    /*  Show Error                                                          */
    /* ------------------------------------------------------------------ */

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorState').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('emptyState').style.display = 'none';
    }

    /* ------------------------------------------------------------------ */
    /*  Format Date to IST                                                 */
    /* ------------------------------------------------------------------ */

    function formatDateToIST(dateString) {
        try {
            // Parse the ISO date string - ensure it's treated as UTC
            // If the string doesn't have 'Z' or timezone, add 'Z' to force UTC interpretation
            var normalizedDateString = dateString;
            if (!normalizedDateString.endsWith('Z') && 
                !normalizedDateString.includes('+') && 
                normalizedDateString.indexOf('-', 10) === -1) {
                normalizedDateString = normalizedDateString + 'Z';
            }
            
            var date = new Date(normalizedDateString);
            
            // Check if date is valid
            if (isNaN(date.getTime())) {
                console.error('Invalid date:', dateString);
                return 'Invalid Date';
            }
            
            // Get UTC timestamp in milliseconds
            var utcTimestamp = date.getTime();
            
            // IST is UTC+5:30 = 5 hours 30 minutes = 19800000 milliseconds
            var istOffsetMs = (5 * 60 + 30) * 60 * 1000;
            var istTimestamp = utcTimestamp + istOffsetMs;
            
            // Create new date object with IST timestamp
            var istDate = new Date(istTimestamp);
            
            // Extract UTC components (which now represent IST after adding offset)
            var day = String(istDate.getUTCDate()).padStart(2, '0');
            var month = String(istDate.getUTCMonth() + 1).padStart(2, '0');
            var year = istDate.getUTCFullYear();
            var hours = String(istDate.getUTCHours()).padStart(2, '0');
            var minutes = String(istDate.getUTCMinutes()).padStart(2, '0');
            var seconds = String(istDate.getUTCSeconds()).padStart(2, '0');
            
            // Format: DD/MM/YYYY, HH:MM:SS IST
            return day + '/' + month + '/' + year + ', ' + hours + ':' + minutes + ':' + seconds + ' IST';
        } catch (e) {
            console.error('Error formatting date:', e, dateString);
            return 'Invalid Date';
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Escape HTML                                                         */
    /* ------------------------------------------------------------------ */

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

})();
