/**
 * main.js — Vanilla JS utilities for Parkinson's Disease Prediction System.
 *
 * No jQuery dependency. Provides:
 *  - Unified toast notification (showNotification) as single source of truth
 *  - Smooth scrolling for anchor links
 *  - Bootstrap tooltip initialization
 *  - API health check
 *  - IntersectionObserver for scroll-triggered fade-in animations
 *  - Helper functions
 */

(function () {
    'use strict';

    /* -------------------------------------------------------------- */
    /*  Unified Toast Notification (single source of truth)            */
    /* -------------------------------------------------------------- */

    var toastColors = {
        success: { bg: 'rgba(16,185,129,.15)', border: 'rgba(16,185,129,.25)', text: '#10b981' },
        danger:  { bg: 'rgba(244,63,94,.15)',  border: 'rgba(244,63,94,.25)',  text: '#f43f5e' },
        warning: { bg: 'rgba(245,158,11,.15)', border: 'rgba(245,158,11,.25)', text: '#f59e0b' },
        info:    { bg: 'rgba(6,182,212,.12)',  border: 'rgba(6,182,212,.2)',   text: '#06b6d4' }
    };

    window.showNotification = function (message, type) {
        type = type || 'info';
        var c = toastColors[type] || toastColors.info;

        var toastEl = document.createElement('div');
        toastEl.className = 'toast align-items-center border-0';
        toastEl.setAttribute('role', 'alert');
        toastEl.style.cssText =
            'background:' + c.bg + ';' +
            'border:1px solid ' + c.border + ' !important;' +
            'backdrop-filter:blur(12px);' +
            'border-radius:12px;' +
            'color:' + c.text + ';';

        toastEl.innerHTML =
            '<div class="d-flex">' +
            '<div class="toast-body" style="font-weight:500;font-size:.88rem;">' + message + '</div>' +
            '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>' +
            '</div>';

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

    /* -------------------------------------------------------------- */
    /*  DOMContentLoaded setup                                         */
    /* -------------------------------------------------------------- */

    document.addEventListener('DOMContentLoaded', function () {

        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(function (link) {
            link.addEventListener('click', function (e) {
                var hash = this.getAttribute('href');
                if (hash === '#' || hash.length < 2) return;
                var target = document.querySelector(hash);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });

        // Initialize Bootstrap tooltips
        var tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltipEls.forEach(function (el) { new bootstrap.Tooltip(el); });

        // API health check (vanilla fetch)
        checkAPIHealth();

        // Scroll-triggered fade-in animations
        initScrollAnimations();
    });

    /* -------------------------------------------------------------- */
    /*  API Health Check                                                */
    /* -------------------------------------------------------------- */

    function checkAPIHealth() {
        fetch('/api/health')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.status === 'healthy' && data.models_loaded && data.models_loaded.length === 0) {
                    showNotification('No models loaded. Please train models first.', 'warning');
                }
            })
            .catch(function () {
                // Silently fail — API might not be ready yet
            });
    }

    /* -------------------------------------------------------------- */
    /*  IntersectionObserver for fade-in animations                    */
    /* -------------------------------------------------------------- */

    function initScrollAnimations() {
        // Check for reduced motion preference
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

        var elements = document.querySelectorAll('.fade-in-up');
        if (!elements.length || !('IntersectionObserver' in window)) return;

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '';
                    entry.target.style.transform = '';
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

        elements.forEach(function (el) {
            // Only observe elements not already visible
            var rect = el.getBoundingClientRect();
            if (rect.top > window.innerHeight) {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                observer.observe(el);
            }
        });
    }

    /* -------------------------------------------------------------- */
    /*  Helper functions                                                */
    /* -------------------------------------------------------------- */

    window.formatPercentage = function (value) {
        return (value * 100).toFixed(2) + '%';
    };

    window.formatNumber = function (value, decimals) {
        return parseFloat(value).toFixed(decimals || 4);
    };

})();
