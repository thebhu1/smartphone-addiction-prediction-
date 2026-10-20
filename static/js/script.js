// Main JavaScript for Smartphone Addiction Prediction

const TIME_FIELDS = ['daily_screen_time', 'social_media_time', 'gaming_time', 'night_usage'];
let currentUnit = 'hours';
let latestSubmittedInput = null;
let latestPredictionPayload = null;
let toastTimer = null;

// Wait for DOM to load
window.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const unitToggle = document.getElementById('unitToggle');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const hoursLabel = document.getElementById('hoursLabel');
    const minutesLabel = document.getElementById('minutesLabel');

    if (form) {
        form.addEventListener('submit', handleFormSubmit);
        form.addEventListener('reset', () => {
            hideResults();
            clearPredictionCache();
            resetUnitToggleToHours();
        });
    }

    if (unitToggle) {
        unitToggle.addEventListener('change', handleUnitToggle);
    }

    if (hoursLabel) {
        hoursLabel.addEventListener('click', () => {
            if (unitToggle && unitToggle.checked) {
                unitToggle.checked = false;
                handleUnitToggle();
            }
        });
    }

    if (minutesLabel) {
        minutesLabel.addEventListener('click', () => {
            if (unitToggle && !unitToggle.checked) {
                unitToggle.checked = true;
                handleUnitToggle();
            }
        });
    }

    if (generateReportBtn) {
        generateReportBtn.addEventListener('click', handleGenerateReport);
        setGenerateReportEnabled(false);
    }
});

function resetPredictionView() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.reset();
    }
    hideResults();
    clearPredictionCache();
    resetUnitToggleToHours();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function clearPredictionCache() {
    latestSubmittedInput = null;
    latestPredictionPayload = null;
    setGenerateReportEnabled(false);
}

function setGenerateReportEnabled(enabled) {
    const button = document.getElementById('generateReportBtn');
    if (!button) return;

    button.disabled = !enabled;
    button.classList.toggle('btn-disabled', !enabled);
}

function resetUnitToggleToHours() {
    const unitToggle = document.getElementById('unitToggle');
    if (!unitToggle) return;

    currentUnit = 'hours';
    unitToggle.checked = false;

    const hoursLabel = document.getElementById('hoursLabel');
    const minutesLabel = document.getElementById('minutesLabel');

    if (hoursLabel) hoursLabel.classList.add('unit-active');
    if (minutesLabel) minutesLabel.classList.remove('unit-active');

    document.querySelectorAll('.time-unit').forEach((span) => {
        span.textContent = 'hours';
    });

    // Keep min/max/step aligned with hour mode after reset values are restored
    setTimeout(() => {
        TIME_FIELDS.forEach((fieldId) => {
            const input = document.getElementById(fieldId);
            if (!input) return;
            input.min = input.dataset.minHours;
            input.max = input.dataset.maxHours;
            input.step = input.dataset.stepHours;
        });
    }, 0);
}

function handleUnitToggle() {
    const toggle = document.getElementById('unitToggle');
    if (!toggle) return;

    const isMinutes = toggle.checked;
    currentUnit = isMinutes ? 'minutes' : 'hours';

    const hoursLabel = document.getElementById('hoursLabel');
    const minutesLabel = document.getElementById('minutesLabel');

    if (hoursLabel) hoursLabel.classList.toggle('unit-active', !isMinutes);
    if (minutesLabel) minutesLabel.classList.toggle('unit-active', isMinutes);

    document.querySelectorAll('.time-unit').forEach((span) => {
        span.textContent = currentUnit;
    });

    TIME_FIELDS.forEach((fieldId) => {
        const input = document.getElementById(fieldId);
        if (!input) return;

        const currentValue = parseFloat(input.value) || 0;

        if (isMinutes) {
            input.value = Math.round(currentValue * 60);
            input.min = input.dataset.minMinutes;
            input.max = input.dataset.maxMinutes;
            input.step = input.dataset.stepMinutes;
        } else {
            input.value = parseFloat((currentValue / 60).toFixed(2));
            input.min = input.dataset.minHours;
            input.max = input.dataset.maxHours;
            input.step = input.dataset.stepHours;
        }
    });
}

async function handleFormSubmit(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = {};

    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });

    if (currentUnit === 'minutes') {
        TIME_FIELDS.forEach((fieldId) => {
            if (data[fieldId] !== undefined) {
                data[fieldId] = parseFloat((data[fieldId] / 60).toFixed(4));
            }
        });
    }

    latestSubmittedInput = { ...data };

    showLoading();
    hideResults();
    setGenerateReportEnabled(false);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result.prediction);
            latestPredictionPayload = {
                inputs: latestSubmittedInput,
                prediction: result.prediction
            };
            setGenerateReportEnabled(true);
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction request failed:', error);
        showError('Failed to connect to server. Please try again.');
    } finally {
        hideLoading();
    }
}

function showLoading() {
    const loadingDiv = document.getElementById('loadingIndicator');
    if (!loadingDiv) return;

    loadingDiv.style.display = 'block';
    loadingDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideLoading() {
    const loadingDiv = document.getElementById('loadingIndicator');
    if (loadingDiv) {
        loadingDiv.style.display = 'none';
    }
}

function hideResults() {
    const resultsDiv = document.getElementById('resultsSection');
    if (resultsDiv) {
        resultsDiv.style.display = 'none';
    }
}

function displayResults(prediction) {
    const resultsDiv = document.getElementById('resultsSection');
    if (!resultsDiv) return;

    resultsDiv.style.display = 'block';

    const riskCard = document.getElementById('riskLevelCard');
    const riskTitle = document.getElementById('riskTitle');
    const riskDescription = document.getElementById('riskDescription');
    const riskIcon = riskCard ? riskCard.querySelector('.risk-icon') : null;

    const riskClassMap = {
        green: 'low',
        orange: 'moderate',
        red: 'high',
        low: 'low',
        moderate: 'moderate',
        high: 'high'
    };

    const normalizedRiskClass = riskClassMap[(prediction.risk_color || '').toLowerCase()] || 'moderate';

    if (riskCard) {
        riskCard.className = `risk-level-card ${normalizedRiskClass}`;
    }

    if (riskTitle) {
        riskTitle.textContent = prediction.title || `${prediction.addiction_label || 'Moderate'} Risk`;
    }

    if (riskDescription) {
        riskDescription.textContent = prediction.description || '';
    }

    if (riskIcon) {
        const iconMap = {
            low: 'OK',
            moderate: 'WARN',
            high: 'ALERT'
        };
        riskIcon.textContent = iconMap[normalizedRiskClass] || 'RISK';
    }

    updateProbabilityBars(prediction.probabilities || {});
    updateRecommendations(prediction.recommendations || []);
    updateRiskExplanation(prediction.risk_explanation, prediction.summary_sentence);

    setTimeout(() => {
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function updateProbabilityBars(probabilities) {
    const lowPercent = toPercent(probabilities.Low);
    const moderatePercent = toPercent(probabilities.Moderate);
    const highPercent = toPercent(probabilities.High);

    const probLow = document.getElementById('probLow');
    const probLowValue = document.getElementById('probLowValue');
    const probModerate = document.getElementById('probModerate');
    const probModerateValue = document.getElementById('probModerateValue');
    const probHigh = document.getElementById('probHigh');
    const probHighValue = document.getElementById('probHighValue');

    if (probLow) probLow.style.width = `${lowPercent}%`;
    if (probLowValue) probLowValue.textContent = `${lowPercent.toFixed(1)}%`;

    if (probModerate) probModerate.style.width = `${moderatePercent}%`;
    if (probModerateValue) probModerateValue.textContent = `${moderatePercent.toFixed(1)}%`;

    if (probHigh) probHigh.style.width = `${highPercent}%`;
    if (probHighValue) probHighValue.textContent = `${highPercent.toFixed(1)}%`;
}

function toPercent(value) {
    const numeric = Number(value || 0);
    return numeric <= 1 ? numeric * 100 : numeric;
}

function updateRecommendations(recommendations) {
    const list = document.getElementById('recommendationsList');
    if (!list) return;

    list.innerHTML = '';
    recommendations.forEach((rec) => {
        const li = document.createElement('li');
        li.textContent = rec;
        list.appendChild(li);
    });
}

function updateRiskExplanation(explanation, summary) {
    const explanationText = document.getElementById('riskExplanationText');
    const summaryText = document.getElementById('summarySentenceText');

    if (explanationText) {
        explanationText.textContent = explanation || 'Risk explanation is based on your submitted digital behavior indicators.';
    }

    if (summaryText) {
        summaryText.textContent = summary || '';
    }
}

async function handleGenerateReport() {
    if (!latestPredictionPayload) {
        showError('Please generate a prediction first before downloading the report.');
        return;
    }

    const button = document.getElementById('generateReportBtn');
    if (!button) return;

    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = 'Generating...';

    try {
        const response = await fetch('/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(latestPredictionPayload)
        });

        if (!response.ok) {
            let message = 'Failed to generate report.';
            try {
                const payload = await response.json();
                if (payload && payload.error) {
                    message = payload.error;
                }
            } catch (parseError) {
                console.warn('Unable to parse error payload:', parseError);
            }
            throw new Error(message);
        }

        const blob = await response.blob();
        const contentDisposition = response.headers.get('Content-Disposition') || '';
        const filename = extractFilename(contentDisposition) || buildFallbackFilename();

        const objectUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(objectUrl);

        showToast('Report generated successfully');
    } catch (error) {
        console.error('Report generation failed:', error);
        showError(error.message || 'Unable to generate report.');
    } finally {
        button.disabled = false;
        button.textContent = originalLabel;
    }
}

function extractFilename(contentDisposition) {
    const filenameStarMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (filenameStarMatch && filenameStarMatch[1]) {
        return decodeURIComponent(filenameStarMatch[1]);
    }

    const filenameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    return filenameMatch && filenameMatch[1] ? filenameMatch[1] : null;
}

function buildFallbackFilename() {
    const isoStamp = new Date().toISOString().replace(/[:.]/g, '-');
    return `smartphone_addiction_report_${isoStamp}.pdf`;
}

function showToast(message) {
    const toast = document.getElementById('reportToast');
    if (!toast) return;

    toast.textContent = message;
    toast.classList.add('show');

    if (toastTimer) {
        clearTimeout(toastTimer);
    }

    toastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 2600);
}

function showError(message) {
    alert(`Error: ${message}`);
}

// Smooth scrolling for in-page anchors
Array.from(document.querySelectorAll('a[href^="#"]')).forEach((anchor) => {
    anchor.addEventListener('click', function onAnchorClick(event) {
        event.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
