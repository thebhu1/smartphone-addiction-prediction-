// Performance Dashboard JavaScript

// Load data when page loads
document.addEventListener('DOMContentLoaded', async function() {
    await loadModelMetrics();
});

// Load model metrics from API
async function loadModelMetrics() {
    try {
        const response = await fetch('/api/model_metrics');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('Loaded metrics:', data);
        
        // Update summary cards
        updateSummaryCards(data);
        
        // Create comparison chart
        createComparisonChart(data.models);
        
        // Populate metrics table
        populateMetricsTable(data.models, data.best_model);
        
        // Create feature importance chart
        if (data.feature_importance) {
            createFeatureImportanceChart(data.feature_importance);
        }
        
    } catch (error) {
        console.error('Error loading metrics:', error);
        showError('Failed to load model metrics: ' + error.message);
    }
}

// Update summary metric cards
function updateSummaryCards(data) {
    // Best model
    const bestModelElement = document.getElementById('bestModel');
    if (bestModelElement) {
        bestModelElement.textContent = data.best_model;
    }
    
    // Best model metrics
    const bestModelMetrics = data.models[data.best_model];
    
    const accuracyElement = document.getElementById('bestAccuracy');
    if (accuracyElement) {
        accuracyElement.textContent = (bestModelMetrics.accuracy * 100).toFixed(2) + '%';
    }
    
    const f1Element = document.getElementById('bestF1Score');
    if (f1Element) {
        f1Element.textContent = (bestModelMetrics.f1_score * 100).toFixed(2) + '%';
    }
}

// Create model comparison chart
function createComparisonChart(models) {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) {
        console.error('Comparison chart canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    const modelNames = Object.keys(models);
    const accuracies = modelNames.map(name => models[name].accuracy * 100);
    const precisions = modelNames.map(name => models[name].precision * 100);
    const recalls = modelNames.map(name => models[name].recall * 100);
    const f1Scores = modelNames.map(name => models[name].f1_score * 100);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'Accuracy',
                    data: accuracies,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Precision',
                    data: precisions,
                    backgroundColor: 'rgba(118, 75, 162, 0.8)',
                    borderColor: 'rgba(118, 75, 162, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Recall',
                    data: recalls,
                    backgroundColor: 'rgba(255, 193, 7, 0.8)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    borderWidth: 1
                },
                {
                    label: 'F1-Score',
                    data: f1Scores,
                    backgroundColor: 'rgba(76, 175, 80, 0.8)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + 
                                   context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Populate metrics table
function populateMetricsTable(models, bestModel) {
    const tbody = document.querySelector('#metricsTable tbody');
    if (!tbody) {
        console.error('Metrics table tbody not found');
        return;
    }
    
    tbody.innerHTML = '';
    
    Object.entries(models).forEach(([modelName, metrics]) => {
        const row = document.createElement('tr');
        
        // Highlight best model
        if (modelName === bestModel) {
            row.classList.add('best-model-row');
        }
        
        row.innerHTML = `
            <td><strong>${modelName}</strong></td>
            <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
            <td>${(metrics.precision * 100).toFixed(2)}%</td>
            <td>${(metrics.recall * 100).toFixed(2)}%</td>
            <td>${(metrics.f1_score * 100).toFixed(2)}%</td>
            <td>${modelName === bestModel ? 
                '<span class="badge badge-success">✅ Best</span>' : 
                '<span class="badge badge-default">-</span>'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// Create feature importance chart (Fixed for Chart.js v3)
function createFeatureImportanceChart(featureImportance) {
    const canvas = document.getElementById('featureImportanceChart');
    if (!canvas) {
        console.error('Feature importance chart canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Sort features by importance
    const sortedFeatures = Object.entries(featureImportance)
        .sort((a, b) => b[1] - a[1]);
    
    const labels = sortedFeatures.map(([name, _]) => formatFeatureName(name));
    const values = sortedFeatures.map(([_, value]) => value * 100);
    
    new Chart(ctx, {
        type: 'bar',  // Changed from 'horizontalBar' to 'bar'
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance (%)',
                data: values,
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',  // This makes it horizontal in Chart.js v3
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: Math.max(...values) * 1.1,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Importance (Random Forest)',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Importance: ' + context.parsed.x.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Format feature names for display
function formatFeatureName(name) {
    return name.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Show error message
function showError(message) {
    // Create a more user-friendly error display
    const container = document.querySelector('.container');
    if (container) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            text-align: center;
        `;
        errorDiv.innerHTML = `
            <h3>⚠️ Error Loading Metrics</h3>
            <p>${message}</p>
            <p style="margin-top: 10px;">
                <strong>Solution:</strong> Please run <code>python train_model.py</code> first to train the models.
            </p>
        `;
        container.insertBefore(errorDiv, container.firstChild);
    }
    
    // Also show browser alert
    alert('Error: ' + message);
}