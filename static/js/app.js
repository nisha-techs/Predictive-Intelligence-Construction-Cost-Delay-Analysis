/* ==========================================================================
   CONSTRUCTAI - PREDICTIVE INTELLIGENCE PLATFORM JAVASCRIPT
   Single Page Application (SPA) Controller & REST API Client
   ========================================================================== */

const API_BASE_URL = 'http://127.0.0.1:5000';

let progressChartInstance = null;
let currentAnalysisFilter = 'all'; // 'all', 'monitored', 'high-risk'
let cachedProjectsList = [];
let projectMonitoringInfoMap = {}; // { [pid]: { isMonitored: boolean, count: number, latestRisk: string|null } }

function navigateToProjectAnalysis(filterMode = 'all') {
    currentAnalysisFilter = filterMode;
    switchView('project-analysis');
}

function resetAnalysisFilter() {
    currentAnalysisFilter = 'all';
    populateProjectDropdowns();
}

// Currency Formatter (Indian Rupee format: ₹1,55,90,454)
function formatINR(amount) {
    if (isNaN(amount) || amount === null) return '₹0';
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0
    }).format(amount);
}

// Date Formatter (DD/MM/YYYY format: 01/09/2026)
function formatDate(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    const day = String(d.getDate()).padStart(2, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const year = d.getFullYear();
    return `${day}/${month}/${year}`;
}

// Toast Notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const iconClass = type === 'success' ? 'fa-circle-check text-emerald-400' : 'fa-circle-exclamation text-rose-400';
    toast.innerHTML = `<i class="fa-solid ${iconClass}"></i> <span>${message}</span>`;
    
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// SPA Navigation Router
function switchView(viewId) {
    // Hide all view panels
    document.querySelectorAll('.view-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // Deactivate nav items
    document.querySelectorAll('.nav-item').forEach(nav => {
        nav.classList.remove('active');
    });

    const targetPanel = document.getElementById(`view-${viewId}`);
    const targetNav = document.querySelector(`.nav-item[data-view="${viewId}"]`);

    if (targetPanel) {
        targetPanel.classList.add('active');
        if (targetNav) targetNav.classList.add('active');
        
        // Update header title
        const titleMap = {
            'dashboard': ['Project Overview & Monitoring Dashboard', 'Construction Cost Estimation and Project Delay Analysis'],
            'cost-estimation': ['Stage 1: Pre-Construction Cost Estimation', 'Estimate project cost and analyze SHAP contributing factors'],
            'monitoring': ['Stage 2: Ongoing Execution Monitoring', 'Analyze delay forecast, risk category, and site progress'],
            'project-analysis': ['Project Lifecycle & Analysis', 'View milestone tracking, progress history, and snapshot records'],
            'model-performance': ['Machine Learning Model Performance', 'Evaluation metrics for Stage 1 and Stage 2 models']
        };

        if (titleMap[viewId]) {
            document.getElementById('page-title').innerText = titleMap[viewId][0];
            document.getElementById('page-subtitle').innerText = titleMap[viewId][1];
        }

        // Trigger view-specific data loading
        if (viewId === 'dashboard') loadDashboardData();
        if (viewId === 'monitoring') populateProjectDropdowns();
        if (viewId === 'project-analysis') populateProjectDropdowns();
        if (viewId === 'model-performance') loadModelPerformanceData();
    }
}

// Event Listeners for Nav
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-item').forEach(nav => {
        nav.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = nav.getAttribute('data-view');
            switchView(viewId);
            window.location.hash = viewId;
        });
    });

    // Check URL Hash on Load (Default to 'dashboard')
    const initialHash = window.location.hash.replace('#', '');
    const validViews = ['dashboard', 'cost-estimation', 'monitoring', 'project-analysis', 'model-performance'];
    const targetView = validViews.includes(initialHash) ? initialHash : 'dashboard';
    switchView(targetView);

    // Refresh Data Button
    document.getElementById('btn-refresh-data').addEventListener('click', () => {
        const activeNav = document.querySelector('.nav-item.active');
        const activeView = activeNav ? activeNav.getAttribute('data-view') : 'dashboard';
        switchView(activeView);
        showToast('Data refreshed successfully', 'success');
    });

    // Form Event Listeners
    setupFormControllers();

    // Dashboard KPI Card Interactive Event Listeners
    const cardMonitored = document.getElementById('card-monitored-projects');
    if (cardMonitored) {
        cardMonitored.addEventListener('click', () => navigateToProjectAnalysis('monitored'));
        cardMonitored.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                navigateToProjectAnalysis('monitored');
            }
        });
    }

    const cardHighRisk = document.getElementById('card-high-risk-projects');
    if (cardHighRisk) {
        cardHighRisk.addEventListener('click', () => navigateToProjectAnalysis('high-risk'));
        cardHighRisk.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                navigateToProjectAnalysis('high-risk');
            }
        });
    }
});

// --------------------------------------------------------------------------
// 1. DASHBOARD CONTROLLER
// --------------------------------------------------------------------------
let dashboardLoadRequestId = 0;

async function loadDashboardData() {
    const currentRequestId = ++dashboardLoadRequestId;
    try {
        const res = await fetch(`${API_BASE_URL}/api/projects`);
        const data = await res.json();

        if (currentRequestId !== dashboardLoadRequestId) return;

        if (!data.success) {
            showToast('Unable to load projects for dashboard', 'error');
            return;
        }

        const projects = data.projects || [];
        cachedProjectsList = projects;
        projectMonitoringInfoMap = {};
        document.getElementById('stat-total-projects').innerText = projects.length;

        if (projects.length === 0) {
            document.getElementById('stat-monitored-projects').innerText = '0';
            document.getElementById('stat-high-risk-projects').innerText = '0';
            document.getElementById('stat-avg-cost').innerText = '₹0';
            document.getElementById('dashboard-projects-table-body').innerHTML = `
                <tr><td colspan="7" class="text-center py-6 text-slate-400">No construction projects created yet. Click "New Estimate" to create one.</td></tr>
            `;
            return;
        }

        let totalCost = 0;
        projects.forEach(p => totalCost += (p.predicted_cost || 0));
        const avgCost = totalCost / projects.length;
        document.getElementById('stat-avg-cost').innerText = formatINR(avgCost);

        // Fetch monitoring stats across projects to compute monitored count & high risk
        let monitoredCount = 0;
        let highRiskCount = 0;

        const tableRows = [];

        for (const p of projects) {
            const mRes = await fetch(`${API_BASE_URL}/api/projects/${p.project_id}/monitoring`);
            const mData = await mRes.json();

            if (currentRequestId !== dashboardLoadRequestId) return;

            let statusBadge = '<span class="badge badge-sky">Pre-Construction</span>';
            if (mData.success && mData.count > 0) {
                monitoredCount++;
                const latest = mData.monitoring_history[mData.monitoring_history.length - 1];
                const risk = latest.delay_risk;
                projectMonitoringInfoMap[p.project_id] = {
                    isMonitored: true,
                    count: mData.count,
                    latestRisk: risk
                };

                if (risk === 'High') {
                    highRiskCount++;
                    statusBadge = '<span class="risk-badge risk-high" style="padding:2px 10px; font-size:11px;">HIGH RISK</span>';
                } else if (risk === 'Medium') {
                    statusBadge = '<span class="risk-badge risk-medium" style="padding:2px 10px; font-size:11px;">MEDIUM RISK</span>';
                } else {
                    statusBadge = '<span class="risk-badge risk-low" style="padding:2px 10px; font-size:11px;">LOW RISK</span>';
                }
            } else {
                projectMonitoringInfoMap[p.project_id] = {
                    isMonitored: false,
                    count: 0,
                    latestRisk: null
                };
            }

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${p.project_id}</strong></td>
                <td>${p.project_type}</td>
                <td>${p.location}</td>
                <td>${p.area_sqft.toLocaleString()} sq.ft / ${p.number_of_floors} floors</td>
                <td class="text-emerald-400 font-semibold">${formatINR(p.predicted_cost)}</td>
                <td>${formatDate(p.created_at)}</td>
                <td>
                    <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="inspectProjectAnalysis('${p.project_id}')">
                        <i class="fa-solid fa-eye"></i> Inspect
                    </button>
                </td>
            `;
            tableRows.push(tr);
        }

        if (currentRequestId !== dashboardLoadRequestId) return;

        const tableBody = document.getElementById('dashboard-projects-table-body');
        tableBody.replaceChildren(...tableRows);

        document.getElementById('stat-monitored-projects').innerText = monitoredCount;
        document.getElementById('stat-high-risk-projects').innerText = highRiskCount;

    } catch (err) {
        console.error("Dashboard error:", err);
        showToast("Unable to connect to backend server", "error");
    }
}

function inspectProjectAnalysis(projectId) {
    currentAnalysisFilter = 'all';
    switchView('project-analysis');
    setTimeout(() => {
        const select = document.getElementById('analysis-select-project');
        if (select) {
            select.value = projectId;
            select.dispatchEvent(new Event('change'));
        }
    }, 150);
}

// --------------------------------------------------------------------------
// 2. STAGE 1 COST ESTIMATION CONTROLLER
// --------------------------------------------------------------------------
function setupFormControllers() {
    const s1Form = document.getElementById('form-stage1-cost');
    s1Form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitBtn = document.getElementById('btn-submit-s1');
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Estimating Cost...`;

        const formData = new FormData(s1Form);
        const payload = {};
        formData.forEach((value, key) => {
            if (['Area_sqft', 'Number_of_Floors', 'Material_Cost', 'Labour_Cost', 'Labour_Count', 'Equipment_Cost', 'Planned_Duration'].includes(key)) {
                payload[key] = parseFloat(value);
            } else {
                payload[key] = value;
            }
        });

        try {
            const res = await fetch(`${API_BASE_URL}/api/stage1/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok || !data.success) {
                showToast(data.message || 'Stage 1 prediction failed', 'error');
                return;
            }

            renderStage1Results(data);
            populateProjectDropdowns();
            showToast(`Stage 1 Cost Estimation Complete! Project ID: ${data.project_id}`, 'success');

        } catch (err) {
            console.error("Stage 1 error:", err);
            showToast("Server connection error during Stage 1 prediction", "error");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<i class="fa-solid fa-calculator"></i> Estimate Construction Cost`;
        }
    });

    // Stage 2 Monitoring Form Listener
    const s2Form = document.getElementById('form-stage2-monitoring');
    s2Form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitBtn = document.getElementById('btn-submit-s2');
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Delay & Risk...`;

        const formData = new FormData(s2Form);
        const payload = {};
        formData.forEach((value, key) => {
            if (['Planned_Progress_Percent', 'Actual_Progress_Percent', 'Actual_Labour_Count', 'Material_Availability_Percent', 'Equipment_Availability_Percent'].includes(key)) {
                payload[key] = parseFloat(value);
            } else {
                payload[key] = value;
            }
        });

        const isCompleteChecked = document.getElementById('s2-complete-monitoring') ? document.getElementById('s2-complete-monitoring').checked : false;
        payload['is_completed'] = isCompleteChecked ? 1 : 0;

        try {
            const res = await fetch(`${API_BASE_URL}/api/stage2/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok || !data.success) {
                showToast(data.message || 'Stage 2 monitoring prediction failed', 'error');
                return;
            }

            renderStage2Results(data);
            if (projectMonitoringInfoMap[data.project_id]) {
                delete projectMonitoringInfoMap[data.project_id];
            }
            populateProjectDropdowns();
            const msg = isCompleteChecked 
                ? `Monitoring Completed & Final Analysis Generated for ${data.project_id}!` 
                : `Ongoing Analysis Complete for ${data.project_id}!`;
            showToast(msg, 'success');

        } catch (err) {
            console.error("Stage 2 error:", err);
            showToast("Server connection error during Stage 2 prediction", "error");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<i class="fa-solid fa-clock-rotate-left"></i> Analyze Delay & Risk`;
        }
    });
}

function renderStage1Results(data) {
    document.getElementById('s1-placeholder').classList.add('hidden');
    const wrapper = document.getElementById('s1-results');
    wrapper.classList.remove('hidden');

    document.getElementById('s1-res-project-id').innerText = data.project_id;
    document.getElementById('s1-res-cost').innerText = formatINR(data.prediction.total_construction_cost);

    // Render SHAP Bars
    const shapContainer = document.getElementById('s1-shap-bars');
    shapContainer.innerHTML = '';

    const topFactors = data.explanation.top_factors || [];
    const maxMag = topFactors.length > 0 ? topFactors[0].magnitude : 1;

    topFactors.forEach(factor => {
        const pct = Math.min(100, Math.max(12, (factor.magnitude / maxMag) * 100));
        const isPositive = factor.direction === 'increases' || factor.shap_value > 0;
        const dirLabel = isPositive ? 'Higher contribution' : 'Lower contribution';
        const fillClass = isPositive ? 'shap-positive' : 'shap-negative';
        const badgeClass = isPositive ? 'badge-pos' : 'badge-neg';
        const signSymbol = isPositive ? '+' : '-';
        const formattedVal = `${signSymbol}₹${Math.abs(factor.magnitude).toLocaleString('en-IN', {maximumFractionDigits: 0})}`;

        const item = document.createElement('div');
        item.className = 'shap-bar-item';
        item.innerHTML = `
            <div class="shap-meta">
                <div class="shap-feature-info">
                    <strong>${factor.feature}</strong>
                    <span class="shap-dir-badge ${badgeClass}">${dirLabel}</span>
                </div>
                <span class="shap-val-text">${formattedVal}</span>
            </div>
            <div class="shap-track">
                <div class="shap-fill ${fillClass}" style="width: ${pct}%"></div>
            </div>
        `;
        shapContainer.appendChild(item);
    });

    // Render Recommendations
    const recsContainer = document.getElementById('s1-recommendations-list');
    recsContainer.innerHTML = '';

    const recs = data.recommendations || [];
    recs.forEach(rec => {
        const prioClass = `prio-${rec.priority.toLowerCase()}`;
        const card = document.createElement('div');
        card.className = `rec-card ${prioClass}`;
        card.innerHTML = `
            <div class="rec-head">
                <span class="rec-factor">${rec.factor}</span>
                <span class="badge badge-${rec.priority === 'HIGH' ? 'rose' : rec.priority === 'MEDIUM' ? 'amber' : 'sky'}">${rec.priority} PRIORITY</span>
            </div>
            <div class="rec-issue">${rec.issue}</div>
            <div class="rec-action"><i class="fa-solid fa-angle-right text-sky-400"></i> ${rec.action}</div>
        `;
        recsContainer.appendChild(card);
    });
}

// --------------------------------------------------------------------------
// 3. STAGE 2 ONGOING MONITORING CONTROLLER
// --------------------------------------------------------------------------
let dropdownLoadRequestId = 0;

async function populateProjectDropdowns() {
    const currentRequestId = ++dropdownLoadRequestId;
    try {
        const s2Select = document.getElementById('s2-select-project');
        const anSelect = document.getElementById('analysis-select-project');
        if (!s2Select || !anSelect) return;

        const res = await fetch(`${API_BASE_URL}/api/projects`);
        const data = await res.json();
        if (currentRequestId !== dropdownLoadRequestId) return;
        if (!data.success) return;

        cachedProjectsList = data.projects || [];

        // Ensure monitoring info map is populated for all projects
        for (const p of cachedProjectsList) {
            if (!projectMonitoringInfoMap[p.project_id]) {
                const mRes = await fetch(`${API_BASE_URL}/api/projects/${p.project_id}/monitoring`);
                const mData = await mRes.json();
                if (currentRequestId !== dropdownLoadRequestId) return;
                if (mData.success && mData.count > 0) {
                    const latest = mData.monitoring_history[mData.monitoring_history.length - 1];
                    projectMonitoringInfoMap[p.project_id] = {
                        isMonitored: true,
                        count: mData.count,
                        latestRisk: latest.delay_risk
                    };
                } else {
                    projectMonitoringInfoMap[p.project_id] = {
                        isMonitored: false,
                        count: 0,
                        latestRisk: null
                    };
                }
            }
        }

        if (currentRequestId !== dropdownLoadRequestId) return;

        // Populate Stage 2 dropdown (all projects)
        const currentS2Val = s2Select.value;
        s2Select.innerHTML = '<option value="">-- Select Stored Project --</option>';
        cachedProjectsList.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.project_id;
            opt.innerText = `${p.project_id} - ${p.project_type} (${p.location})`;
            s2Select.appendChild(opt);
        });
        if (currentS2Val) s2Select.value = currentS2Val;

        // Filter projects for Project Analysis dropdown based on currentAnalysisFilter
        let filteredProjects = cachedProjectsList;
        if (currentAnalysisFilter === 'monitored') {
            filteredProjects = cachedProjectsList.filter(p => {
                const info = projectMonitoringInfoMap[p.project_id];
                return info && info.isMonitored;
            });
        } else if (currentAnalysisFilter === 'high-risk') {
            filteredProjects = cachedProjectsList.filter(p => {
                const info = projectMonitoringInfoMap[p.project_id];
                return info && info.latestRisk && info.latestRisk.toLowerCase() === 'high';
            });
        }

        const filterBadge = document.getElementById('analysis-filter-badge');
        const filterText = document.getElementById('analysis-filter-text');

        if (currentAnalysisFilter === 'monitored') {
            if (filterBadge && filterText) {
                filterBadge.className = 'badge badge-amber';
                filterText.innerText = `Filter: Monitored Projects (${filteredProjects.length})`;
                filterBadge.classList.remove('hidden');
            }
        } else if (currentAnalysisFilter === 'high-risk') {
            if (filterBadge && filterText) {
                filterBadge.className = 'badge badge-rose';
                filterText.innerText = `Filter: High Risk Projects (${filteredProjects.length})`;
                filterBadge.classList.remove('hidden');
            }
        } else {
            if (filterBadge) filterBadge.classList.add('hidden');
        }

        anSelect.innerHTML = '<option value="">-- Select Project ID --</option>';
        filteredProjects.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.project_id;
            const info = projectMonitoringInfoMap[p.project_id];
            const riskTag = (info && info.latestRisk) ? ` [${info.latestRisk.toUpperCase()} RISK]` : '';
            opt.innerText = `${p.project_id} - ${p.project_type} (${p.location})${riskTag}`;
            anSelect.appendChild(opt);
        });

        // Add Listeners
        s2Select.onchange = async () => {
            const pid = s2Select.value;
            const ctxCard = document.getElementById('s2-project-context');
            if (!pid) {
                ctxCard.classList.add('hidden');
                return;
            }

            const pRes = await fetch(`${API_BASE_URL}/api/projects/${pid}`);
            const pData = await pRes.json();
            if (pData.success) {
                const p = pData.project;
                document.getElementById('ctx-type').innerText = p.project_type;
                document.getElementById('ctx-loc').innerText = p.location;
                document.getElementById('ctx-scale').innerText = `${p.area_sqft.toLocaleString()} sq.ft (${p.number_of_floors} fl)`;
                document.getElementById('ctx-cost').innerText = formatINR(p.predicted_cost);
                ctxCard.classList.remove('hidden');

                const chk = document.getElementById('s2-complete-monitoring');
                if (chk) {
                    chk.checked = p.is_completed === 1 || p.is_completed === true;
                }
            }
        };

        anSelect.onchange = () => {
            const pid = anSelect.value;
            if (pid) loadProjectAnalysis(pid);
        };

        // Auto-select first matching project if filter is active
        if (filteredProjects.length > 0) {
            anSelect.value = filteredProjects[0].project_id;
            loadProjectAnalysis(filteredProjects[0].project_id);
        } else {
            document.getElementById('analysis-content').classList.add('hidden');
            const placeholder = document.getElementById('analysis-placeholder');
            placeholder.classList.remove('hidden');
            const placeholderText = placeholder.querySelector('p');
            if (placeholderText) {
                placeholderText.innerText = `No projects currently match the "${currentAnalysisFilter}" filter. Click the filter badge above to clear.`;
            }
        }

    } catch (err) {
        console.error("Error populating dropdowns:", err);
    }
}

function renderStage2Results(data) {
    document.getElementById('s2-placeholder').classList.add('hidden');
    const wrapper = document.getElementById('s2-results');
    wrapper.classList.remove('hidden');

    const predDelay = data.prediction.expected_delay_days;
    const predRisk = data.prediction.delay_risk;
    const probs = data.prediction.risk_probabilities || {};

    document.getElementById('s2-res-delay').innerText = `${predDelay.toFixed(1)} Days`;

    const badge = document.getElementById('s2-res-risk-badge');
    badge.className = `risk-badge risk-${predRisk.toLowerCase()}`;
    badge.innerText = predRisk.toUpperCase();

    // Render Probabilities
    document.getElementById('prob-high-bar').style.width = `${(probs['High'] || 0) * 100}%`;
    document.getElementById('prob-high-val').innerText = `${((probs['High'] || 0) * 100).toFixed(1)}%`;

    document.getElementById('prob-med-bar').style.width = `${(probs['Medium'] || 0) * 100}%`;
    document.getElementById('prob-med-val').innerText = `${((probs['Medium'] || 0) * 100).toFixed(1)}%`;

    document.getElementById('prob-low-bar').style.width = `${(probs['Low'] || 0) * 100}%`;
    document.getElementById('prob-low-val').innerText = `${((probs['Low'] || 0) * 100).toFixed(1)}%`;

    // Update SHAP panel explanation description dynamically
    const s2ShapDesc = document.getElementById('s2-shap-description');
    if (s2ShapDesc) {
        s2ShapDesc.innerText = `SHAP explains how each feature contributed to the model's current prediction of ${predRisk} Risk. Positive contributions support the ${predRisk} Risk prediction, while negative contributions move the prediction away from it.`;
    }

    // Render SHAP Bars
    const shapContainer = document.getElementById('s2-shap-bars');
    shapContainer.innerHTML = '';

    const topFactors = data.explanation.risk_top_factors || [];
    const maxMag = topFactors.length > 0 ? topFactors[0].magnitude : 1;

    topFactors.forEach(factor => {
        const pct = Math.min(100, Math.max(12, (factor.magnitude / maxMag) * 100));
        const isPositive = factor.direction === 'supports_class' || factor.shap_value > 0;
        
        let dirLabel = factor.contribution_direction || (isPositive ? `Supports ${predRisk} Risk` : `Moves away from ${predRisk} Risk`);
        
        let badgeClass = 'badge-pos';
        let fillClass = 'shap-positive';
        
        if (isPositive) {
            if (predRisk === 'Low') {
                badgeClass = 'badge-pos';
                fillClass = 'shap-positive';
            } else if (predRisk === 'High') {
                badgeClass = 'badge-neg';
                fillClass = 'shap-negative';
            } else {
                badgeClass = 'badge-amber';
                fillClass = 'shap-amber';
            }
        } else {
            badgeClass = 'badge-slate';
            fillClass = 'shap-neutral';
        }

        const shapValStr = factor.shap_value >= 0 ? `+${factor.shap_value.toFixed(4)}` : `${factor.shap_value.toFixed(4)}`;

        const item = document.createElement('div');
        item.className = 'shap-bar-item';
        item.innerHTML = `
            <div class="shap-meta">
                <div class="shap-feature-info">
                    <strong>${factor.feature}</strong>
                    <span class="shap-dir-badge ${badgeClass}">${dirLabel}</span>
                </div>
                <span class="shap-val-text">SHAP ${shapValStr}</span>
            </div>
            <div class="shap-track">
                <div class="shap-fill ${fillClass}" style="width: ${pct}%"></div>
            </div>
        `;
        shapContainer.appendChild(item);
    });

    // Render Recommendations
    const recsContainer = document.getElementById('s2-recommendations-list');
    recsContainer.innerHTML = '';

    const recs = data.recommendations || [];
    recs.forEach(rec => {
        const prioClass = `prio-${rec.priority.toLowerCase()}`;
        const card = document.createElement('div');
        card.className = `rec-card ${prioClass}`;
        card.innerHTML = `
            <div class="rec-head">
                <span class="rec-factor">${rec.factor}</span>
                <span class="badge badge-${rec.priority === 'HIGH' ? 'rose' : rec.priority === 'MEDIUM' ? 'amber' : 'sky'}">${rec.priority} PRIORITY</span>
            </div>
            <div class="rec-issue">${rec.issue}</div>
            <div class="rec-action"><i class="fa-solid fa-angle-right text-sky-400"></i> ${rec.action}</div>
        `;
        recsContainer.appendChild(card);
    });
}

// --------------------------------------------------------------------------
// 4. PROJECT ANALYSIS CONTROLLER
// --------------------------------------------------------------------------
function updateLifecycleProgress(project, monitoringHistory) {
    const step1 = document.getElementById('flow-step-1');
    const step2 = document.getElementById('flow-step-2');
    const step3 = document.getElementById('flow-step-3');
    const step4 = document.getElementById('flow-step-4');

    if (!step1 || !step2 || !step3 || !step4) return;

    // Reset step classes
    [step1, step2, step3, step4].forEach(step => {
        step.className = 'flow-step';
    });

    if (!project) return;

    // 1. Stage 1 "Project Created": Circle GREEN when project successfully created
    step1.classList.add('completed');

    // 2. Stage 2 "Cost Estimated": Circle GREEN only when valid Stage 1 cost estimation completed
    if (project.predicted_cost !== undefined && project.predicted_cost !== null && project.predicted_cost > 0) {
        step2.classList.add('completed');
    }

    const isCompleted = project.is_completed === 1 || project.is_completed === true;
    const hasHistory = monitoringHistory && monitoringHistory.length > 0;

    // 3. Stage 3 "Ongoing Monitoring": Remains CURRENT/ACTIVE while undergoing monitoring, GREEN when explicitly completed
    if (isCompleted) {
        step3.classList.add('completed');
    } else {
        step3.classList.add('active');
    }

    // 4. Stage 4 "Delay & Risk SHAP": Remains pending/dark while monitoring active. Mark GREEN after monitoring completed AND final delay/risk analysis generated
    if (isCompleted && hasHistory) {
        step4.classList.add('completed');
    } else if (isCompleted) {
        step4.classList.add('active');
    }
}

function renderCompletionActionButton(p) {
    const actionContainer = document.getElementById('analysis-completion-action');
    if (!actionContainer) return;

    const isCompleted = p.is_completed === 1 || p.is_completed === true;
    if (isCompleted) {
        actionContainer.innerHTML = `
            <span class="badge badge-emerald" style="font-size: 11px; padding: 6px 12px; display: inline-flex; align-items: center; gap: 4px;">
                <i class="fa-solid fa-circle-check"></i> Monitoring Completed
            </span>
        `;
    } else {
        actionContainer.innerHTML = `
            <button class="btn btn-emerald text-xs" id="btn-complete-project-monitoring" style="padding: 6px 12px; font-size: 11px; font-weight: 600;" title="Mark ongoing monitoring as completed for this project">
                <i class="fa-solid fa-circle-check ml-0 mr-1"></i> Complete Monitoring
            </button>
        `;
        const btn = document.getElementById('btn-complete-project-monitoring');
        if (btn) {
            btn.onclick = async () => {
                btn.disabled = true;
                btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1"></i> Completing...`;
                try {
                    const cRes = await fetch(`${API_BASE_URL}/api/projects/${p.project_id}/complete`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ is_completed: 1 })
                    });
                    const cData = await cRes.json();
                    if (cData.success) {
                        showToast(`Project ${p.project_id} monitoring completed successfully!`, 'success');
                        if (projectMonitoringInfoMap[p.project_id]) {
                            delete projectMonitoringInfoMap[p.project_id];
                        }
                        loadProjectAnalysis(p.project_id);
                        populateProjectDropdowns();
                    } else {
                        showToast(cData.message || 'Failed to update project completion status', 'error');
                        btn.disabled = false;
                        btn.innerHTML = `<i class="fa-solid fa-circle-check ml-0 mr-1"></i> Complete Monitoring`;
                    }
                } catch (err) {
                    console.error("Error completing monitoring:", err);
                    showToast('Server connection error while updating completion status', 'error');
                    btn.disabled = false;
                    btn.innerHTML = `<i class="fa-solid fa-circle-check ml-0 mr-1"></i> Complete Monitoring`;
                }
            };
        }
    }
}

async function loadProjectAnalysis(projectId) {
    document.getElementById('analysis-placeholder').classList.add('hidden');
    const content = document.getElementById('analysis-content');
    content.classList.remove('hidden');

    try {
        const pRes = await fetch(`${API_BASE_URL}/api/projects/${projectId}`);
        const pData = await pRes.json();

        if (!pData.success) {
            showToast('Unable to fetch project analysis details', 'error');
            return;
        }

        const p = pData.project;
        document.getElementById('an-pid').innerText = p.project_id;
        document.getElementById('an-type').innerText = p.project_type;
        document.getElementById('an-loc').innerText = p.location;
        document.getElementById('an-area').innerText = `${p.area_sqft.toLocaleString()} sq.ft`;
        document.getElementById('an-floors').innerText = p.number_of_floors;
        document.getElementById('an-duration').innerText = `${p.planned_duration} days`;
        document.getElementById('an-cost').innerText = formatINR(p.predicted_cost);

        renderCompletionActionButton(p);

        // Fetch Monitoring History
        const mRes = await fetch(`${API_BASE_URL}/api/projects/${projectId}/monitoring`);
        const mData = await mRes.json();

        const historyTableBody = document.getElementById('analysis-history-table-body');
        const latestContainer = document.getElementById('an-latest-snapshot');

        historyTableBody.innerHTML = '';

        if (!mData.success || mData.count === 0) {
            latestContainer.innerHTML = `<p class="text-slate-400">No monitoring snapshots recorded for this project yet.</p>`;
            historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-slate-400">No monitoring records submitted yet.</td></tr>`;
            renderProgressHistoryChart([], [], []);
            updateLifecycleProgress(p, []);
            return;
        }

        const history = mData.monitoring_history;
        updateLifecycleProgress(p, history);
        const latest = history[history.length - 1];

        latestContainer.innerHTML = `
            <div><span>Sequence Snapshot:</span> <strong>#${latest.sequence_number}</strong></div>
            <div><span>Expected Delay:</span> <strong class="text-amber-400">${latest.expected_delay_days.toFixed(1)} Days</strong></div>
            <div><span>Delay Risk:</span> <span class="risk-badge risk-${latest.delay_risk.toLowerCase()}" style="padding:2px 10px; font-size:11px;">${latest.delay_risk.toUpperCase()}</span></div>
            <div><span>Risk Confidence:</span> <strong>${(latest.risk_probability * 100).toFixed(1)}%</strong></div>
            <div><span>Planned Progress:</span> <strong>${latest.planned_progress_percent}%</strong></div>
            <div><span>Actual Progress:</span> <strong>${latest.actual_progress_percent}%</strong></div>
        `;

        const labels = [];
        const plannedSeries = [];
        const actualSeries = [];

        history.forEach(m => {
            const gap = m.planned_progress_percent - m.actual_progress_percent;
            labels.push(`Snapshot #${m.sequence_number}`);
            plannedSeries.push(m.planned_progress_percent);
            actualSeries.push(m.actual_progress_percent);

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${m.sequence_number}</strong></td>
                <td>${formatDate(m.created_at)}</td>
                <td>${m.planned_progress_percent}%</td>
                <td>${m.actual_progress_percent}%</td>
                <td class="${gap > 0 ? 'text-rose-400 font-semibold' : 'text-emerald-400'}">${gap > 0 ? '+' : ''}${gap.toFixed(1)}%</td>
                <td>${m.actual_labour_count} workers</td>
                <td class="font-semibold">${m.expected_delay_days.toFixed(1)} days</td>
                <td><span class="risk-badge risk-${m.delay_risk.toLowerCase()}" style="padding:2px 8px; font-size:11px;">${m.delay_risk.toUpperCase()}</span></td>
            `;
            historyTableBody.appendChild(tr);
        });

        renderProgressHistoryChart(labels, plannedSeries, actualSeries);

    } catch (err) {
        console.error("Analysis loading error:", err);
        showToast("Error loading project analysis details", "error");
    }
}

function renderProgressHistoryChart(labels, plannedSeries, actualSeries) {
    const ctx = document.getElementById('chart-progress-history').getContext('2d');
    if (progressChartInstance) {
        progressChartInstance.destroy();
    }

    progressChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Planned Progress (%)',
                    data: plannedSeries,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'Actual Progress (%)',
                    data: actualSeries,
                    borderColor: '#dc2626',
                    backgroundColor: 'rgba(220, 38, 38, 0.08)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                }
            ]
        }, 
        
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: '#e2e8f0' },
                    ticks: { color: '#475569' }
                },
                x: {
                    grid: { color: '#e2e8f0' },
                    ticks: { color: '#475569' }
                }
            },
            plugins: {
                legend: { labels: { color: '#0f172a', font: { family: 'Inter', weight: '500' } } }
            }
        }
    });
}

// --------------------------------------------------------------------------
// 5. MODEL PERFORMANCE CONTROLLER
// --------------------------------------------------------------------------
async function loadModelPerformanceData() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/model-performance`);
        const data = await res.json();

        if (!data.success) {
            showToast('Unable to load model performance evaluation summary', 'error');
            return;
        }

        const perf = data.performance;

        // Stage 1 Table
        const s1Body = document.getElementById('perf-s1-table-body');
        s1Body.innerHTML = '';
        (perf.stage1_cost || []).forEach(row => {
            const isBest = row.Model.includes('XGBoost');
            const tr = document.createElement('tr');
            if (isBest) tr.style.backgroundColor = 'rgba(16, 185, 129, 0.08)';
            tr.innerHTML = `
                <td><strong>${row.Model}</strong> ${isBest ? '<span class="badge badge-emerald">Selected Best</span>' : ''}</td>
                <td>₹${row.Test_MAE.toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                <td>₹${row.Test_RMSE.toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                <td class="font-bold text-emerald-400">${row.Test_R2.toFixed(4)}</td>
                <td>${row.CV_R2_Mean.toFixed(4)}</td>
                <td>${row.Tuned ? 'Yes (5-Fold CV)' : 'No (Baseline)'}</td>
            `;
            s1Body.appendChild(tr);
        });

        // Stage 2 Delay Table
        const s2DelayBody = document.getElementById('perf-s2-delay-table-body');
        s2DelayBody.innerHTML = '';
        (perf.stage2_delay || []).forEach(row => {
            const isBest = row.Model.includes('XGBoost');
            const tr = document.createElement('tr');
            if (isBest) tr.style.backgroundColor = 'rgba(245, 158, 11, 0.08)';
            tr.innerHTML = `
                <td><strong>${row.Model}</strong> ${isBest ? '<span class="badge badge-amber">Selected Best</span>' : ''}</td>
                <td>${row.Test_MAE.toFixed(2)} days</td>
                <td>${row.Test_RMSE.toFixed(2)} days</td>
                <td class="font-bold text-amber-400">${row.Test_R2.toFixed(4)}</td>
                <td>${row.CV_R2_Mean.toFixed(4)}</td>
                <td>${row.Tuned ? 'Yes (5-Fold CV)' : 'No'}</td>
            `;
            s2DelayBody.appendChild(tr);
        }); 

        // Stage 2 Risk Table
        const s2RiskBody = document.getElementById('perf-s2-risk-table-body');
        s2RiskBody.innerHTML = '';
        (perf.stage2_risk || []).forEach(row => {
            const isBest = row.Model.includes('Random Forest');
            const tr = document.createElement('tr');
            if (isBest) tr.style.backgroundColor = 'rgba(244, 63, 94, 0.08)';
            tr.innerHTML = `
                <td><strong>${row.Model}</strong> ${isBest ? '<span class="badge badge-rose">Selected Best</span>' : ''}</td>
                <td>${(row.Accuracy * 100).toFixed(2)}%</td>
                <td>${(row.Weighted_Precision * 100).toFixed(2)}%</td>
                <td>${(row.Weighted_Recall * 100).toFixed(2)}%</td>
                <td class="font-bold text-rose-400">${row.Weighted_F1.toFixed(4)}</td>
                <td>${row.Macro_F1.toFixed(4)}</td>
                <td>${row.ROC_AUC_Weighted.toFixed(4)}</td>
            `;
            s2RiskBody.appendChild(tr); 
        }); 

    } catch (err) {
        console.error("Error loading performance data:", err);
        showToast("Error loading model performance scores", "error"); 
    }
}
 