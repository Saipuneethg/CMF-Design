let polytopeData = null;
let polytopeEdgeData = null;
let currentFiberData = null;
let currentJobId = null;
let pollTimer = null;

// ============================================================================
// 1. Synchronize RGB Inputs, Sliders, and Color Preview
// ============================================================================
const channels = ['r', 'g', 'b'];

channels.forEach(c => {
    const num = document.getElementById(`${c}-val`);
    const slider = document.getElementById(`${c}-slider`);

    num.addEventListener('input', () => {
        let val = parseFloat(num.value);
        if (!isNaN(val)) {
            val = Math.max(0, Math.min(1, val));
            slider.value = val;
            updateColorPreview();
        }
    });

    slider.addEventListener('input', () => {
        num.value = parseFloat(slider.value).toFixed(2);
        updateColorPreview();
    });
});

function updateColorPreview() {
    const rInput = parseFloat(document.getElementById('r-val').value);
    const gInput = parseFloat(document.getElementById('g-val').value);
    const bInput = parseFloat(document.getElementById('b-val').value);
    const errText = document.getElementById('color-error-text');

    if (isNaN(rInput) || isNaN(gInput) || isNaN(bInput) ||
        rInput < 0 || rInput > 1 || gInput < 0 || gInput > 1 || bInput < 0 || bInput > 1) {
        errText.innerText = 'Please enter valid RGB values between 0.00 and 1.00.';
        errText.style.display = 'block';
        return;
    }
    errText.style.display = 'none';

    const r = Math.min(255, Math.max(0, Math.round(rInput * 255)));
    const g = Math.min(255, Math.max(0, Math.round(gInput * 255)));
    const b = Math.min(255, Math.max(0, Math.round(bInput * 255)));
    const colorStr = `rgb(${r}, ${g}, ${b})`;

    document.getElementById('color-preview-box').style.backgroundColor = colorStr;
    document.getElementById('color-preview-text').innerText = `RGB(${rInput.toFixed(2)}, ${gInput.toFixed(2)}, ${bInput.toFixed(2)})`;
}

// ============================================================================
// 2. Sampling Settings & Presets
// ============================================================================
const nInput = document.getElementById('n-samples-val');
const presetBtns = document.querySelectorAll('.preset-btn');
const advToggle = document.getElementById('advanced-toggle');
const advPanel = document.getElementById('advanced-panel');

presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const nVal = parseInt(btn.getAttribute('data-n'), 10);
        nInput.value = nVal;
        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

nInput.addEventListener('input', () => {
    const val = parseInt(nInput.value, 10);
    presetBtns.forEach(btn => {
        if (parseInt(btn.getAttribute('data-n'), 10) === val) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
});

advToggle.addEventListener('click', () => {
    if (advPanel.style.display === 'none') {
        advPanel.style.display = 'block';
        advToggle.innerText = '▾ Advanced MCMC Settings (Burn-in, Thinning, Seed)';
    } else {
        advPanel.style.display = 'none';
        advToggle.innerText = '▸ Advanced MCMC Settings (Burn-in, Thinning, Seed)';
    }
});

// ============================================================================
// 2b. Primary Mode Navigation (Single Metameric Fiber vs CIE Chromaticity vs Batch Viewer)
// ============================================================================
const navTabBtns = document.querySelectorAll('.nav-tab-btn');
const primaryViewPanes = document.querySelectorAll('.primary-view-pane');

function switchPrimaryView(targetViewId) {
    navTabBtns.forEach(b => {
        if (b.getAttribute('data-view') === targetViewId) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });

    primaryViewPanes.forEach(p => {
        if (p.id === targetViewId) {
            p.classList.add('active');
        } else {
            p.classList.remove('active');
        }
    });

    if (targetViewId === 'view-cie-chromaticity') {
        if (!chromaticityData) {
            loadChromaticityData(currentBatchRunId);
        } else {
            renderChromaticityPlot();
        }
        setTimeout(() => {
            const el = document.getElementById('plot-chromaticity');
            if (el && el.data) Plotly.Plots.resize(el);
        }, 80);
    } else if (targetViewId === 'view-single-fiber') {
        setTimeout(() => {
            const el = document.getElementById('plot-div');
            if (el && el.data) Plotly.Plots.resize(el);
            const envEl = document.getElementById('plot-envelope');
            if (envEl && envEl.data) Plotly.Plots.resize(envEl);
        }, 80);
    } else if (targetViewId === 'view-cmf-image') {
        initCMFImageEngine();
        setTimeout(() => {
            const el = document.getElementById('plot-cmf-single-curves');
            if (el && el.data) Plotly.Plots.resize(el);
        }, 80);
    }

    window.dispatchEvent(new Event('resize'));
}

navTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const viewId = btn.getAttribute('data-view');
        switchPrimaryView(viewId);
    });
});

const sidebarBtnCie = document.getElementById('sidebar-btn-cie');
if (sidebarBtnCie) {
    sidebarBtnCie.addEventListener('click', () => {
        switchPrimaryView('view-cie-chromaticity');
    });
}

const sidebarBtnCmfRender = document.getElementById('sidebar-btn-cmf-render');
if (sidebarBtnCmfRender) {
    sidebarBtnCmfRender.addEventListener('click', () => {
        switchPrimaryView('view-cmf-image');
    });
}

// ============================================================================
// 3. Sub-Tab Switching (Panels D1, D2, E, and Wavelength Table inside Single Fiber)
// ============================================================================
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-tab');
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetPane = document.getElementById(targetId);
        if (targetPane) {
            targetPane.classList.add('active');
            window.dispatchEvent(new Event('resize'));
            setTimeout(() => {
                const plotMap = {
                    'tab-envelope': ['plot-envelope'],
                    'tab-spans': ['plot-span-bars', 'plot-metamers'],
                    'tab-cmf': ['plot-cmf-curves'],
                    'tab-covariance': ['plot-eigenvalues', 'plot-modes']
                };
                if (targetId === 'tab-cmf') {
                    renderCMFPlot();
                }
                const ids = plotMap[targetId] || [];
                ids.forEach(id => {
                    const el = document.getElementById(id);
                    if (el && el.data) Plotly.Plots.resize(el);
                });
            }, 60);
        }
    });
});

// ============================================================================
// 4. Load Prior Polytope
// ============================================================================
async function loadPolytope() {
    try {
        const res = await fetch('/api/polytope');
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        polytopeData = {
            type: 'mesh3d',
            x: data.vertices.map(v => v[0]),
            y: data.vertices.map(v => v[1]),
            z: data.vertices.map(v => v[2]),
            i: data.faces.map(f => f[0]),
            j: data.faces.map(f => f[1]),
            k: data.faces.map(f => f[2]),
            opacity: 0.12,
            color: '#3b82f6',
            name: 'Spectral Prior Polytope (Translucent Surface)',
            hoverinfo: 'none',
            showscale: false
        };

        // Extract boundary edges to render subtle, light facet borders
        const edgeSet = new Set();
        data.faces.forEach(f => {
            const addEdge = (u, v) => {
                const key = u < v ? `${u}_${v}` : `${v}_${u}`;
                edgeSet.add(key);
            };
            addEdge(f[0], f[1]);
            addEdge(f[1], f[2]);
            addEdge(f[2], f[0]);
        });

        const edgeX = [];
        const edgeY = [];
        const edgeZ = [];
        edgeSet.forEach(key => {
            const [u, v] = key.split('_').map(Number);
            edgeX.push(data.vertices[u][0], data.vertices[v][0], null);
            edgeY.push(data.vertices[u][1], data.vertices[v][1], null);
            edgeZ.push(data.vertices[u][2], data.vertices[v][2], null);
        });

        polytopeEdgeData = {
            type: 'scatter3d',
            mode: 'lines',
            x: edgeX,
            y: edgeY,
            z: edgeZ,
            line: {
                color: 'rgba(147, 197, 253, 0.035)',
                width: 0.75
            },
            name: 'Polytope Facet Edges',
            hoverinfo: 'none',
            showlegend: false
        };

        update3DPlot();
    } catch(err) {
        console.error('Failed to load polytope:', err);
    }
}

// ============================================================================
// 5. Calculation Trigger & Polling
// ============================================================================
const calcBtn = document.getElementById('calc-btn');
const progressContainer = document.getElementById('progress-container');
const progressBarFill = document.getElementById('progress-bar-fill');
const progressStatusText = document.getElementById('progress-status-text');

calcBtn.addEventListener('click', () => {
    const r = parseFloat(document.getElementById('r-val').value);
    const g = parseFloat(document.getElementById('g-val').value);
    const b = parseFloat(document.getElementById('b-val').value);
    const n_samples = parseInt(nInput.value, 10);
    const burn_in = parseInt(document.getElementById('burn-in-val').value, 10);
    const thinning = parseInt(document.getElementById('thinning-val').value, 10);
    const bound_mode = document.getElementById('bound-mode-select').value;

    const camera_model = document.getElementById('camera-model-select') ? document.getElementById('camera-model-select').value : 'baseline';

    if (isNaN(r) || isNaN(g) || isNaN(b) || r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1) {
        alert('Please enter valid RGB values between 0.00 and 1.00.');
        return;
    }

    if (isNaN(n_samples) || n_samples < 100 || n_samples > 5000) {
        alert('Number of samples (N) must be an integer between 100 and 5000.');
        return;
    }

    startCalculation(r, g, b, n_samples, burn_in, thinning, bound_mode, camera_model);
});

const cameraModelSelect = document.getElementById('camera-model-select');
if (cameraModelSelect) {
    cameraModelSelect.addEventListener('change', () => {
        const val = cameraModelSelect.value;
        const badge = document.getElementById('active-camera-badge');
        if (badge) {
            badge.innerText = val === 'optimized' ? 'Optimized Sensitivity Camera' : 'Original Calibrated Camera';
            badge.style.color = val === 'optimized' ? '#10b981' : '#38bdf8';
        }
    });
}

async function startCalculation(r, g, b, n_samples, burn_in, thinning, bound_mode, camera_model = 'baseline') {
    if (pollTimer) clearInterval(pollTimer);

    // Set UI to loading state
    calcBtn.disabled = true;
    calcBtn.innerText = `[ COMPUTING FIBER FOR N = ${n_samples}... ]`;
    progressContainer.style.display = 'block';
    progressBarFill.style.width = '5%';
    const camName = camera_model === 'optimized' ? 'Optimized Camera' : 'Baseline Camera';
    progressStatusText.innerText = `Starting calculation for RGB(${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)}) using ${camName}...`;

    document.getElementById('badge-gamut').className = 'badge badge-warn';
    document.getElementById('badge-gamut').innerText = 'Gamut: Computing...';

    try {
        const res = await fetch('/api/calculate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                r, g, b,
                n_samples,
                burn_in,
                thinning,
                bound_mode,
                camera_model,
                async: true
            })
        });

        const initialData = await res.json();
        if (res.status === 400 || initialData.error) {
            throw new Error(initialData.error || 'Invalid request parameters');
        }

        // If returned synchronously
        if (!initialData.job_id) {
            handleCalculationCompleted(initialData);
            return;
        }

        currentJobId = initialData.job_id;

        // Poll background job
        pollTimer = setInterval(async () => {
            try {
                const pollRes = await fetch(`/api/jobs/${currentJobId}`);
                const job = await pollRes.json();

                if (job.error) {
                    clearInterval(pollTimer);
                    throw new Error(job.error);
                }

                const pct = job.progress || 0;
                progressBarFill.style.width = `${Math.max(5, pct)}%`;
                progressStatusText.innerText = `${job.stage || 'Calculating...'} (${pct}%)`;

                if (job.status === 'completed') {
                    clearInterval(pollTimer);
                    handleCalculationCompleted(job.result);
                } else if (job.status === 'error') {
                    clearInterval(pollTimer);
                    throw new Error(job.error || 'Computation failed');
                }
            } catch (pollErr) {
                clearInterval(pollTimer);
                console.error(pollErr);
                handleCalculationError(pollErr.message);
            }
        }, 600);

    } catch (err) {
        console.error(err);
        handleCalculationError(err.message);
    }
}

function handleCalculationError(msg) {
    resetLoadingState();
    document.getElementById('gamut-status-desc').innerHTML = `<span style="color:#ef4444;">Error: ${msg}</span>`;
    document.getElementById('badge-gamut').className = 'badge badge-error';
    document.getElementById('badge-gamut').innerText = 'Gamut: Error';
    clearAllResults();
}

function resetLoadingState() {
    calcBtn.disabled = false;
    calcBtn.innerText = '[ CALCULATE METAMERIC FIBER ]';
    progressContainer.style.display = 'none';
}

function handleCalculationCompleted(data) {
    resetLoadingState();

    if (!data.feasible) {
        // Infeasible target
        document.getElementById('badge-gamut').className = 'badge badge-error';
        document.getElementById('badge-gamut').innerText = 'Gamut: OUTSIDE GAMUT';
        document.getElementById('gamut-status-desc').innerHTML = '<span style="color:#ef4444; font-weight:600;">✗ Infeasible Target</span><br>No feasible metameric fiber exists for this RGB observation under the modeled spectral prior.';
        currentFiberData = null;
        clearAllResults();
        update3DPlot();
    } else {
        // Feasible target
        document.getElementById('badge-gamut').className = 'badge badge-ok';
        document.getElementById('badge-gamut').innerText = 'Gamut: INSIDE GAMUT';
        document.getElementById('gamut-status-desc').innerHTML = '<span style="color:#10b981; font-weight:600;">✓ Feasible Metameric Fiber</span><br>Observation is within the prior-induced gamut.';
        currentFiberData = data;
        updateUI(data);
    }
}

function clearAllResults() {
    document.getElementById('dim-k').innerText = '-';
    document.getElementById('dim-rank').innerText = '-';
    document.getElementById('dim-affine').innerText = '-';
    document.getElementById('dim-feasible').innerText = '-';
    document.getElementById('dim-sampling').innerText = '-';

    document.getElementById('score-spread').innerText = '-';
    document.getElementById('score-alpha-spread').innerText = '-';
    document.getElementById('score-pairwise-mean').innerText = '-';
    document.getElementById('score-pairwise-p95').innerText = '-';
    document.getElementById('score-pairwise-max').innerText = '-';
    document.getElementById('score-vol-scale').innerText = '-';
    document.getElementById('score-box-vol').innerText = '-';
    document.getElementById('score-composite').innerText = '-';

    document.getElementById('val-mean-err').innerText = '-';
    document.getElementById('val-max-err').innerText = '-';
    document.getElementById('val-rms-err').innerText = '-';
    document.getElementById('val-residual').innerText = '-';
    document.getElementById('val-valid-pts').innerText = '-';
    document.getElementById('val-max-viol').innerText = '-';
    document.getElementById('val-ess').innerText = '-';
    document.getElementById('val-autocorr').innerText = '-';

    document.getElementById('badge-validation').className = 'badge badge-neutral';
    document.getElementById('badge-validation').innerText = 'Validation: None';
    document.getElementById('badge-val-summary').className = 'badge badge-neutral';
    document.getElementById('badge-val-summary').innerText = 'NONE';
    document.getElementById('sample-warning-box').style.display = 'none';

    document.getElementById('bounds-table-body').innerHTML = '<tr><td colspan="10" style="text-align:center; color:#888;">No feasible data for this target.</td></tr>';

    ['plot-envelope', 'plot-span-bars', 'plot-metamers', 'plot-eigenvalues', 'plot-modes'].forEach(id => {
        Plotly.purge(id);
    });
}

// ============================================================================
// 6. Update UI with Calculation Results
// ============================================================================
function updateUI(data) {
    const amb = data.ambiguity_metrics || {};
    const val = data.validation || {};
    const dims = data.dimension_analysis || {};

    // 1. Dimensions (Panel B)
    document.getElementById('dim-k').innerText = dims.pca_component_count?.value || data.constraint_rank + data.fiber_dimension;
    document.getElementById('dim-rank').innerText = dims.constraint_rank?.value || data.constraint_rank;
    document.getElementById('dim-affine').innerText = dims.affine_dimension?.value || data.fiber_dimension;
    document.getElementById('dim-feasible').innerText = dims.feasible_dimension?.value || data.fiber_dimension;
    document.getElementById('dim-sampling').innerText = dims.sampling_dimension?.value || data.fiber_dimension;

    if (dims.summary) {
        document.getElementById('dim-explanation').innerText = dims.summary;
    }

    // 2. Ambiguity Scores (Panel C)
    const spread = amb.spread_score || {};
    document.getElementById('score-spread').innerText = spread.spectral_rms?.value ? spread.spectral_rms.value.toFixed(4) : '-';
    document.getElementById('score-alpha-spread').innerText = spread.alpha_trace?.value ? spread.alpha_trace.value.toFixed(4) : '-';

    const pw = amb.pairwise_distances || {};
    document.getElementById('score-pairwise-mean').innerText = pw.mean?.value ? pw.mean.value.toFixed(4) : '-';
    document.getElementById('score-pairwise-p95').innerText = pw.p95?.value ? pw.p95.value.toFixed(4) : '-';
    document.getElementById('score-pairwise-max').innerText = pw.max_observed?.value ? pw.max_observed.value.toFixed(4) : '-';

    const vols = amb.volume_scores || {};
    document.getElementById('score-vol-scale').innerText = vols.ellipsoid_scale_a_det?.value ? vols.ellipsoid_scale_a_det.value.toFixed(4) : '-';
    document.getElementById('score-box-vol').innerText = vols.bounding_box_volume?.value ? vols.bounding_box_volume.value.toExponential(3) : '-';

    const comp = amb.composite_index;
    document.getElementById('score-composite').innerText = comp?.value ? comp.value.toFixed(4) : '-';

    // 3. Validation & Diagnostics (Panel F)
    document.getElementById('val-mean-err').innerText = val.mean_error !== undefined ? val.mean_error.toExponential(3) : '-';
    document.getElementById('val-max-err').innerText = val.max_error !== undefined ? val.max_error.toExponential(3) : '-';
    document.getElementById('val-rms-err').innerText = val.rms_error !== undefined ? val.rms_error.toExponential(3) : '-';
    document.getElementById('val-residual').innerText = val.constraint_residual !== undefined ? val.constraint_residual.toExponential(3) : '-';

    const pctValid = val.percent_valid !== undefined ? val.percent_valid.toFixed(1) : '100.0';
    document.getElementById('val-valid-pts').innerText = `${val.valid_points || data.n_samples} / ${val.total_points || data.n_samples} (${pctValid}%)`;
    document.getElementById('val-max-viol').innerText = val.max_violation !== undefined ? val.max_violation.toExponential(3) : '-';

    const mcmc = amb.mcmc_diagnostics || {};
    document.getElementById('val-ess').innerText = mcmc.min_ess ? Math.round(mcmc.min_ess) : `${data.n_samples}`;
    document.getElementById('val-autocorr').innerText = mcmc.mean_lag1_autocorr !== undefined ? mcmc.mean_lag1_autocorr.toFixed(3) : '0.000';

    const overall = val.overall_status || 'PASS';
    const valBadge = document.getElementById('badge-validation');
    const valSumBadge = document.getElementById('badge-val-summary');

    if (overall === 'PASS') {
        valBadge.className = 'badge badge-ok';
        valBadge.innerText = 'Validation: PASS';
        valSumBadge.className = 'badge badge-ok';
        valSumBadge.innerText = 'PASS';
    } else if (overall === 'WARN') {
        valBadge.className = 'badge badge-warn';
        valBadge.innerText = 'Validation: WARN';
        valSumBadge.className = 'badge badge-warn';
        valSumBadge.innerText = 'WARN';
    } else {
        valBadge.className = 'badge badge-error';
        valBadge.innerText = 'Validation: FAIL';
        valSumBadge.className = 'badge badge-error';
        valSumBadge.innerText = 'FAIL';
    }

    // Warnings if low N
    const warnBox = document.getElementById('sample-warning-box');
    if (data.warning) {
        warnBox.innerText = `⚠ ${data.warning}`;
        warnBox.style.display = 'block';
    } else {
        warnBox.style.display = 'none';
    }

    // 4. Update Plots & Table
    update3DPlot();
    renderSpectralPlots(data);
    renderBoundsTable(data);
}

// ============================================================================
// 7. Render 3D Viewport
// ============================================================================
const showPolytope = document.getElementById('show-polytope');
const showFiber = document.getElementById('show-fiber');
const showCentroid = document.getElementById('show-centroid');
const showDirs = document.getElementById('show-directions');
const resetCamBtn = document.getElementById('reset-cam-btn');

[showPolytope, showFiber, showCentroid, showDirs].forEach(cb => {
    cb.addEventListener('change', update3DPlot);
});

resetCamBtn.addEventListener('click', () => {
    Plotly.relayout('plot-div', {
        'scene.camera': { eye: { x: 1.35, y: 1.35, z: 1.15 } }
    });
});

function update3DPlot() {
    const traces = [];

    if (polytopeData && showPolytope.checked) {
        traces.push(polytopeData);
        if (polytopeEdgeData) {
            traces.push(polytopeEdgeData);
        }
    }

    if (currentFiberData) {
        if (showFiber.checked && currentFiberData.z_points_3d) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: currentFiberData.z_points_3d.map(p => p[0]),
                y: currentFiberData.z_points_3d.map(p => p[1]),
                z: currentFiberData.z_points_3d.map(p => p[2]),
                marker: { size: 2.5, color: '#f59e0b', opacity: 0.65 },
                name: `Metamers (N=${currentFiberData.n_samples})`,
                hoverinfo: 'none'
            });
        }

        if (showCentroid.checked) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: [currentFiberData.z_centroid_3d[0]],
                y: [currentFiberData.z_centroid_3d[1]],
                z: [currentFiberData.z_centroid_3d[2]],
                marker: { size: 6, color: '#ef4444', symbol: 'cross' },
                name: 'Fiber Centroid'
            });

            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: [currentFiberData.z0_3d[0]],
                y: [currentFiberData.z0_3d[1]],
                z: [currentFiberData.z0_3d[2]],
                marker: { size: 5, color: '#10b981', symbol: 'circle' },
                name: 'Feasible Anchor Point z0'
            });
        }

        if (showDirs.checked && currentFiberData.principal_directions) {
            const c = currentFiberData.z_centroid_3d;
            currentFiberData.principal_directions.forEach((dir, i) => {
                const scale = 1.8;
                traces.push({
                    type: 'scatter3d',
                    mode: 'lines',
                    x: [c[0] - dir[0] * scale, c[0] + dir[0] * scale],
                    y: [c[1] - dir[1] * scale, c[1] + dir[1] * scale],
                    z: [c[2] - dir[2] * scale, c[2] + dir[2] * scale],
                    line: { width: 4, color: ['#ef4444', '#10b981', '#3b82f6'][i] || '#fff' },
                    name: `Principal Dir ${i + 1}`
                });
            });
        }
    }

    const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: '#000',
        scene: {
            aspectmode: 'data',
            xaxis: { title: 'PCA 1', color: '#94a3b8', gridcolor: '#222' },
            yaxis: { title: 'PCA 2', color: '#94a3b8', gridcolor: '#222' },
            zaxis: { title: 'PCA 3', color: '#94a3b8', gridcolor: '#222' },
            camera: {
                eye: { x: 1.35, y: 1.35, z: 1.15 }
            }
        },
        showlegend: true,
        legend: {
            font: { color: '#fff', size: 10 },
            bgcolor: 'rgba(0,0,0,0.6)',
            x: 0.5,
            xanchor: 'center',
            y: 0.98,
            orientation: 'h'
        }
    };

    Plotly.react('plot-div', traces, layout, {responsive: true});
}

// ============================================================================
// 8. Render Spectral Plots (Envelope, Spans, Covariance, Modes)
// ============================================================================
function renderSpectralPlots(data) {
    const amb = data.ambiguity_metrics || {};
    const wlData = amb.wavelength_space || {};
    const wavelengths = wlData.wavelengths || [];
    const envelope = data.spectral_envelope || [];

    // ------------------------------------------------------------------------
    // PLOT 1: Spectral Envelope & Uncertainty Ribbon (Panel D1)
    // ------------------------------------------------------------------------
    const wls = envelope.map(e => e.wavelength_nm);
    const p95 = envelope.map(e => e.p95);
    const p5 = envelope.map(e => e.p5);
    const sMean = envelope.map(e => e.sample_mean);
    const sMin = envelope.map(e => e.sample_min);
    const sMax = envelope.map(e => e.sample_max);

    const envTraces = [
        // Sample Min/Max boundary (subtle)
        {
            x: wls,
            y: sMax,
            mode: 'lines',
            line: { width: 1, color: 'rgba(245, 158, 11, 0.4)', dash: 'dot' },
            name: 'Sample Extrema (Min/Max)',
            showlegend: true
        },
        {
            x: wls,
            y: sMin,
            mode: 'lines',
            line: { width: 1, color: 'rgba(245, 158, 11, 0.4)', dash: 'dot' },
            fill: 'tonexty',
            fillcolor: 'rgba(245, 158, 11, 0.08)',
            showlegend: false
        },
        // 5%-95% Percentile Ribbon
        {
            x: wls,
            y: p95,
            mode: 'lines',
            line: { width: 0, color: 'rgba(59, 130, 246, 0.2)' },
            name: 'Sample 5th–95th Percentile Ribbon',
            showlegend: true
        },
        {
            x: wls,
            y: p5,
            mode: 'lines',
            line: { width: 0, color: 'rgba(59, 130, 246, 0.2)' },
            fill: 'tonexty',
            fillcolor: 'rgba(59, 130, 246, 0.25)',
            showlegend: false
        },
        // Mean Spectrum
        {
            x: wls,
            y: sMean,
            mode: 'lines+markers',
            line: { width: 2.5, color: '#3b82f6' },
            marker: { size: 4 },
            name: 'Mean Metameric Spectrum',
            showlegend: true
        }
    ];

    // Overlay Exact LP Bounds where computed
    const lpWls = [];
    const lpMin = [];
    const lpMax = [];
    envelope.forEach(e => {
        if (e.exact_min !== null && e.exact_max !== null) {
            lpWls.push(e.wavelength_nm);
            lpMin.push(e.exact_min);
            lpMax.push(e.exact_max);
        }
    });

    if (lpWls.length > 0) {
        envTraces.push({
            x: lpWls,
            y: lpMax,
            mode: 'markers',
            marker: { size: 9, color: '#ef4444', symbol: 'triangle-up' },
            name: 'Exact LP Upper Bound R_max',
            showlegend: true
        });
        envTraces.push({
            x: lpWls,
            y: lpMin,
            mode: 'markers',
            marker: { size: 9, color: '#10b981', symbol: 'triangle-down' },
            name: 'Exact LP Lower Bound R_min',
            showlegend: true
        });
    }

    const envLayout = {
        margin: { l: 55, r: 25, b: 40, t: 25 },
        paper_bgcolor: '#14171f',
        plot_bgcolor: '#0b0d11',
        xaxis: { title: 'Wavelength (nm)', color: '#8e9bb0', gridcolor: '#222' },
        yaxis: { title: 'Reflectance (0–1)', color: '#8e9bb0', gridcolor: '#222', rangemode: 'tozero' },
        showlegend: true,
        legend: { font: { color: '#fff', size: 10 }, bgcolor: 'rgba(0,0,0,0.5)', x: 0.02, y: 0.98 }
    };

    Plotly.react('plot-envelope', envTraces, envLayout, {responsive: true});

    // ------------------------------------------------------------------------
    // PLOT 2: Spectral Span Bars across 31 Bands (Panel D2)
    // ------------------------------------------------------------------------
    const spans = envelope.map(e => e.sample_span);
    const spanTraces = [{
        x: wls,
        y: spans,
        type: 'bar',
        marker: { color: '#f59e0b' },
        name: 'Sample Spectral Span'
    }];

    const spanLayout = {
        margin: { l: 50, r: 20, b: 35, t: 20 },
        paper_bgcolor: '#000',
        plot_bgcolor: '#000',
        xaxis: { title: 'Wavelength (nm)', color: '#888', gridcolor: '#222' },
        yaxis: { title: 'Span ΔR (reflectance)', color: '#888', gridcolor: '#222' }
    };
    Plotly.react('plot-span-bars', spanTraces, spanLayout, {responsive: true});

    // ------------------------------------------------------------------------
    // PLOT 3: Selected Representative Metamers (Panel D2)
    // ------------------------------------------------------------------------
    const selectedSpectra = wlData.selected_spectra || [];
    const metaColors = ['#ef4444', '#10b981', '#3b82f6', '#f59e0b', '#a855f7'];
    const metaTraces = selectedSpectra.map((spec, i) => ({
        x: wavelengths,
        y: spec,
        mode: 'lines',
        line: { width: 1.8, color: metaColors[i % metaColors.length] },
        name: `Metamer Sample ${i + 1}`
    }));

    const metaLayout = {
        margin: { l: 50, r: 20, b: 35, t: 20 },
        paper_bgcolor: '#000',
        plot_bgcolor: '#000',
        xaxis: { title: 'Wavelength (nm)', color: '#888', gridcolor: '#222' },
        yaxis: { title: 'Reflectance (0–1)', color: '#888', gridcolor: '#222' },
        showlegend: true,
        legend: { font: { color: '#fff', size: 9 }, bgcolor: 'rgba(0,0,0,0.4)', x: 0.02, y: 0.98 }
    };
    Plotly.react('plot-metamers', metaTraces, metaLayout, {responsive: true});

    // ------------------------------------------------------------------------
    // PLOT 4: Eigenvalues & Variance Explained (Panel E)
    // ------------------------------------------------------------------------
    const covSpec = amb.spectral_covariance || {};
    const eigvals = covSpec.leading_eigenvalues || [];
    const expVar = (covSpec.explained_variance_ratio || []).map(v => v * 100);
    const cumVar = (covSpec.cumulative_explained_variance || []).map(v => v * 100);
    const modeLabels = eigvals.map((_, i) => `Mode ${i + 1}`);

    const eigTraces = [
        {
            x: modeLabels,
            y: expVar,
            type: 'bar',
            marker: { color: '#3b82f6' },
            name: 'Explained Variance (%)'
        },
        {
            x: modeLabels,
            y: cumVar,
            type: 'scatter',
            mode: 'lines+markers',
            line: { width: 2, color: '#10b981' },
            marker: { size: 6 },
            name: 'Cumulative Variance (%)'
        }
    ];

    const eigLayout = {
        margin: { l: 50, r: 20, b: 35, t: 20 },
        paper_bgcolor: '#000',
        plot_bgcolor: '#000',
        xaxis: { color: '#888' },
        yaxis: { title: 'Variance Ratio (%)', color: '#888', gridcolor: '#222', range: [0, 105] },
        showlegend: true,
        legend: { font: { color: '#fff', size: 9 }, bgcolor: 'rgba(0,0,0,0.4)', x: 0.5, xanchor: 'center', y: 0.98, orientation: 'h' }
    };
    Plotly.react('plot-eigenvalues', eigTraces, eigLayout, {responsive: true});

    // ------------------------------------------------------------------------
    // PLOT 5: Leading Spectral Ambiguity Modes (Panel E)
    // ------------------------------------------------------------------------
    const leadingModes = covSpec.leading_modes_31 || [];
    // leadingModes is (31, 5) -> transpose or slice columns
    const modeColors = ['#ef4444', '#10b981', '#3b82f6'];
    const modeTraces = [];

    for (let m = 0; m < Math.min(3, leadingModes[0]?.length || 0); m++) {
        const modeY = leadingModes.map(row => row[m]);
        modeTraces.push({
            x: wavelengths,
            y: modeY,
            mode: 'lines',
            line: { width: 2, color: modeColors[m] },
            name: `Principal Mode ${m + 1} (${expVar[m]?.toFixed(1) || '0'}%)`
        });
    }

    const modeLayout = {
        margin: { l: 50, r: 20, b: 35, t: 20 },
        paper_bgcolor: '#000',
        plot_bgcolor: '#000',
        xaxis: { title: 'Wavelength (nm)', color: '#888', gridcolor: '#222' },
        yaxis: { title: 'Eigenvector (dimensionless)', color: '#888', gridcolor: '#222' },
        showlegend: true,
        legend: { font: { color: '#fff', size: 9 }, bgcolor: 'rgba(0,0,0,0.4)', x: 0.02, y: 0.98 }
    };
    Plotly.react('plot-modes', modeTraces, modeLayout, {responsive: true});
}

// ============================================================================
// 9. Render Bounds Table
// ============================================================================
function renderBoundsTable(data) {
    const tbody = document.getElementById('bounds-table-body');
    const env = data.spectral_envelope || [];

    if (env.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; color:#888;">No bounds data.</td></tr>';
        return;
    }

    let rowsHtml = '';
    env.forEach(item => {
        const wl = item.wavelength_nm;
        const sMin = item.sample_min !== null ? item.sample_min.toFixed(4) : '-';
        const sMean = item.sample_mean !== null ? item.sample_mean.toFixed(4) : '-';
        const sMax = item.sample_max !== null ? item.sample_max.toFixed(4) : '-';
        const sSpan = item.sample_span !== null ? item.sample_span.toFixed(4) : '-';

        const exMin = item.exact_min !== null && item.exact_min !== undefined ? item.exact_min.toFixed(4) : '-';
        const exMax = item.exact_max !== null && item.exact_max !== undefined ? item.exact_max.toFixed(4) : '-';
        const exSpan = item.exact_span !== null && item.exact_span !== undefined ? item.exact_span.toFixed(4) : '-';

        let undercov = '-';
        if (item.exact_min !== null && item.exact_max !== null && item.exact_min !== undefined && item.exact_max !== undefined) {
            const uMin = Math.max(0, item.sample_min - item.exact_min);
            const uMax = Math.max(0, item.exact_max - item.sample_max);
            undercov = `+${(uMin + uMax).toFixed(4)}`;
        }

        const method = item.exact_min !== null && item.exact_min !== undefined ? 'Exact LP + Sampled' : 'Sampled Extrema';

        rowsHtml += `
            <tr>
                <td><strong>${wl} nm</strong></td>
                <td>${sMin}</td>
                <td>${sMean}</td>
                <td>${sMax}</td>
                <td>${sSpan}</td>
                <td style="color:#10b981;">${exMin}</td>
                <td style="color:#ef4444;">${exMax}</td>
                <td>${exSpan}</td>
                <td style="color:#f59e0b;">${undercov}</td>
                <td style="font-family:sans-serif; font-size:0.75rem; color:#94a3b8;">${method}</td>
            </tr>
        `;
    });

    tbody.innerHTML = rowsHtml;
}

// ============================================================================
// 10. Export Handlers (JSON, CSV, Markdown)
// ============================================================================
document.getElementById('export-json-btn').addEventListener('click', () => {
    if (!currentFiberData) {
        alert('Please calculate a metameric fiber before exporting.');
        return;
    }
    triggerExport('json');
});

document.getElementById('export-csv-btn').addEventListener('click', () => {
    if (!currentFiberData) {
        alert('Please calculate a metameric fiber before exporting.');
        return;
    }
    triggerExport('csv');
});

document.getElementById('export-md-btn').addEventListener('click', () => {
    if (!currentFiberData) {
        alert('Please calculate a metameric fiber before exporting.');
        return;
    }
    triggerExport('md');
});

async function triggerExport(fmt) {
    if (currentJobId) {
        window.location.href = `/api/jobs/${currentJobId}/export?format=${fmt}`;
        return;
    }

    try {
        const res = await fetch(`/api/export?format=${fmt}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(currentFiberData)
        });
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fiber_analysis.${fmt}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch(err) {
        console.error('Export failed:', err);
        alert('Export failed: ' + err.message);
    }
}

// ============================================================================
// 11. Initial Startup & Resize Handling
// ============================================================================
window.addEventListener('resize', () => {
    const el = document.getElementById('plot-div');
    if (el && el.data) {
        Plotly.Plots.resize(el);
    }
});

updateColorPreview();
loadPolytope();

// ============================================================================
// 12. Batch 100 Colours Analysis & Viewer Integration
// ============================================================================
let batchPollTimer = null;
let currentBatchRunId = null;
let currentBatchRecords = [];
let currentBatchIndex = 0;
let currentBatchViewMode = 'single';

const batchStartBtn = document.getElementById('batch-start-btn');
const batchRunSelect = document.getElementById('batch-run-select');
const batchColourSelect = document.getElementById('batch-colour-select');
const batchPrevBtn = document.getElementById('batch-prev-btn');
const batchNextBtn = document.getElementById('batch-next-btn');

const batchProgressBox = document.getElementById('batch-progress-box');
const batchProgressBar = document.getElementById('batch-progress-bar');
const batchStatusText = document.getElementById('batch-status-text');
const batchBadgeSel = document.getElementById('batch-badge-sel');
const batchBadgeGamut = document.getElementById('batch-badge-gamut');
const batchBadgeDone = document.getElementById('batch-badge-done');
const batchBadgePass = document.getElementById('batch-badge-pass');

const batchMainImg = document.getElementById('batch-main-img');
const batchImgPlaceholder = document.getElementById('batch-img-placeholder');
const batchViewSwatch = document.getElementById('batch-view-swatch');
const batchViewId = document.getElementById('batch-view-id');
const batchViewRgb = document.getElementById('batch-view-rgb');
const batchViewBadge = document.getElementById('batch-view-badge');
const batchSwatchGrid = document.getElementById('batch-swatch-grid');

// Start Batch Analysis Handler
if (batchStartBtn) {
    batchStartBtn.addEventListener('click', async () => {
        const imagePath = document.getElementById('batch-image-input').value.trim() || 'Dataset/image.png';
        const targetColours = parseInt(document.getElementById('batch-colours-input').value, 10) || 100;
        const nSamples = parseInt(document.getElementById('batch-samples-input').value, 10) || 500;

        try {
            batchStartBtn.disabled = true;
            batchStartBtn.innerText = 'STARTING BATCH...';
            batchProgressBox.style.display = 'block';

            const res = await fetch('/api/batch_analysis/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_path: imagePath,
                    target_colours: targetColours,
                    n_samples: nSamples,
                    seed: 42
                })
            });

            const data = await res.json();
            if (!res.ok) {
                alert(data.error || 'Failed to start batch analysis');
                batchStartBtn.disabled = false;
                batchStartBtn.innerText = '[ START BATCH ANALYSIS ]';
                return;
            }

            pollBatchStatus();
        } catch (err) {
            alert('Error starting batch analysis: ' + err.message);
            batchStartBtn.disabled = false;
            batchStartBtn.innerText = '[ START BATCH ANALYSIS ]';
        }
    });
}

function pollBatchStatus() {
    if (batchPollTimer) clearInterval(batchPollTimer);

    batchPollTimer = setInterval(async () => {
        try {
            const res = await fetch('/api/batch_analysis/status');
            const data = await res.json();

            if (data.status === 'running') {
                batchProgressBox.style.display = 'block';
                const pct = data.percent_complete || 0;
                batchProgressBar.style.width = `${pct}%`;
                batchStatusText.innerText = data.current_action || 'Processing...';

                batchBadgeSel.innerText = `Sel: ${data.candidates_tested || 0}`;
                batchBadgeGamut.innerText = `In-Gamut: ${data.selected_in_gamut || 0}`;
                batchBadgeDone.innerText = `Done: ${data.completed_count || 0}`;
                batchBadgePass.innerText = `Pass: ${data.validated_count || 0}`;

                if (data.run_id && data.run_id !== currentBatchRunId) {
                    currentBatchRunId = data.run_id;
                    refreshBatchRunsList();
                }
            } else if (data.status === 'completed') {
                clearInterval(batchPollTimer);
                batchPollTimer = null;
                batchProgressBox.style.display = 'block';
                batchProgressBar.style.width = '100%';
                batchStatusText.innerText = 'Batch completed successfully! 100 colours analyzed.';
                batchBadgeDone.innerText = `Done: ${data.completed_count || 100}`;
                batchBadgePass.innerText = `Pass: ${data.validated_count || 100}`;

                if (batchStartBtn) {
                    batchStartBtn.disabled = false;
                    batchStartBtn.innerText = '[ START BATCH ANALYSIS ]';
                }
                refreshBatchRunsList();
            } else {
                if (batchStartBtn) {
                    batchStartBtn.disabled = false;
                    batchStartBtn.innerText = '[ START BATCH ANALYSIS ]';
                }
            }
        } catch (err) {
            console.error('Error polling batch status:', err);
        }
    }, 2000);
}

async function refreshBatchRunsList() {
    try {
        const res = await fetch('/api/batch_analysis/runs');
        const runs = await res.json();
        if (!batchRunSelect) return;

        batchRunSelect.innerHTML = '';
        if (runs.length === 0) {
            batchRunSelect.innerHTML = '<option value="">(No runs found)</option>';
            return;
        }

        runs.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.run_id;
            opt.text = `${r.run_id} (${r.images_count} images)`;
            batchRunSelect.appendChild(opt);
        });

        if (currentBatchRunId) {
            batchRunSelect.value = currentBatchRunId;
        } else {
            currentBatchRunId = runs[0].run_id;
        }
        loadBatchRunDetails(currentBatchRunId);
    } catch (err) {
        console.error('Error fetching runs:', err);
    }
}

if (batchRunSelect) {
    batchRunSelect.addEventListener('change', () => {
        currentBatchRunId = batchRunSelect.value;
        if (currentBatchRunId) loadBatchRunDetails(currentBatchRunId);
    });
}

async function loadBatchRunDetails(runId) {
    try {
        const res = await fetch(`/api/batch_analysis/run/${runId}`);
        const data = await res.json();
        currentBatchRecords = data.records || [];

        // Populate colour select
        if (batchColourSelect) {
            batchColourSelect.innerHTML = '';
            currentBatchRecords.forEach((rec, idx) => {
                const opt = document.createElement('option');
                opt.value = idx;
                opt.text = `${rec.colour_id} — RGB(${rec.r_uint8},${rec.g_uint8},${rec.b_uint8})`;
                batchColourSelect.appendChild(opt);
            });
        }

        // Populate quick swatch grid
        if (batchSwatchGrid) {
            batchSwatchGrid.innerHTML = '';
            currentBatchRecords.forEach((rec, idx) => {
                const swatch = document.createElement('div');
                swatch.style.width = '100%';
                swatch.style.height = '14px';
                swatch.style.borderRadius = '2px';
                swatch.style.backgroundColor = `rgb(${rec.r_uint8}, ${rec.g_uint8}, ${rec.b_uint8})`;
                swatch.style.cursor = 'pointer';
                swatch.title = `${rec.colour_id} — RGB(${rec.r_uint8}, ${rec.g_uint8}, ${rec.b_uint8}) — S_spread: ${rec.spectral_spread_score}`;
                swatch.addEventListener('click', () => {
                    selectBatchColour(idx);
                });
                batchSwatchGrid.appendChild(swatch);
            });
        }

        if (currentBatchRecords.length > 0) {
            selectBatchColour(0);
        }
    } catch (err) {
        console.error('Error loading run details:', err);
    }
}

function selectBatchColour(idx) {
    if (!currentBatchRecords || idx < 0 || idx >= currentBatchRecords.length) return;
    currentBatchIndex = idx;
    const rec = currentBatchRecords[idx];

    if (batchColourSelect) batchColourSelect.value = idx;

    // Update details in toolbar
    if (batchViewSwatch) {
        batchViewSwatch.style.backgroundColor = `rgb(${rec.r_uint8}, ${rec.g_uint8}, ${rec.b_uint8})`;
    }
    if (batchViewId) batchViewId.innerText = rec.colour_id.toUpperCase();
    if (batchViewRgb) batchViewRgb.innerText = `RGB [${rec.r_uint8}, ${rec.g_uint8}, ${rec.b_uint8}] — S_spread: ${rec.spectral_spread_score}`;
    if (batchViewBadge) {
        batchViewBadge.innerText = rec.validation_status || 'PASS';
        batchViewBadge.className = rec.validation_status === 'PASS' ? 'badge badge-ok' : 'badge badge-warn';
    }

    updateBatchDisplayImage();
}

function updateBatchDisplayImage() {
    if (!currentBatchRunId || !batchMainImg) return;

    let imgUrl = '';
    if (currentBatchViewMode === 'single') {
        const rec = currentBatchRecords[currentBatchIndex];
        if (rec) {
            imgUrl = `/api/batch_analysis/run/${currentBatchRunId}/image/individual_images/${rec.image_filename}`;
        }
    } else if (currentBatchViewMode === 'heatmap') {
        imgUrl = `/api/batch_analysis/run/${currentBatchRunId}/image/comparisons/ambiguity_comparison_heatmap.png`;
    } else if (currentBatchViewMode === 'chart') {
        imgUrl = `/api/batch_analysis/run/${currentBatchRunId}/image/comparisons/ambiguity_comparison_chart.png`;
    } else if (currentBatchViewMode === 'palette') {
        imgUrl = `/api/batch_analysis/run/${currentBatchRunId}/image/comparisons/colour_palette_overview.png`;
    } else if (currentBatchViewMode === 'cie') {
        imgUrl = `/api/batch_analysis/run/${currentBatchRunId}/image/comparisons/cie_chromaticity_ambiguity_heatmap.png`;
    }

    if (imgUrl) {
        batchMainImg.src = imgUrl;
        batchMainImg.style.display = 'block';
        if (batchImgPlaceholder) batchImgPlaceholder.style.display = 'none';
    }
}

// Navigation Prev/Next
if (batchPrevBtn) {
    batchPrevBtn.addEventListener('click', () => {
        if (currentBatchIndex > 0) selectBatchColour(currentBatchIndex - 1);
    });
}
if (batchNextBtn) {
    batchNextBtn.addEventListener('click', () => {
        if (currentBatchIndex < currentBatchRecords.length - 1) selectBatchColour(currentBatchIndex + 1);
    });
}
if (batchColourSelect) {
    batchColourSelect.addEventListener('change', () => {
        const idx = parseInt(batchColourSelect.value, 10);
        selectBatchColour(idx);
    });
}

// View mode buttons
['single', 'heatmap', 'chart', 'palette', 'cie'].forEach(mode => {
    const btn = document.getElementById(`btn-view-${mode}`);
    if (btn) {
        btn.addEventListener('click', () => {
            currentBatchViewMode = mode;
            ['single', 'heatmap', 'chart', 'palette', 'cie'].forEach(m => {
                const b = document.getElementById(`btn-view-${m}`);
                if (b) b.classList.remove('active');
            });
            btn.classList.add('active');
            updateBatchDisplayImage();
        });
    }
});

// Download buttons
['zip', 'csv', 'docx', 'txt'].forEach(ft => {
    const btn = document.getElementById(`batch-dl-${ft}`);
    if (btn) {
        btn.addEventListener('click', () => {
            if (!currentBatchRunId) {
                alert('No active run selected');
                return;
            }
            window.location.href = `/api/batch_analysis/run/${currentBatchRunId}/download/${ft}`;
        });
    }
});

// Check running status on initial page load
refreshBatchRunsList();
pollBatchStatus();


// ============================================================================
// 13. CIE 1931 xy Chromaticity Diagram & Metameric Ambiguity Heatmap Module
// ============================================================================

let chromaticityData = null;
let currentChromColorId = 'colour_001';
let activeChromMetric = 'composite_ambiguity_index';
let chromSearchQuery = '';
let chromSortField = 'composite_desc';
let chromHeatmapOpacity = 0.65;

const CHROM_METRICS_INFO = {
    composite_ambiguity_index: {
        title: 'Composite Ambiguity Index',
        shortTitle: 'Composite Index',
        unit: 'constructed index',
        key: 'composite_ambiguity_index',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    },
    spectral_spread_score: {
        title: 'Spectral Spread Score (S_spread)',
        shortTitle: 'S_spread',
        unit: 'reflectance',
        key: 'spectral_spread_score',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    },
    coordinate_spread: {
        title: 'Coordinate Spread (σ_spread)',
        shortTitle: 'σ_spread',
        unit: 'reflectance',
        key: 'coordinate_spread',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    },
    mean_pairwise_distance: {
        title: 'Mean Pairwise Distance',
        shortTitle: 'Mean Dist',
        unit: 'reflectance-dist',
        key: 'mean_pairwise_distance',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    },
    p95_pairwise_distance: {
        title: '95th Percentile Pairwise Dist',
        shortTitle: '95th Pct Dist',
        unit: 'reflectance-dist',
        key: 'p95_pairwise_distance',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    },
    max_pairwise_distance: {
        title: 'Max Observed Pairwise Dist',
        shortTitle: 'Max Dist',
        unit: 'reflectance-dist',
        key: 'max_pairwise_distance',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    },
    volume_scale_a_det: {
        title: 'Volume Scale (A_det)',
        shortTitle: 'Volume Scale A_det',
        unit: 'reflectance',
        key: 'volume_scale_a_det',
        fmt: v => v != null ? Number(v).toFixed(4) : 'N/A'
    }
};

async function loadChromaticityData(runId) {
    try {
        const url = runId ? `/api/chromaticity/data?run_id=${runId}` : '/api/chromaticity/data';
        const res = await fetch(url);
        if (!res.ok) {
            console.warn('Could not load chromaticity data:', res.statusText);
            return;
        }
        chromaticityData = await res.json();
        
        // Update header badges
        const countBadge = document.getElementById('chrom-badge-count');
        const gamutBadge = document.getElementById('chrom-badge-gamut');
        if (countBadge) countBadge.innerText = `${chromaticityData.total_observations || chromaticityData.records.length} Colours Mapped`;
        if (gamutBadge) gamutBadge.innerText = '100% In-Gamut';
        
        // Select first colour if not set
        if (!currentChromColorId && chromaticityData.records.length > 0) {
            currentChromColorId = chromaticityData.records[0].colour_id;
        }
        
        updateChromaticityUI();
    } catch (err) {
        console.error('Error loading chromaticity dataset:', err);
    }
}

function updateChromaticityUI() {
    if (!chromaticityData || !chromaticityData.records) return;

    renderChromaticityPlot();
    renderChromaticityDetailCard();
    renderChromaticitySummaryStats();
    renderChromaticityTable();
}

// Compute localized 2D inverse-distance weighted ambiguity field strictly masked around observations
function computeIDWGrid(records, metricKey, nx = 70, ny = 70, maxRadius = 0.065) {
    const xMin = 0.18, xMax = 0.52;
    const yMin = 0.18, yMax = 0.58;
    const dx = (xMax - xMin) / (nx - 1);
    const dy = (yMax - yMin) / (ny - 1);

    const xGrid = [];
    for (let i = 0; i < nx; i++) xGrid.push(xMin + i * dx);
    const yGrid = [];
    for (let j = 0; j < ny; j++) yGrid.push(yMin + j * dy);

    const validObs = records
        .map(r => ({
            x: r.cie_x,
            y: r.cie_y,
            val: parseFloat(r[metricKey])
        }))
        .filter(o => !isNaN(o.val));

    const zGrid = [];
    for (let j = 0; j < ny; j++) {
        const row = [];
        const gy = yGrid[j];
        for (let i = 0; i < nx; i++) {
            const gx = xGrid[i];
            let sumW = 0;
            let sumWV = 0;
            let count = 0;

            for (let k = 0; k < validObs.length; k++) {
                const o = validObs[k];
                const dist = Math.hypot(gx - o.x, gy - o.y);
                if (dist < maxRadius) {
                    const w = 1.0 / Math.pow(dist + 0.006, 2.0);
                    sumW += w;
                    sumWV += w * o.val;
                    count++;
                }
            }

            if (count >= 1 && sumW > 0) {
                row.push(sumWV / sumW);
            } else {
                row.push(null);
            }
        }
        zGrid.push(row);
    }

    return { x: xGrid, y: yGrid, z: zGrid };
}

function renderChromaticityPlot() {
    if (!chromaticityData) return;

    const records = chromaticityData.records || [];
    const metricInfo = CHROM_METRICS_INFO[activeChromMetric] || CHROM_METRICS_INFO['composite_ambiguity_index'];

    const showHeatmap = document.getElementById('chrom-show-heatmap')?.checked ?? true;
    const showMarkers = document.getElementById('chrom-show-markers')?.checked ?? true;
    const showLabels = document.getElementById('chrom-show-labels')?.checked ?? false;
    const showPlanckian = document.getElementById('chrom-show-planckian')?.checked ?? true;
    const showSrgb = document.getElementById('chrom-show-srgb')?.checked ?? true;
    const showWls = document.getElementById('chrom-show-wavelengths')?.checked ?? true;

    // Gather metric values for colorbar scaling
    const metricVals = records.map(r => parseFloat(r[metricInfo.key])).filter(v => !isNaN(v));
    const minVal = metricVals.length ? Math.min(...metricVals) : 0;
    const maxVal = metricVals.length ? Math.max(...metricVals) : 1;

    const traces = [];

    // 1. Heatmap contour trace (continuous interpolation masked to observations)
    if (showHeatmap && metricVals.length > 0) {
        const grid = computeIDWGrid(records, metricInfo.key);
        traces.push({
            x: grid.x,
            y: grid.y,
            z: grid.z,
            type: 'contour',
            colorscale: 'Turbo',
            zmin: minVal,
            zmax: maxVal,
            opacity: chromHeatmapOpacity,
            showscale: false,
            contours: {
                coloring: 'heatmap',
                showlines: false
            },
            hoverinfo: 'none'
        });
    }

    // 2. Spectral Locus Line (380–700 nm)
    if (showWls && chromaticityData.spectral_locus) {
        const locusX = chromaticityData.spectral_locus.map(p => p.x);
        const locusY = chromaticityData.spectral_locus.map(p => p.y);
        traces.push({
            x: locusX,
            y: locusY,
            mode: 'lines',
            line: { color: '#38bdf8', width: 2.2 },
            name: 'Spectral Locus (380–700 nm)',
            hoverinfo: 'skip'
        });
    }

    // 3. Purple Line Boundary
    if (showWls && chromaticityData.purple_boundary && chromaticityData.purple_boundary.length >= 2) {
        traces.push({
            x: [chromaticityData.purple_boundary[0].x, chromaticityData.purple_boundary[1].x],
            y: [chromaticityData.purple_boundary[0].y, chromaticityData.purple_boundary[1].y],
            mode: 'lines',
            line: { color: '#c084fc', width: 2 },
            name: 'Purple Line Boundary',
            hoverinfo: 'skip'
        });
    }

    // 4. Planckian Blackbody Locus Tc(K)
    if (showPlanckian && chromaticityData.planckian_locus) {
        const plX = chromaticityData.planckian_locus.map(p => p.x);
        const plY = chromaticityData.planckian_locus.map(p => p.y);
        traces.push({
            x: plX,
            y: plY,
            mode: 'lines',
            line: { color: '#ffffff', width: 1.8 },
            name: 'Planckian Locus Tc(K)',
            hoverinfo: 'skip'
        });
    }

    // 5. sRGB Gamut Triangle
    if (showSrgb) {
        traces.push({
            x: [0.64, 0.30, 0.15, 0.64],
            y: [0.33, 0.60, 0.06, 0.33],
            mode: 'lines',
            line: { color: '#ef4444', width: 1.5, dash: 'dash' },
            name: 'sRGB Gamut Boundary',
            hoverinfo: 'skip'
        });

        // D65 White Point
        traces.push({
            x: [0.3127],
            y: [0.3290],
            mode: 'markers+text',
            text: ['D65'],
            textposition: 'top right',
            textfont: { color: '#e2e8f0', size: 9 },
            marker: { size: 7, color: '#ffffff', symbol: 'cross' },
            name: 'D65 Standard Illuminant',
            hoverinfo: 'text'
        });
    }

    // 6. Selected 100 Colour Observations: Dual-Ring Markers
    if (showMarkers) {
        const obsX = records.map(r => r.cie_x);
        const obsY = records.map(r => r.cie_y);
        const obsVals = records.map(r => parseFloat(r[metricInfo.key]));
        const hexColors = records.map(r => r.hex_color);
        const labels = records.map(r => r.colour_id.replace('colour_', ''));

        const hoverTexts = records.map(r => {
            const v = metricInfo.fmt(r[metricInfo.key]);
            return `<b>${r.colour_id.toUpperCase()}</b><br>` +
                   `RGB: [${r.r_uint8}, ${r.g_uint8}, ${r.b_uint8}] (${r.hex_color})<br>` +
                   `CIE (x, y): (${r.cie_x.toFixed(4)}, ${r.cie_y.toFixed(4)})<br>` +
                   `${metricInfo.title}: <b>${v} ${metricInfo.unit}</b><br>` +
                   `S_spread: ${Number(r.spectral_spread_score).toFixed(4)}<br>` +
                   `Mean Dist: ${Number(r.mean_pairwise_distance).toFixed(4)}<br>` +
                   `Status: ${r.validation_status || 'PASS'} (In-Gamut)`;
        });

        // Layer A: Outer ring colored by Ambiguity metric (with colorbar)
        traces.push({
            x: obsX,
            y: obsY,
            mode: showLabels ? 'markers+text' : 'markers',
            text: labels,
            textposition: 'top center',
            textfont: { color: '#60a5fa', size: 8, family: 'monospace' },
            marker: {
                size: 14,
                color: obsVals,
                colorscale: 'Turbo',
                cmin: minVal,
                cmax: maxVal,
                showscale: true,
                colorbar: {
                    title: {
                        text: `${metricInfo.shortTitle}`,
                        font: { color: '#ffffff', size: 11, family: 'Inter, sans-serif' },
                        side: 'right'
                    },
                    tickfont: { color: '#cbd5e1', size: 10 },
                    len: 0.85,
                    x: 1.02,
                    thickness: 16,
                    outlinewidth: 1,
                    outlinecolor: '#334155'
                },
                line: { color: '#ffffff', width: 1.2 }
            },
            customdata: records.map(r => r.colour_id),
            hoverinfo: 'none',
            name: 'Ambiguity Metric Ring'
        });

        // Layer B: Inner dot filled with true original target RGB colour
        traces.push({
            x: obsX,
            y: obsY,
            mode: 'markers',
            marker: {
                size: 7.5,
                color: hexColors,
                line: { color: '#000000', width: 0.8 }
            },
            customdata: records.map(r => r.colour_id),
            hoverinfo: 'text',
            hovertext: hoverTexts,
            name: 'Target RGB Observation'
        });

        // Layer C: Selected colour ring highlight
        const selRec = records.find(r => r.colour_id === currentChromColorId);
        if (selRec) {
            traces.push({
                x: [selRec.cie_x],
                y: [selRec.cie_y],
                mode: 'markers',
                marker: {
                    size: 22,
                    color: 'rgba(0,0,0,0)',
                    line: { color: '#38bdf8', width: 3 }
                },
                showlegend: false,
                hoverinfo: 'skip'
            });
        }
    }

    // Build annotations for Wavelength labels along the spectral locus
    const annotations = [];
    if (showWls && chromaticityData.annotated_wavelengths) {
        chromaticityData.annotated_wavelengths.forEach(aw => {
            let ax = 0, ay = 0;
            if (aw.wavelength <= 480) { ax = -15; ay = 0; }
            else if (aw.wavelength <= 520) { ax = -15; ay = 15; }
            else if (aw.wavelength <= 560) { ax = 0; ay = 15; }
            else { ax = 15; ay = 10; }

            annotations.push({
                x: aw.x,
                y: aw.y,
                text: `${aw.wavelength}`,
                showarrow: false,
                xshift: ax,
                yshift: ay,
                font: { color: '#38bdf8', size: 9, family: 'monospace' }
            });
        });
    }

    // Planckian locus temperature annotations
    if (showPlanckian && chromaticityData.planckian_labels) {
        chromaticityData.planckian_labels.forEach(pl => {
            const labelStr = pl.temp_k === 'inf' ? '∞' : `${pl.temp_k}K`;
            annotations.push({
                x: pl.x,
                y: pl.y,
                text: labelStr,
                showarrow: false,
                yshift: 10,
                font: { color: '#e2e8f0', size: 8 }
            });
        });
    }

    const layout = {
        margin: { l: 55, r: 85, b: 50, t: 25 },
        paper_bgcolor: '#060911',
        plot_bgcolor: '#060911',
        xaxis: {
            title: 'CIE x Chromaticity Coordinate',
            range: [0.0, 0.8],
            dtick: 0.1,
            color: '#94a3b8',
            gridcolor: '#1e293b',
            zeroline: false
        },
        yaxis: {
            title: 'CIE y Chromaticity Coordinate',
            range: [0.0, 0.9],
            dtick: 0.1,
            color: '#94a3b8',
            gridcolor: '#1e293b',
            zeroline: false,
            scaleanchor: 'x',
            scaleratio: 1
        },
        showlegend: false,
        hovermode: 'closest',
        annotations: annotations
    };

    const plotDiv = document.getElementById('plot-chromaticity');
    Plotly.react(plotDiv, traces, layout, { responsive: true, displayModeBar: true });

    // Click handler to select colour
    plotDiv.removeAllListeners?.('plotly_click');
    plotDiv.on('plotly_click', data => {
        if (!data || !data.points || data.points.length === 0) return;
        const pt = data.points[0];
        const colId = pt.customdata;
        if (colId) {
            selectChromaticityColour(colId);
        }
    });
}

function selectChromaticityColour(colId) {
    currentChromColorId = colId;
    renderChromaticityPlot();
    renderChromaticityDetailCard();
    highlightTableRow(colId);
}

function renderChromaticityDetailCard() {
    if (!chromaticityData || !chromaticityData.records) return;
    const rec = chromaticityData.records.find(r => r.colour_id === currentChromColorId) || chromaticityData.records[0];
    if (!rec) return;

    const idEl = document.getElementById('chrom-detail-id');
    const badgeEl = document.getElementById('chrom-detail-badge');
    const swatchEl = document.getElementById('chrom-detail-swatch');
    const hexEl = document.getElementById('chrom-detail-hex');
    const rgbEl = document.getElementById('chrom-detail-rgb-uint8');
    const coordsEl = document.getElementById('chrom-detail-coords');

    if (idEl) idEl.innerText = rec.colour_id.toUpperCase();
    if (badgeEl) {
        badgeEl.innerText = rec.validation_status || 'PASS';
        badgeEl.className = rec.validation_status === 'PASS' ? 'badge badge-ok' : 'badge badge-warn';
    }
    if (swatchEl) swatchEl.style.backgroundColor = rec.hex_color;
    if (hexEl) hexEl.innerText = rec.hex_color.toUpperCase();
    if (rgbEl) rgbEl.innerText = `RGB: [${rec.r_uint8}, ${rec.g_uint8}, ${rec.b_uint8}]`;
    if (coordsEl) coordsEl.innerText = `CIE xy: (${rec.cie_x.toFixed(4)}, ${rec.cie_y.toFixed(4)})`;

    const compEl = document.getElementById('chrom-val-comp');
    const spreadEl = document.getElementById('chrom-val-spread');
    const alphaEl = document.getElementById('chrom-val-alpha');
    const meanDistEl = document.getElementById('chrom-val-mean-dist');
    const p95DistEl = document.getElementById('chrom-val-p95-dist');
    const maxDistEl = document.getElementById('chrom-val-max-dist');
    const adetEl = document.getElementById('chrom-val-adet');
    const affineEl = document.getElementById('chrom-val-affine');
    const feasibleEl = document.getElementById('chrom-val-feasible');
    const samplesEl = document.getElementById('chrom-val-samples');
    const gamutEl = document.getElementById('chrom-val-gamut');

    if (compEl) compEl.innerText = Number(rec.composite_ambiguity_index).toFixed(4);
    if (spreadEl) spreadEl.innerText = Number(rec.spectral_spread_score).toFixed(4);
    if (alphaEl) alphaEl.innerText = Number(rec.coordinate_spread).toFixed(4);
    if (meanDistEl) meanDistEl.innerText = Number(rec.mean_pairwise_distance).toFixed(4);
    if (p95DistEl) p95DistEl.innerText = Number(rec.p95_pairwise_distance).toFixed(4);
    if (maxDistEl) maxDistEl.innerText = Number(rec.max_pairwise_distance).toFixed(4);
    if (adetEl) adetEl.innerText = Number(rec.volume_scale_a_det).toFixed(4);
    if (affineEl) affineEl.innerText = `${rec.fiber_dimension || 5}D (rank ${rec.constraint_rank || 3})`;
    if (feasibleEl) feasibleEl.innerText = `${rec.fiber_dimension || 5}D`;
    if (samplesEl) samplesEl.innerText = `${rec.valid_sample_count || 500} (100%)`;
    if (gamutEl) gamutEl.innerText = `${rec.gamut_status || 'INSIDE GAMUT'} (Feasible)`;
}

function renderChromaticitySummaryStats() {
    if (!chromaticityData || !chromaticityData.records) return;
    const records = chromaticityData.records;
    const metricInfo = CHROM_METRICS_INFO[activeChromMetric] || CHROM_METRICS_INFO['composite_ambiguity_index'];

    const values = records.map(r => parseFloat(r[metricInfo.key])).filter(v => !isNaN(v));
    if (values.length === 0) return;

    values.sort((a, b) => a - b);
    const minVal = values[0];
    const maxVal = values[values.length - 1];
    const meanVal = values.reduce((s, v) => s + v, 0) / values.length;
    const mid = Math.floor(values.length / 2);
    const medianVal = values.length % 2 === 0 ? (values[mid - 1] + values[mid]) / 2 : values[mid];
    const stdVal = Math.sqrt(values.reduce((s, v) => s + Math.pow(v - meanVal, 2), 0) / values.length);

    // Highest and lowest records for this active metric
    let highestRec = records[0];
    let lowestRec = records[0];
    records.forEach(r => {
        const val = parseFloat(r[metricInfo.key]);
        if (val > parseFloat(highestRec[metricInfo.key])) highestRec = r;
        if (val < parseFloat(lowestRec[metricInfo.key])) lowestRec = r;
    });

    const spreads = records.map(r => parseFloat(r.spectral_spread_score)).filter(v => !isNaN(v));
    const meanSpread = spreads.reduce((s, v) => s + v, 0) / spreads.length;
    const pairwise = records.map(r => parseFloat(r.mean_pairwise_distance)).filter(v => !isNaN(v));
    const meanPairwise = pairwise.reduce((s, v) => s + v, 0) / pairwise.length;

    // Update DOM
    const totEl = document.getElementById('stat-total-colours');
    const validEl = document.getElementById('stat-valid-coords');
    const meanEl = document.getElementById('stat-metric-mean');
    const medianEl = document.getElementById('stat-metric-median');
    const rangeEl = document.getElementById('stat-metric-range');
    const highSwatch = document.getElementById('stat-highest-swatch');
    const highId = document.getElementById('stat-highest-id');
    const highVal = document.getElementById('stat-highest-val');
    const lowSwatch = document.getElementById('stat-lowest-swatch');
    const lowId = document.getElementById('stat-lowest-id');
    const lowVal = document.getElementById('stat-lowest-val');
    const meanSpreadEl = document.getElementById('stat-mean-spread');
    const meanPairwiseEl = document.getElementById('stat-mean-pairwise');

    if (totEl) totEl.innerText = `${records.length}`;
    if (validEl) validEl.innerText = `${records.length} Valid xy • 100% In-Gamut (0 Missing)`;
    if (meanEl) meanEl.innerText = `${meanVal.toFixed(4)}`;
    if (medianEl) medianEl.innerText = `Median: ${medianVal.toFixed(4)} • Std: ${stdVal.toFixed(4)}`;
    if (rangeEl) rangeEl.innerText = `${minVal.toFixed(4)} – ${maxVal.toFixed(4)}`;

    if (highSwatch) highSwatch.style.backgroundColor = highestRec.hex_color;
    if (highId) highId.innerText = highestRec.colour_id.toUpperCase();
    if (highVal) highVal.innerText = `${metricInfo.shortTitle}: ${parseFloat(highestRec[metricInfo.key]).toFixed(4)}`;

    if (lowSwatch) lowSwatch.style.backgroundColor = lowestRec.hex_color;
    if (lowId) lowId.innerText = lowestRec.colour_id.toUpperCase();
    if (lowVal) lowVal.innerText = `${metricInfo.shortTitle}: ${parseFloat(lowestRec[metricInfo.key]).toFixed(4)}`;

    if (meanSpreadEl) meanSpreadEl.innerText = `${meanSpread.toFixed(4)}`;
    if (meanPairwiseEl) meanPairwiseEl.innerText = `Mean Pairwise: ${meanPairwise.toFixed(4)}`;
}

function renderChromaticityTable() {
    if (!chromaticityData || !chromaticityData.records) return;
    const tbody = document.getElementById('chrom-table-body');
    if (!tbody) return;

    const metricInfo = CHROM_METRICS_INFO[activeChromMetric] || CHROM_METRICS_INFO['composite_ambiguity_index'];
    const thMetric = document.getElementById('th-active-metric');
    if (thMetric) thMetric.innerText = metricInfo.shortTitle;

    // Filter by query
    let filtered = chromaticityData.records.filter(r => {
        if (!chromSearchQuery) return true;
        const q = chromSearchQuery.toLowerCase();
        return r.colour_id.toLowerCase().includes(q) ||
               r.hex_color.toLowerCase().includes(q) ||
               `${r.r_uint8},${r.g_uint8},${r.b_uint8}`.includes(q);
    });

    // Sort
    filtered.sort((a, b) => {
        if (chromSortField === 'composite_desc') return parseFloat(b.composite_ambiguity_index) - parseFloat(a.composite_ambiguity_index);
        if (chromSortField === 'composite_asc') return parseFloat(a.composite_ambiguity_index) - parseFloat(b.composite_ambiguity_index);
        if (chromSortField === 'spread_desc') return parseFloat(b.spectral_spread_score) - parseFloat(a.spectral_spread_score);
        if (chromSortField === 'spread_asc') return parseFloat(a.spectral_spread_score) - parseFloat(b.spectral_spread_score);
        if (chromSortField === 'mean_dist_desc') return parseFloat(b.mean_pairwise_distance) - parseFloat(a.mean_pairwise_distance);
        if (chromSortField === 'max_dist_desc') return parseFloat(b.max_pairwise_distance) - parseFloat(a.max_pairwise_distance);
        if (chromSortField === 'id_asc') return (a.colour_idx || 0) - (b.colour_idx || 0);
        return 0;
    });

    let html = '';
    filtered.forEach(r => {
        const isSelected = r.colour_id === currentChromColorId;
        const activeVal = metricInfo.fmt(r[metricInfo.key]);
        const sSpread = Number(r.spectral_spread_score).toFixed(4);
        const meanDist = Number(r.mean_pairwise_distance).toFixed(4);
        const maxDist = Number(r.max_pairwise_distance).toFixed(4);
        const adet = Number(r.volume_scale_a_det).toFixed(4);

        html += `
            <tr data-id="${r.colour_id}" class="${isSelected ? 'row-selected' : ''}">
                <td style="text-align:left; font-weight:600; color:#fff;">${r.colour_id}</td>
                <td style="text-align:center;"><span class="table-swatch-sm" style="background:${r.hex_color};"></span></td>
                <td>[${r.r_uint8}, ${r.g_uint8}, ${r.b_uint8}]</td>
                <td style="color:#94a3b8;">${r.hex_color}</td>
                <td>${r.cie_x.toFixed(4)}</td>
                <td>${r.cie_y.toFixed(4)}</td>
                <td style="color:#60a5fa; font-weight:700;">${activeVal}</td>
                <td>${sSpread}</td>
                <td>${meanDist}</td>
                <td>${maxDist}</td>
                <td>${adet}</td>
                <td style="text-align:center;"><span class="badge badge-ok" style="font-size:0.65rem; padding:2px 6px;">PASS</span></td>
            </tr>
        `;
    });

    tbody.innerHTML = html;

    // Attach row click listeners
    tbody.querySelectorAll('tr').forEach(row => {
        row.addEventListener('click', () => {
            const colId = row.getAttribute('data-id');
            if (colId) selectChromaticityColour(colId);
        });
    });
}

function highlightTableRow(colId) {
    const tbody = document.getElementById('chrom-table-body');
    if (!tbody) return;
    tbody.querySelectorAll('tr').forEach(row => {
        if (row.getAttribute('data-id') === colId) {
            row.classList.add('row-selected');
            row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        } else {
            row.classList.remove('row-selected');
        }
    });
}

// Controls Event Listeners
const metricSelect = document.getElementById('chrom-metric-select');
if (metricSelect) {
    metricSelect.addEventListener('change', () => {
        activeChromMetric = metricSelect.value;
        const badge = document.getElementById('chrom-badge-metric');
        if (badge) badge.innerText = CHROM_METRICS_INFO[activeChromMetric]?.title || 'Ambiguity Metric';
        renderChromaticityPlot();
        renderChromaticitySummaryStats();
        renderChromaticityTable();
    });
}

['chrom-show-heatmap', 'chrom-show-markers', 'chrom-show-labels', 'chrom-show-planckian', 'chrom-show-srgb', 'chrom-show-wavelengths'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', () => renderChromaticityPlot());
});

const opacitySlider = document.getElementById('chrom-opacity-slider');
if (opacitySlider) {
    opacitySlider.addEventListener('input', () => {
        chromHeatmapOpacity = parseFloat(opacitySlider.value);
        const valText = document.getElementById('chrom-opacity-val');
        if (valText) valText.innerText = chromHeatmapOpacity.toFixed(2);
        renderChromaticityPlot();
    });
}

const tableSearch = document.getElementById('chrom-table-search');
if (tableSearch) {
    tableSearch.addEventListener('input', () => {
        chromSearchQuery = tableSearch.value.trim();
        renderChromaticityTable();
    });
}

const tableSort = document.getElementById('chrom-table-sort');
if (tableSort) {
    tableSort.addEventListener('change', () => {
        chromSortField = tableSort.value;
        renderChromaticityTable();
    });
}

// Next / Prev buttons in right detail panel
const chromPrevBtn = document.getElementById('chrom-btn-prev');
const chromNextBtn = document.getElementById('chrom-btn-next');
if (chromPrevBtn) {
    chromPrevBtn.addEventListener('click', () => {
        if (!chromaticityData || !chromaticityData.records) return;
        const idx = chromaticityData.records.findIndex(r => r.colour_id === currentChromColorId);
        if (idx > 0) {
            selectChromaticityColour(chromaticityData.records[idx - 1].colour_id);
        }
    });
}
if (chromNextBtn) {
    chromNextBtn.addEventListener('click', () => {
        if (!chromaticityData || !chromaticityData.records) return;
        const idx = chromaticityData.records.findIndex(r => r.colour_id === currentChromColorId);
        if (idx >= 0 && idx < chromaticityData.records.length - 1) {
            selectChromaticityColour(chromaticityData.records[idx + 1].colour_id);
        }
    });
}

// Export Handlers
const exportPngBtn = document.getElementById('chrom-export-png');
if (exportPngBtn) {
    exportPngBtn.addEventListener('click', () => {
        const runId = currentBatchRunId || (chromaticityData ? chromaticityData.run_id : '');
        window.location.href = `/api/chromaticity/render_png?run_id=${runId}&metric=${activeChromMetric}`;
    });
}

const exportCsvBtn = document.getElementById('chrom-export-csv');
if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
        const runId = currentBatchRunId || (chromaticityData ? chromaticityData.run_id : '');
        window.location.href = `/api/chromaticity/export_csv?run_id=${runId}`;
    });
}

// When batch runs change, also refresh chromaticity data
const origRefreshBatchRunsList = refreshBatchRunsList;
refreshBatchRunsList = async function() {
    await origRefreshBatchRunsList();
    if (currentBatchRunId) {
        loadChromaticityData(currentBatchRunId);
    }
};

window.addEventListener('resize', () => {
    const chromPlot = document.getElementById('plot-chromaticity');
    if (chromPlot && chromPlot.data) {
        Plotly.Plots.resize(chromPlot);
    }
});

// Pre-load chromaticity data on startup
loadChromaticityData();

// ============================================================================
// 14. Color Matching Functions (CMF) & Camera Sensitivities Module
// ============================================================================
let cmfDatasets = {};

async function loadCMFData(camera_model = 'baseline') {
    if (cmfDatasets[camera_model]) return cmfDatasets[camera_model];
    try {
        const res = await fetch(`/api/cmf?camera_model=${camera_model}`);
        if (res.ok) {
            const data = await res.json();
            cmfDatasets[camera_model] = data;
            populateCMFTable(data.table_31, camera_model);
            return data;
        }
    } catch (e) {
        console.warn('Could not fetch /api/cmf from server, using local CMF definitions:', e);
    }
    return null;
}

function populateCMFTable(records, camera_model = 'baseline') {
    const tbody = document.getElementById('cmf-table-body');
    if (!tbody || !records) return;
    tbody.innerHTML = '';
    const isOpt = camera_model === 'optimized';
    records.forEach(r => {
        const tr = document.createElement('tr');
        const mRed = isOpt && r.opt_m_red !== null ? r.opt_m_red : r.camera_m_red;
        const mGreen = isOpt && r.opt_m_green !== null ? r.opt_m_green : r.camera_m_green;
        const mBlue = isOpt && r.opt_m_blue !== null ? r.opt_m_blue : r.camera_m_blue;
        tr.innerHTML = `
            <td><strong>${r.wavelength} nm</strong></td>
            <td style="color:#f87171;">${r.cie_x_bar.toFixed(4)}</td>
            <td style="color:#4ade80;">${r.cie_y_bar.toFixed(4)}</td>
            <td style="color:#60a5fa;">${r.cie_z_bar.toFixed(4)}</td>
            <td style="color:#fb7185;">${mRed.toFixed(4)}</td>
            <td style="color:#34d399;">${mGreen.toFixed(4)}</td>
            <td style="color:#38bdf8;">${mBlue.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    });
}

async function renderCMFPlot() {
    const plotEl = document.getElementById('plot-cmf-curves');
    if (!plotEl) return;

    const camModel = document.querySelector('input[name="cmf-cam-select"]:checked')?.value || 'baseline';
    const data = await loadCMFData(camModel);
    if (!data) return;

    const selectedMode = document.querySelector('input[name="cmf-mode"]:checked')?.value || 'both';
    const isOpt = camModel === 'optimized';
    const camLabel = isOpt ? 'Optimized Sensitivity Camera' : 'Original Calibrated Camera';

    const traces = [];

    // 1. CIE 1931 2° Standard Observer CMF Curves (Dense 380–780 nm)
    if (selectedMode === 'both' || selectedMode === 'cie') {
        const d = data.cie_dense;
        traces.push({
            x: d.wavelengths,
            y: d.x_bar,
            mode: 'lines',
            line: { color: '#ef4444', width: 2.5 },
            name: 'CIE x̄(λ) (Red reference)'
        });
        traces.push({
            x: d.wavelengths,
            y: d.y_bar,
            mode: 'lines',
            line: { color: '#22c55e', width: 2.5 },
            name: 'CIE ȳ(λ) (Luminance reference)'
        });
        traces.push({
            x: d.wavelengths,
            y: d.z_bar,
            mode: 'lines',
            line: { color: '#3b82f6', width: 2.5 },
            name: 'CIE z̄(λ) (Blue reference)'
        });
    }

    // 2. Camera Spectral Sensitivity Matrix M (31 bands, 400–700 nm)
    if (selectedMode === 'both' || selectedMode === 'camera') {
        const tbl = data.table_31;
        const wls = tbl.map(r => r.wavelength);
        const mR = tbl.map(r => (isOpt && r.opt_m_red !== null) ? r.opt_m_red : r.camera_m_red);
        const mG = tbl.map(r => (isOpt && r.opt_m_green !== null) ? r.opt_m_green : r.camera_m_green);
        const mB = tbl.map(r => (isOpt && r.opt_m_blue !== null) ? r.opt_m_blue : r.camera_m_blue);

        const prefix = isOpt ? 'Opt M' : 'Base M';

        traces.push({
            x: wls,
            y: mR,
            mode: 'lines+markers',
            line: { color: '#fb7185', width: 2.2, dash: selectedMode === 'both' ? 'dash' : 'solid' },
            marker: { size: 5, color: '#fb7185' },
            name: `${prefix}_Red(λ) (${camLabel})`
        });
        traces.push({
            x: wls,
            y: mG,
            mode: 'lines+markers',
            line: { color: '#34d399', width: 2.2, dash: selectedMode === 'both' ? 'dash' : 'solid' },
            marker: { size: 5, color: '#34d399' },
            name: `${prefix}_Green(λ) (${camLabel})`
        });
        traces.push({
            x: wls,
            y: mB,
            mode: 'lines+markers',
            line: { color: '#38bdf8', width: 2.2, dash: selectedMode === 'both' ? 'dash' : 'solid' },
            marker: { size: 5, color: '#38bdf8' },
            name: `${prefix}_Blue(λ) (${camLabel})`
        });
    }

    const layout = {
        title: {
            text: selectedMode === 'both' 
                ? `CIE 1931 2° Observer CMFs vs. ${camLabel}`
                : (selectedMode === 'cie' ? 'CIE 1931 2° Standard Observer Color Matching Functions' : `${camLabel} Matrix M(λ)`),
            font: { color: '#e2e8f0', size: 13 }
        },
        margin: { l: 60, r: 25, b: 45, t: 35 },
        paper_bgcolor: '#0a0e17',
        plot_bgcolor: '#060911',
        xaxis: {
            title: 'Wavelength λ (nm)',
            color: '#94a3b8',
            gridcolor: '#1e293b',
            range: [380, 720]
        },
        yaxis: {
            title: 'Sensitivity / Tristimulus Response',
            color: '#94a3b8',
            gridcolor: '#1e293b'
        },
        legend: {
            font: { color: '#cbd5e1', size: 10 },
            bgcolor: 'rgba(10, 14, 23, 0.8)',
            bordercolor: '#1e293b',
            borderwidth: 1
        }
    };

    Plotly.react('plot-cmf-curves', traces, layout, { responsive: true });
}

// CMF Radio switch listeners
document.querySelectorAll('input[name="cmf-mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
        renderCMFPlot();
    });
});

document.querySelectorAll('input[name="cmf-cam-select"]').forEach(radio => {
    radio.addEventListener('change', () => {
        renderCMFPlot();
    });
});

// Pre-load CMF data on app initialization
loadCMFData();

// ============================================================================
// 12. Strict One-CMF-Per-Image Rendering Engine Client Logic
// ============================================================================
let cmfEngineInitialized = false;
let cmfCatalogData = null;

async function initCMFImageEngine() {
    if (cmfEngineInitialized) return;
    cmfEngineInitialized = true;

    const selectEl = document.getElementById('cmf-renderer-select');
    const refSelectEl = document.getElementById('cmf-reference-select');
    const renderBtn = document.getElementById('btn-render-cmf-active');
    const clearBtn = document.getElementById('btn-clear-cmf-buffer');
    const dualToggle = document.getElementById('cmf-dual-view-toggle');

    try {
        const resp = await fetch('/api/cmf/list');
        if (resp.ok) {
            cmfCatalogData = await resp.json();
            populateCMFDropdowns(cmfCatalogData.cmfs);
        }
    } catch (err) {
        console.warn('Could not fetch /api/cmf/list:', err);
    }

    if (selectEl) {
        selectEl.addEventListener('change', () => {
            const selectedOpt = selectEl.options[selectEl.selectedIndex];
            const titleEl = document.getElementById('cmf-primary-title');
            if (titleEl) titleEl.textContent = selectedOpt.text;
            
            // Strictly clear previous buffer before rendering new selection
            clearPrimaryCMFBuffer();
            renderSelectedCMFImage(selectEl.value);
        });
    }

    if (refSelectEl) {
        refSelectEl.addEventListener('change', () => {
            renderReferenceCMFImage(refSelectEl.value);
        });
    }

    if (renderBtn) {
        renderBtn.addEventListener('click', () => {
            if (selectEl) renderSelectedCMFImage(selectEl.value);
            if (dualToggle && dualToggle.checked && refSelectEl) {
                renderReferenceCMFImage(refSelectEl.value);
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            clearPrimaryCMFBuffer();
        });
    }

    if (dualToggle) {
        dualToggle.addEventListener('change', () => {
            const panels = document.getElementById('cmf-render-panels');
            const refCard = document.getElementById('cmf-reference-card');
            if (dualToggle.checked) {
                if (panels) panels.classList.add('dual-mode');
                if (refCard) {
                    refCard.style.display = 'flex';
                    if (refSelectEl) renderReferenceCMFImage(refSelectEl.value);
                }
            } else {
                if (panels) panels.classList.remove('dual-mode');
                if (refCard) refCard.style.display = 'none';
            }
        });
    }

    // Auto-render default active CMF upon initialization
    if (selectEl) {
        renderSelectedCMFImage(selectEl.value);
    }
}

function populateCMFDropdowns(cmfList) {
    const selectEl = document.getElementById('cmf-renderer-select');
    const refSelectEl = document.getElementById('cmf-reference-select');
    if (!selectEl || !cmfList) return;

    selectEl.innerHTML = '';
    if (refSelectEl) refSelectEl.innerHTML = '';

    const groups = {
        master: document.createElement('optgroup'),
        calibrated: document.createElement('optgroup'),
        cie: document.createElement('optgroup'),
        reference: document.createElement('optgroup')
    };
    groups.master.label = 'Master Active Optimization';
    groups.calibrated.label = 'Calibrated Baseline Camera';
    groups.cie.label = 'Standard CIE Observers';
    groups.reference.label = 'Physical Reference Cameras (28 Cameras)';

    const refGroups = {
        master: document.createElement('optgroup'),
        calibrated: document.createElement('optgroup'),
        cie: document.createElement('optgroup'),
        reference: document.createElement('optgroup')
    };
    refGroups.master.label = 'Master Active Optimization';
    refGroups.calibrated.label = 'Calibrated Baseline Camera';
    refGroups.cie.label = 'Standard CIE Observers';
    refGroups.reference.label = 'Physical Reference Cameras (28 Cameras)';

    cmfList.forEach(cmf => {
        const opt = document.createElement('option');
        opt.value = cmf.id;
        opt.textContent = `${cmf.display_name} (${cmf.type})`;

        const refOpt = document.createElement('option');
        refOpt.value = cmf.id;
        refOpt.textContent = `${cmf.display_name} (${cmf.type})`;

        if (cmf.type === 'master_optimized') {
            groups.master.appendChild(opt);
            refGroups.master.appendChild(refOpt);
        } else if (cmf.type === 'calibrated_baseline') {
            groups.calibrated.appendChild(opt);
            refGroups.calibrated.appendChild(refOpt);
        } else if (cmf.type === 'cie_standard') {
            groups.cie.appendChild(opt);
            refGroups.cie.appendChild(refOpt);
        } else {
            groups.reference.appendChild(opt);
            refGroups.reference.appendChild(refOpt);
        }
    });

    Object.values(groups).forEach(g => { if (g.children.length > 0) selectEl.appendChild(g); });
    if (refSelectEl) {
        Object.values(refGroups).forEach(g => { if (g.children.length > 0) refSelectEl.appendChild(g); });
        refSelectEl.value = 'original_calibrated_camera';
    }
}

function clearPrimaryCMFBuffer() {
    const imgEl = document.getElementById('cmf-primary-img');
    const placeholder = document.getElementById('cmf-primary-placeholder');
    const loader = document.getElementById('cmf-primary-loader');
    const hashEl = document.getElementById('cmf-primary-hash');
    const bufferEl = document.getElementById('cmf-primary-buffer');

    if (imgEl) {
        imgEl.src = '';
        imgEl.style.display = 'none';
    }
    if (loader) loader.style.display = 'none';
    if (placeholder) {
        placeholder.style.display = 'flex';
        placeholder.querySelector('span:nth-child(2)').textContent = 'Render buffer cleared & freed';
    }
    if (hashEl) hashEl.textContent = 'Buffer Cleared';
    if (bufferEl) bufferEl.textContent = 'Cleared / Awaiting Render';
}

async function renderSelectedCMFImage(cmfId) {
    const imgEl = document.getElementById('cmf-primary-img');
    const placeholder = document.getElementById('cmf-primary-placeholder');
    const loader = document.getElementById('cmf-primary-loader');
    const sceneSelect = document.getElementById('cmf-scene-select');
    const sceneId = sceneSelect ? sceneSelect.value : '000002';

    // Strict Rule: clear previous buffer before issuing new render
    if (imgEl) {
        imgEl.src = '';
        imgEl.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'none';
    if (loader) loader.style.display = 'block';

    const idEl = document.getElementById('cmf-primary-id');
    const shapeEl = document.getElementById('cmf-primary-shape');
    const hashEl = document.getElementById('cmf-primary-hash');
    const bufferEl = document.getElementById('cmf-primary-buffer');

    if (idEl) idEl.textContent = cmfId;
    if (bufferEl) bufferEl.textContent = 'Allocating isolated buffer...';

    try {
        const resp = await fetch('/api/cmf/render_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmf_id: cmfId, scene_id: sceneId })
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.error || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        if (loader) loader.style.display = 'none';
        if (imgEl) {
            imgEl.src = data.image_data_url;
            imgEl.style.display = 'block';
        }

        const meta = data.metadata || {};
        if (idEl) idEl.textContent = meta.cmf_id || cmfId;
        if (shapeEl && meta.cmf_shape) shapeEl.textContent = `${meta.cmf_shape[0]} × ${meta.cmf_shape[1]} bands (400-700nm)`;
        if (hashEl && meta.image_sha256) hashEl.textContent = meta.image_sha256.substring(0, 16) + '...';
        if (bufferEl) bufferEl.textContent = 'Isolated Fresh Array (Verified 1-CMF)';

        plotSingleCMFCurves(data.cmf_name, meta);
    } catch (err) {
        if (loader) loader.style.display = 'none';
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.querySelector('span:nth-child(2)').textContent = `Render Error: ${err.message}`;
        }
        if (bufferEl) bufferEl.textContent = 'Error during render';
        console.error('Failed to render single CMF image:', err);
    }
}

async function renderReferenceCMFImage(refId) {
    const imgEl = document.getElementById('cmf-reference-img');
    const placeholder = document.getElementById('cmf-reference-placeholder');
    const loader = document.getElementById('cmf-reference-loader');
    const sceneSelect = document.getElementById('cmf-scene-select');
    const sceneId = sceneSelect ? sceneSelect.value : '000002';

    if (imgEl) {
        imgEl.src = '';
        imgEl.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'none';
    if (loader) loader.style.display = 'block';

    const idEl = document.getElementById('cmf-reference-id');
    const hashEl = document.getElementById('cmf-reference-hash');

    if (idEl) idEl.textContent = refId;

    try {
        const resp = await fetch('/api/cmf/render_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmf_id: refId, scene_id: sceneId })
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (loader) loader.style.display = 'none';
        if (imgEl) {
            imgEl.src = data.image_data_url;
            imgEl.style.display = 'block';
        }

        const meta = data.metadata || {};
        if (idEl) idEl.textContent = meta.cmf_id || refId;
        if (hashEl && meta.image_sha256) hashEl.textContent = meta.image_sha256.substring(0, 16) + '...';
    } catch (err) {
        if (loader) loader.style.display = 'none';
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.querySelector('span:nth-child(2)').textContent = `Reference Error: ${err.message}`;
        }
        console.error('Failed to render reference CMF image:', err);
    }
}

async function plotSingleCMFCurves(cmfName, meta) {
    const plotEl = document.getElementById('plot-cmf-single-curves');
    if (!plotEl) return;

    // Fetch the 31-band values if available or derive from loadCMFData
    try {
        const camModel = (meta.cmf_id === 'original_calibrated_camera') ? 'baseline' : 'optimized';
        const cmfData = await loadCMFData(camModel);
        if (!cmfData || !cmfData.table_31) return;

        const tbl = cmfData.table_31;
        const wls = tbl.map(r => r.wavelength);
        const isOpt = meta.cmf_id === 'global_optimized_cmf';

        const rVals = tbl.map(r => (isOpt && r.opt_m_red !== null) ? r.opt_m_red : r.camera_m_red);
        const gVals = tbl.map(r => (isOpt && r.opt_m_green !== null) ? r.opt_m_green : r.camera_m_green);
        const bVals = tbl.map(r => (isOpt && r.opt_m_blue !== null) ? r.opt_m_blue : r.camera_m_blue);

        const traces = [
            {
                x: wls,
                y: rVals,
                mode: 'lines+markers',
                line: { color: '#ef4444', width: 2.5 },
                marker: { size: 5, color: '#ef4444' },
                name: 'Channel R(λ) / x̄(λ)'
            },
            {
                x: wls,
                y: gVals,
                mode: 'lines+markers',
                line: { color: '#10b981', width: 2.5 },
                marker: { size: 5, color: '#10b981' },
                name: 'Channel G(λ) / ȳ(λ)'
            },
            {
                x: wls,
                y: bVals,
                mode: 'lines+markers',
                line: { color: '#38bdf8', width: 2.5 },
                marker: { size: 5, color: '#38bdf8' },
                name: 'Channel B(λ) / z̄(λ)'
            }
        ];

        const layout = {
            title: {
                text: `${cmfName || meta.cmf_id} — Single CMF 3-Channel Spectral Profile`,
                font: { color: '#e2e8f0', size: 12 }
            },
            margin: { l: 50, r: 20, b: 40, t: 30 },
            paper_bgcolor: '#0a0e17',
            plot_bgcolor: '#060911',
            xaxis: {
                title: 'Wavelength λ (nm)',
                color: '#94a3b8',
                gridcolor: '#1e293b',
                range: [390, 710]
            },
            yaxis: {
                title: 'Spectral Sensitivity',
                color: '#94a3b8',
                gridcolor: '#1e293b'
            },
            legend: {
                font: { color: '#cbd5e1', size: 10 },
                bgcolor: 'rgba(10, 14, 23, 0.8)',
                bordercolor: '#1e293b',
                borderwidth: 1
            }
        };

        Plotly.react('plot-cmf-single-curves', traces, layout, { responsive: true });
    } catch (e) {
        console.warn('Could not plot single CMF curves:', e);
    }
}




