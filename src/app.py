import os
import time
import uuid
import threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from scipy.spatial import ConvexHull
from src.metamer_solver import MetamerSolver
from src.spectral_reconstruction import SpectralReconstructor
from src.validation import Validator
from src.fiber_dimension import FiberDimensionAnalyzer
from src.export_utils import export_as_json, export_as_csv, export_as_markdown, convert_numpy_to_python
from src.batch_analyzer import BatchAnalyzer
from src.chromaticity_analyzer import ChromaticityAnalyzer
from src.camera_model import CameraModel

app = Flask(__name__, static_folder='../visualization')
artifacts_dir = os.path.join(os.path.dirname(__file__), '../artifacts')

solvers = {
    "baseline": MetamerSolver(artifacts_dir)
}
m_opt_path = os.path.join(artifacts_dir, 'M_camera_optimized.npy')
if os.path.exists(m_opt_path):
    try:
        M_opt_data = np.load(m_opt_path)
        cam_opt_model = CameraModel(M=M_opt_data, name="Optimized Sensitivity Camera")
        solvers["optimized"] = MetamerSolver(artifacts_dir, camera_model=cam_opt_model)
    except Exception as e:
        print(f"Warning: Failed to load optimized camera model: {e}")

def get_solver(camera_model="baseline"):
    if camera_model == "optimized" and "optimized" in solvers:
        return solvers["optimized"]
    return solvers["baseline"]

validator = Validator(artifacts_dir=artifacts_dir)
batch_analyzer = BatchAnalyzer(artifacts_dir)

# Thread-safe in-memory job store for background calculations
jobs = {}
jobs_lock = threading.Lock()


def _validate_sampling_parameters(data):
    """
    Rigorously validates sampling parameters server-side:
    - n_samples: integer in [100, 5000]
    - burn_in: integer >= 0
    - thinning: integer >= 1
    Returns (n_samples, burn_in, thinning, error_message)
    """
    raw_n = data.get('n_samples', data.get('num_points', 1000))
    if isinstance(raw_n, bool) or not (isinstance(raw_n, int) or (isinstance(raw_n, float) and raw_n.is_integer())):
        return None, None, None, "n_samples must be an integer, got: " + repr(raw_n)
    n_samples = int(raw_n)
    if n_samples < 100 or n_samples > 5000:
        return None, None, None, f"Number of samples (N) must be between 100 and 5000, got: {n_samples}"

    raw_b = data.get('burn_in', 50)
    if isinstance(raw_b, bool) or not (isinstance(raw_b, int) or (isinstance(raw_b, float) and raw_b.is_integer())):
        return None, None, None, "burn_in must be an integer >= 0, got: " + repr(raw_b)
    burn_in = int(raw_b)
    if burn_in < 0:
        return None, None, None, f"Burn-in must be an integer >= 0, got: {burn_in}"

    raw_t = data.get('thinning', 2)
    if isinstance(raw_t, bool) or not (isinstance(raw_t, int) or (isinstance(raw_t, float) and raw_t.is_integer())):
        return None, None, None, "thinning must be an integer >= 1, got: " + repr(raw_t)
    thinning = int(raw_t)
    if thinning < 1:
        return None, None, None, f"Thinning must be an integer >= 1, got: {thinning}"

    return n_samples, burn_in, thinning, None


def _execute_calculation(rgb, n_samples, burn_in, thinning, bound_mode="550nm", camera_model="baseline", progress_cb=None):
    """
    Executes the full pipeline: feasibility check, Hit-and-Run, validation,
    statistics, dimension analysis, and exact LP bounds for the selected camera model.
    """
    active_solver = get_solver(camera_model)
    is_opt = bool(camera_model == "optimized" and "optimized" in solvers)
    active_camera_label = "Optimized Sensitivity Camera" if is_opt else "Original Calibrated Camera"

    # 1. Feasibility & Exact Fiber Construction
    if progress_cb:
        progress_cb(0, n_samples, f"Checking gamut feasibility ({active_camera_label})...")
        
    fiber = active_solver.construct_fiber(rgb)
    if not fiber["feasible"]:
        return {
            "feasible": False,
            "rgb": rgb,
            "camera_model": camera_model,
            "active_camera_label": active_camera_label,
            "message": f"RGB observation is outside the prior-induced gamut for {active_camera_label}."
        }

    # 2. Hit-and-Run sampling with live progress
    if progress_cb:
        progress_cb(0, n_samples, f"Sampling N={n_samples} points with Hit-and-Run...")

    def hr_cb(current, total):
        if progress_cb:
            progress_cb(current, total, f"Sampling metamer points ({current}/{total})...")

    # Determine bound wavelengths based on requested mode:
    # '550nm': only 550nm (fastest, ~4s)
    # '7_bands': key wavelengths [400, 450, 500, 550, 600, 650, 700] (~20s)
    # 'all': all 31 wavelengths
    if bound_mode == 'all':
        bound_wls = list(range(31))
    elif bound_mode == '7_bands':
        bound_wls = [0, 5, 10, 15, 20, 25, 30]
    else:
        bound_wls = [15]

    result = active_solver.solve(
        rgb=rgb,
        num_points=n_samples,
        burn_in=burn_in,
        thinning=thinning,
        seed=42,
        compute_bounds=True,
        bound_wavelengths=bound_wls,
        progress_callback=hr_cb
    )

    if progress_cb:
        progress_cb(n_samples, n_samples, "Validating samples and computing statistics...")

    # 3. Validation with active camera model
    val_model = Validator(artifacts_dir=artifacts_dir, camera_model=active_solver.camera_model)
    validation = val_model.validate_points(rgb, result['z_points'])

    # 4. Format 3D visualization points
    z_points_3d = result['z_points'][:, :3].tolist()
    z_centroid_3d = result['z_centroid'][:3].tolist()
    z0_3d = result['z0'][:3].tolist()

    principal_dirs = []
    for i in range(3):
        p_dir = result['principal_directions'][:, i][:3]
        principal_dirs.append(p_dir.tolist())

    # 5. Low sample count warning if N < 500
    warning = None
    if n_samples < 500:
        warning = f"Low sample count (N = {n_samples}): covariance and volume estimates may be unstable (see sampling stability study)."

    response_data = {
        "feasible": True,
        "rgb": rgb,
        "camera_model": camera_model,
        "active_camera_label": active_camera_label,
        "is_optimized": is_opt,
        "fiber_dimension": int(result['fiber_dimension']),
        "constraint_rank": int(result['constraint_rank']),
        "null_space_dimension": int(result['N'].shape[1]),
        "dimension_analysis": result.get("dimension_analysis"),
        "validation": validation,
        "z_points_3d": z_points_3d,
        "z_centroid_3d": z_centroid_3d,
        "z0_3d": z0_3d,
        "principal_directions": principal_dirs,
        "ambiguity_metrics": result.get("ambiguity_metrics", {}),
        "sampling_diagnostics": result.get("sampling_diagnostics", {}),
        "exact_bounds": result.get("exact_bounds"),
        "bound_comparisons": result.get("bound_comparisons"),
        "spectral_envelope": result.get("spectral_envelope"),
        "n_samples": n_samples,
        "burn_in": burn_in,
        "thinning": thinning,
        "bound_mode": bound_mode,
        "seed": 42,
        "warning": warning
    }

    return convert_numpy_to_python(response_data)


def _worker_thread(job_id, rgb, n_samples, burn_in, thinning, bound_mode="550nm", camera_model="baseline"):
    def update_prog(current, total, stage):
        with jobs_lock:
            if job_id in jobs:
                pct = int(100.0 * current / max(total, 1)) if total > 0 else 0
                jobs[job_id]["progress"] = pct
                jobs[job_id]["stage"] = stage

    try:
        res = _execute_calculation(rgb, n_samples, burn_in, thinning, bound_mode=bound_mode, camera_model=camera_model, progress_cb=update_prog)
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["stage"] = "Done"
                jobs[job_id]["result"] = res
    except Exception as e:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/polytope', methods=['GET'])
def get_polytope():
    try:
        P_surf_pca = np.load(os.path.join(artifacts_dir, 'P_surf_pca.npy'))
        P3d = P_surf_pca[:, :3]
        hull = ConvexHull(P3d)
        return jsonify({
            "vertices": P3d.tolist(),
            "faces": hull.simplices.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cmf', methods=['GET'])
def get_cmf_data():
    """
    Exposes official CIE 1931 2° Standard Observer Color Matching Functions
    and the camera spectral sensitivity matrices (baseline and optimized).
    """
    try:
        import colour
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        wls_dense = np.arange(380, 781, 5)
        cie_dense = cmfs[wls_dense]

        wls_31 = np.arange(400, 701, 10)
        cie_31 = cmfs[wls_31]

        m_cam_path = os.path.join(artifacts_dir, 'M_camera.npy')
        M_cam = np.load(m_cam_path) if os.path.exists(m_cam_path) else None

        m_opt_path = os.path.join(artifacts_dir, 'M_camera_optimized.npy')
        has_opt = os.path.exists(m_opt_path)
        M_opt = np.load(m_opt_path) if has_opt else None

        req_model = request.args.get('camera_model', 'baseline')
        active_M = M_opt if (req_model == 'optimized' and has_opt) else M_cam
        active_label = "Optimized Sensitivity Camera" if (req_model == 'optimized' and has_opt) else "Original Calibrated Camera"

        records_31 = []
        for i, wl in enumerate(wls_31):
            records_31.append({
                "wavelength": int(wl),
                "cie_x_bar": round(float(cie_31[i, 0]), 4),
                "cie_y_bar": round(float(cie_31[i, 1]), 4),
                "cie_z_bar": round(float(cie_31[i, 2]), 4),
                "camera_m_red": round(float(active_M[0, i]), 4) if active_M is not None else 0.0,
                "camera_m_green": round(float(active_M[1, i]), 4) if active_M is not None else 0.0,
                "camera_m_blue": round(float(active_M[2, i]), 4) if active_M is not None else 0.0,
                "base_m_red": round(float(M_cam[0, i]), 4) if M_cam is not None else 0.0,
                "base_m_green": round(float(M_cam[1, i]), 4) if M_cam is not None else 0.0,
                "base_m_blue": round(float(M_cam[2, i]), 4) if M_cam is not None else 0.0,
                "opt_m_red": round(float(M_opt[0, i]), 4) if M_opt is not None else None,
                "opt_m_green": round(float(M_opt[1, i]), 4) if M_opt is not None else None,
                "opt_m_blue": round(float(M_opt[2, i]), 4) if M_opt is not None else None
            })

        return jsonify({
            "cie_dense": {
                "wavelengths": wls_dense.tolist(),
                "x_bar": [round(float(v), 5) for v in cie_dense[:, 0]],
                "y_bar": [round(float(v), 5) for v in cie_dense[:, 1]],
                "z_bar": [round(float(v), 5) for v in cie_dense[:, 2]]
            },
            "table_31": records_31,
            "active_camera_model": req_model,
            "active_camera_label": active_label,
            "has_optimized": has_opt
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cmf/list', methods=['GET'])
def get_cmf_list():
    """
    Returns the complete catalog of all available CMFs in the project:
    Global Optimized CMF, Baseline Camera, CIE Observers, and 28 Physical Cameras.
    """
    try:
        from src.cmf_image_renderer import load_all_available_cmfs
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cmf_catalog = load_all_available_cmfs(root_dir)
        summary_list = []
        for c in cmf_catalog:
            summary_list.append({
                "id": c["id"],
                "name": c["name"],
                "slug": c["slug"],
                "category": c["category"],
                "type": c["type"],
                "description": c.get("description", "")
            })
        return jsonify({"count": len(summary_list), "cmfs": summary_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cmf/render_image', methods=['GET', 'POST'])
def render_cmf_image_endpoint():
    """
    Renders an RGB image from a selected spectral scene using strictly ONE selected CMF.
    Enforces the strict one-CMF-per-image rule with fresh buffer isolation.
    """
    try:
        import io
        import base64
        from PIL import Image
        from src.cmf_image_renderer import CMFRenderer, load_all_available_cmfs

        req_data = request.args if request.method == 'GET' else (request.json or {})
        cmf_id = req_data.get('cmf_id', 'global_optimized_cmf')
        image_id = req_data.get('image_id', '000002')

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        all_cmfs = load_all_available_cmfs(root_dir)

        # Match single selected CMF
        matched_cmf = None
        for c in all_cmfs:
            if c["id"] == cmf_id or c["slug"].lower() == cmf_id.lower() or c["name"].lower() == cmf_id.lower():
                matched_cmf = c
                break

        if matched_cmf is None:
            return jsonify({"error": f"CMF '{cmf_id}' not found in available catalog."}), 404

        # Load spectral data for requested scene
        spec_file = os.path.join(root_dir, 'Spectral', f'{image_id}.npy')
        if not os.path.exists(spec_file):
            return jsonify({"error": f"Spectral data for scene '{image_id}' not found."}), 404

        spec_cube = np.load(spec_file)

        # Strict single-CMF render with isolated buffer
        rgb_arr, meta = CMFRenderer.render_single_cmf_image(
            spec_cube,
            matched_cmf,
            source_identifier=image_id
        )

        # Convert to base64 PNG
        pil_img = Image.fromarray(rgb_arr)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_b64}"

        return jsonify({
            "success": True,
            "cmf_id": matched_cmf["id"],
            "cmf_name": matched_cmf["name"],
            "cmf_slug": matched_cmf["slug"],
            "cmf_category": matched_cmf["category"],
            "image_id": image_id,
            "dimensions": list(rgb_arr.shape),
            "image_data": data_url,
            "metadata": meta
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json or {}
    
    # Validate RGB
    try:
        rgb = [float(data.get('r', 0)), float(data.get('g', 0)), float(data.get('b', 0))]
        if any(np.isnan(c) or np.isinf(c) or c < 0.0 or c > 1.0 for c in rgb):
            return jsonify({"error": "RGB values must be finite numbers between 0.0 and 1.0."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid RGB values provided."}), 400

    # Validate sampling parameters
    n_samples, burn_in, thinning, err = _validate_sampling_parameters(data)
    if err:
        return jsonify({"error": err}), 400

    bound_mode = data.get("bound_mode", "550nm")
    camera_model = data.get("camera_model", "baseline")
    is_async = bool(data.get('async', False))

    if not is_async:
        # Synchronous execution
        res = _execute_calculation(rgb, n_samples, burn_in, thinning, bound_mode=bound_mode, camera_model=camera_model)
        return jsonify(res)

    # Launch background job
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": 0,
            "stage": "Starting background job...",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "camera_model": camera_model
        }

    thread = threading.Thread(
        target=_worker_thread,
        args=(job_id, rgb, n_samples, burn_in, thinning, bound_mode, camera_model),
        daemon=True
    )
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "running",
        "n_samples": n_samples,
        "bound_mode": bound_mode,
        "camera_model": camera_model
    }), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
@app.route('/api/calculate/<job_id>', methods=['GET'])
def get_job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


@app.route('/api/jobs/<job_id>/export', methods=['GET'])
def export_job_result(job_id):
    fmt = request.args.get('format', 'json').lower()
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not job.get('result'):
            return jsonify({"error": "Job result not found"}), 404
        res = job['result']

    if fmt == 'csv':
        csv_data = export_as_csv(res)
        return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': f'attachment;filename=fiber_analysis_{job_id[:8]}.csv'})
    elif fmt in ['md', 'markdown']:
        md_data = export_as_markdown(res)
        return Response(md_data, mimetype='text/markdown', headers={'Content-Disposition': f'attachment;filename=fiber_analysis_{job_id[:8]}.md'})
    else:
        return Response(export_as_json(res), mimetype='application/json', headers={'Content-Disposition': f'attachment;filename=fiber_analysis_{job_id[:8]}.json'})


@app.route('/api/export', methods=['POST'])
def export_payload():
    fmt = request.args.get('format', 'json').lower()
    data = request.json or {}
    if fmt == 'csv':
        csv_data = export_as_csv(data)
        return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=fiber_analysis.csv'})
    elif fmt in ['md', 'markdown']:
        md_data = export_as_markdown(data)
        return Response(md_data, mimetype='text/markdown', headers={'Content-Disposition': 'attachment;filename=fiber_analysis.md'})
    else:
        return Response(export_as_json(data), mimetype='application/json', headers={'Content-Disposition': 'attachment;filename=fiber_analysis.json'})


@app.route('/api/batch_analysis/start', methods=['POST'])
def start_batch_analysis():
    data = request.json or {}
    image_path = data.get('image_path', 'Dataset/image.png')
    target_colours = int(data.get('target_colours', 100))
    n_samples = int(data.get('n_samples', 500))
    seed = int(data.get('seed', 42))
    workers = int(data.get('workers', 4))

    status = batch_analyzer.get_status()
    if status.get("status") == "running":
        return jsonify({"error": "A batch analysis run is already in progress.", "current_status": status}), 409

    def _runner():
        try:
            batch_analyzer.run_analysis(
                source_image_path=image_path,
                target_colours=target_colours,
                n_samples=n_samples,
                seed=seed,
                max_workers=workers
            )
        except Exception as e:
            print(f"Batch runner error: {e}")

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    return jsonify({
        "status": "started",
        "message": f"Batch analysis started for {target_colours} colours (N={n_samples})",
        "source_image": image_path,
        "seed": seed
    }), 202


@app.route('/api/batch_analysis/status', methods=['GET'])
def get_batch_status():
    return jsonify(batch_analyzer.get_status())


@app.route('/api/batch_analysis/runs', methods=['GET'])
def list_batch_runs():
    base_dir = os.path.abspath('random_colours_ambiguity_results')
    if not os.path.exists(base_dir):
        return jsonify([])
    
    runs = []
    for item in sorted(os.listdir(base_dir), reverse=True):
        p = os.path.join(base_dir, item)
        if os.path.isdir(p) and item.startswith("run_"):
            meta_file = os.path.join(p, "run_metadata.json")
            meta = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception:
                    pass
            img_dir = os.path.join(p, "individual_images")
            img_count = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
            runs.append({
                "run_id": item,
                "path": p,
                "images_count": img_count,
                "has_zip": os.path.exists(os.path.join(p, "complete_results.zip")),
                "metadata": meta
            })
    return jsonify(runs)


@app.route('/api/batch_analysis/run/<run_id>', methods=['GET'])
def get_batch_run_details(run_id):
    run_dir = os.path.abspath(os.path.join('random_colours_ambiguity_results', run_id))
    if not os.path.exists(run_dir):
        return jsonify({"error": "Run directory not found"}), 404
        
    csv_file = os.path.join(run_dir, "summary", "ambiguity_summary.csv")
    meta_file = os.path.join(run_dir, "run_metadata.json")
    
    meta = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            pass

    records = []
    if os.path.exists(csv_file):
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)

    return jsonify({
        "run_id": run_id,
        "metadata": meta,
        "total_colours": len(records),
        "records": records,
        "comparisons": {
            "heatmap": f"/api/batch_analysis/run/{run_id}/image/comparisons/ambiguity_comparison_heatmap.png",
            "chart": f"/api/batch_analysis/run/{run_id}/image/comparisons/ambiguity_comparison_chart.png",
            "palette": f"/api/batch_analysis/run/{run_id}/image/comparisons/colour_palette_overview.png",
            "chromaticity": f"/api/batch_analysis/run/{run_id}/image/comparisons/cie_chromaticity_ambiguity_heatmap.png"
        }
    })


@app.route('/api/batch_analysis/run/<run_id>/image/<path:filename>', methods=['GET'])
def get_batch_image(run_id, filename):
    run_dir = os.path.abspath(os.path.join('random_colours_ambiguity_results', run_id))
    full_path = os.path.abspath(os.path.join(run_dir, filename))
    if not full_path.startswith(run_dir) or not os.path.exists(full_path):
        # Also check inside individual_images if user just passed colour_001_analysis.png
        alt_path = os.path.abspath(os.path.join(run_dir, "individual_images", filename))
        if os.path.exists(alt_path):
            return send_from_directory(os.path.dirname(alt_path), os.path.basename(alt_path))
        return jsonify({"error": f"Image {filename} not found"}), 404
    return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))


@app.route('/api/batch_analysis/run/<run_id>/download/<file_type>', methods=['GET'])
def download_batch_file(run_id, file_type):
    run_dir = os.path.abspath(os.path.join('random_colours_ambiguity_results', run_id))
    if not os.path.exists(run_dir):
        return jsonify({"error": "Run not found"}), 404

    ft = file_type.lower()
    if ft == 'zip':
        zip_p = os.path.join(run_dir, "complete_results.zip")
        if os.path.exists(zip_p):
            return send_from_directory(run_dir, "complete_results.zip", as_attachment=True)
    elif ft == 'csv':
        csv_p = os.path.join(run_dir, "summary", "ambiguity_summary.csv")
        if os.path.exists(csv_p):
            return send_from_directory(os.path.dirname(csv_p), "ambiguity_summary.csv", as_attachment=True)
    elif ft == 'docx':
        docx_p = os.path.join(run_dir, "summary", "ambiguity_summary.docx")
        if os.path.exists(docx_p):
            return send_from_directory(os.path.dirname(docx_p), "ambiguity_summary.docx", as_attachment=True)
    elif ft == 'txt':
        txt_p = os.path.join(run_dir, "summary", "ambiguity_summary.txt")
        if os.path.exists(txt_p):
            return send_from_directory(os.path.dirname(txt_p), "ambiguity_summary.txt", as_attachment=True)
    elif ft == 'json':
        json_p = os.path.join(run_dir, "summary", "ambiguity_summary.json")
        if os.path.exists(json_p):
            return send_from_directory(os.path.dirname(json_p), "ambiguity_summary.json", as_attachment=True)
    elif ft in ('chromaticity_png', 'cie_png'):
        png_p = os.path.join(run_dir, "comparisons", "cie_chromaticity_ambiguity_heatmap.png")
        if os.path.exists(png_p):
            return send_from_directory(os.path.dirname(png_p), "cie_chromaticity_ambiguity_heatmap.png", as_attachment=True)

    return jsonify({"error": f"File {file_type} not found for run {run_id}"}), 404


@app.route('/api/chromaticity/data', methods=['GET'])
def get_chromaticity_data():
    run_id = request.args.get('run_id', None)
    runs_dir = os.path.abspath('random_colours_ambiguity_results')
    if not os.path.exists(runs_dir):
        return jsonify({"error": "No batch runs directory found"}), 404

    if not run_id:
        all_runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)
        if not all_runs:
            return jsonify({"error": "No batch runs found"}), 404
        run_id = all_runs[0]

    run_dir = os.path.join(runs_dir, run_id)
    cmp_dir = os.path.join(run_dir, "comparisons")
    json_path = os.path.join(cmp_dir, "chromaticity_ambiguity_data.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return Response(f.read(), mimetype='application/json')
        except Exception:
            pass

    csv_file = os.path.join(run_dir, "summary", "ambiguity_summary.csv")
    if not os.path.exists(csv_file):
        return jsonify({"error": f"Summary data not found for run {run_id}"}), 404

    import csv
    with open(csv_file, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    chrom_analyzer = ChromaticityAnalyzer(artifacts_dir)
    data = chrom_analyzer.export_summary_dataset(records, json_path)
    return jsonify(data)


@app.route('/api/chromaticity/render_png', methods=['GET'])
def export_chromaticity_png():
    run_id = request.args.get('run_id', None)
    metric = request.args.get('metric', 'composite_ambiguity_index')

    runs_dir = os.path.abspath('random_colours_ambiguity_results')
    if not run_id:
        all_runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)
        if not all_runs:
            return jsonify({"error": "No batch runs found"}), 404
        run_id = all_runs[0]

    run_dir = os.path.join(runs_dir, run_id)
    cmp_dir = os.path.join(run_dir, "comparisons")

    if metric == 'composite_ambiguity_index':
        std_png = os.path.join(cmp_dir, "cie_chromaticity_ambiguity_heatmap.png")
        if os.path.exists(std_png):
            return send_from_directory(cmp_dir, "cie_chromaticity_ambiguity_heatmap.png", as_attachment=True)

    csv_file = os.path.join(run_dir, "summary", "ambiguity_summary.csv")
    if not os.path.exists(csv_file):
        return jsonify({"error": f"Summary data not found for run {run_id}"}), 404

    import csv
    with open(csv_file, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    chrom_analyzer = ChromaticityAnalyzer(artifacts_dir)
    target_filename = f"cie_chromaticity_{metric}.png"
    target_path = os.path.join(cmp_dir, target_filename)
    chrom_analyzer.render_chromaticity_diagram(records, target_path, metric_key=metric)

    return send_from_directory(cmp_dir, target_filename, as_attachment=True)


@app.route('/api/chromaticity/export_csv', methods=['GET'])
def export_chromaticity_csv():
    run_id = request.args.get('run_id', None)
    runs_dir = os.path.abspath('random_colours_ambiguity_results')
    if not run_id:
        all_runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)
        if not all_runs:
            return jsonify({"error": "No batch runs found"}), 404
        run_id = all_runs[0]

    run_dir = os.path.join(runs_dir, run_id)
    cmp_dir = os.path.join(run_dir, "comparisons")
    json_path = os.path.join(cmp_dir, "chromaticity_ambiguity_data.json")

    import json
    data = None
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        csv_file = os.path.join(run_dir, "summary", "ambiguity_summary.csv")
        if not os.path.exists(csv_file):
            return jsonify({"error": f"Summary data not found for run {run_id}"}), 404
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            records = list(csv.DictReader(f))
        chrom_analyzer = ChromaticityAnalyzer(artifacts_dir)
        data = chrom_analyzer.export_summary_dataset(records, json_path)

    import io
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "colour_id", "r_uint8", "g_uint8", "b_uint8", "hex_color",
        "cie_x", "cie_y", "inside_locus",
        "composite_ambiguity_index", "spectral_spread_score", "coordinate_spread",
        "mean_pairwise_distance", "p95_pairwise_distance", "max_pairwise_distance",
        "volume_scale_adet", "retained_metamer_samples", "validation_status"
    ])
    for pt in data.get("records", []):
        writer.writerow([
            pt.get("colour_id"), pt.get("r_uint8"), pt.get("g_uint8"), pt.get("b_uint8"), pt.get("hex_color"),
            pt.get("cie_x"), pt.get("cie_y"), True,
            pt.get("composite_ambiguity_index"), pt.get("spectral_spread_score"), pt.get("coordinate_spread"),
            pt.get("mean_pairwise_distance"), pt.get("p95_pairwise_distance"), pt.get("max_pairwise_distance"),
            pt.get("volume_scale_a_det"), pt.get("valid_sample_count"), pt.get("validation_status")
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=cie_xy_ambiguity_100_colours_{run_id}.csv"}
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)

