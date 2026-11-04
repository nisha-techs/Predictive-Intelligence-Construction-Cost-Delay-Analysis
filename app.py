import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import database helpers and PredictionService
try:
    from src.prediction_service import PredictionService
    from src.database import list_projects, get_project, get_monitoring_history, update_project_completion
except ModuleNotFoundError:
    from prediction_service import PredictionService
    from database import list_projects, get_project, get_monitoring_history, update_project_completion

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for future frontend integration

@app.route('/')
def index():
    return render_template('index.html')

# Initialize PredictionService (Loads ML models and DB once at startup)
service = PredictionService()

# Custom JSON Error Handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": "Invalid input",
        "message": getattr(error, 'description', 'Invalid input data provided.')
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Not found",
        "message": getattr(error, 'description', 'Requested resource was not found.')
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal error",
        "message": "An unexpected server error occurred while processing prediction."
    }), 500

# 1. Health Check Endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "success": True,
        "status": "healthy",
        "models_loaded": True,
        "framework": "Flask 3.1.3",
        "timestamp": app.config.get('START_TIME', '')
    }), 200

# 2. Stage 1 — Pre-Construction Cost Estimation API
@app.route('/api/stage1/predict', methods=['POST'])
def predict_stage1():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid input",
            "message": "Request body must be a valid JSON object."
        }), 400
        
    try:
        result = service.predict_stage1_cost(data)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": "Invalid input",
            "message": str(ve)
        }), 400
    except Exception as e:
        app.logger.error(f"Stage 1 prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": "Unable to generate pre-construction cost prediction."
        }), 500

# 3. List All Stored Projects API
@app.route('/api/projects', methods=['GET'])
def get_all_projects():
    try:
        projects = list_projects()
        return jsonify({
            "success": True,
            "count": len(projects),
            "projects": projects
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Retrieval error",
            "message": f"Unable to list projects: {str(e)}"
        }), 500

# 4. Get Single Project API
@app.route('/api/projects/<project_id>', methods=['GET'])
def get_single_project(project_id):
    try:
        project = get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": "Project not found",
                "message": f"No project exists with Project_ID {project_id}."
            }), 404
            
        return jsonify({
            "success": True,
            "project": project
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Retrieval error",
            "message": f"Unable to retrieve project details: {str(e)}"
        }), 500

# 5. Stage 2 — Ongoing Monitoring Prediction API
@app.route('/api/stage2/predict', methods=['POST'])
def predict_stage2():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid input",
            "message": "Request body must be a valid JSON object."
        }), 400
        
    try:
        result = service.predict_stage2_ongoing(data)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": "Invalid input",
            "message": str(ve)
        }), 400
    except KeyError as ke:
        return jsonify({
            "success": False,
            "error": "Project not found",
            "message": str(ke).strip("'")
        }), 404
    except Exception as e:
        app.logger.error(f"Stage 2 prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": "Unable to generate ongoing monitoring prediction."
        }), 500

# 6. Get Project Monitoring History API
@app.route('/api/projects/<project_id>/monitoring', methods=['GET'])
def get_project_monitoring(project_id):
    try:
        project = get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": "Project not found",
                "message": f"No project exists with Project_ID {project_id}."
            }), 404
            
        history = get_monitoring_history(project_id)
        return jsonify({
            "success": True,
            "project_id": project_id,
            "count": len(history),
            "monitoring_history": history
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Retrieval error",
            "message": f"Unable to retrieve project monitoring history: {str(e)}"
        }), 500

# 7. Model Performance Comparison Summary API
@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    try:
        perf = service.get_model_performance_summary()
        return jsonify(perf), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Retrieval error",
            "message": f"Unable to fetch model performance summary: {str(e)}"
        }), 500

# 8. Update Project Monitoring Completion Status API
@app.route('/api/projects/<project_id>/complete', methods=['POST'])
def complete_project_monitoring(project_id):
    try:
        project = get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": "Project not found",
                "message": f"No project exists with Project_ID {project_id}."
            }), 404
            
        data = request.get_json(silent=True) or {}
        is_comp = 1 if data.get('is_completed', 1) in (1, True, '1', 'true', 'True') else 0
        update_project_completion(project_id, is_comp)
        
        updated_proj = get_project(project_id)
        return jsonify({
            "success": True,
            "message": f"Project {project_id} monitoring completion status updated.",
            "project": updated_proj
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Update error",
            "message": f"Unable to update project completion status: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n==================================================")
    print(f"Starting Predictive Intelligence Backend API Server")
    print(f"URL: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
