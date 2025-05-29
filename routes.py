import os
import json
import requests
import cv2
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
import base64
import io
from PIL import Image
import numpy as np

# Flask imports
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, session, jsonify, send_from_directory, abort
)
from werkzeug.utils import secure_filename

# Cloud storage
import cloudinary
import cloudinary.uploader
import cloudinary.api

# NLP module (your existing one, slightly modified)
from nlp_module import process_text_with_nlp, extract_text_with_ocr

# ==========================================
# CLOUD CONFIGURATION
# ==========================================

def init_cloudinary():
    """Initialize Cloudinary with environment variables"""
    cloudinary.config(
        cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
        api_key=os.getenv('CLOUDINARY_API_KEY'),
        api_secret=os.getenv('CLOUDINARY_API_SECRET')
    )

# ==========================================
# HUGGING FACE API FUNCTIONS
# ==========================================

def predict_with_huggingface_api(image_path_or_bytes):
    """Call your Hugging Face model API"""
    API_URL = "https://api-inference.huggingface.co/models/DJERI-ALASSANI/MINESMARTWEBSCRAPERCV"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    try:
        # Handle both file path and bytes
        if isinstance(image_path_or_bytes, str):
            with open(image_path_or_bytes, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = image_path_or_bytes
        
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"HF API Error: {response.status_code} - {response.text}")
            return {"error": f"API error: {response.status_code}"}
            
    except Exception as e:
        print(f"Error calling HF API: {str(e)}")
        return {"error": str(e)}

# ==========================================
# CLOUD STORAGE FUNCTIONS
# ==========================================

def upload_image_to_cloudinary(image_data, filename, folder="scraper_images"):
    """Upload image to Cloudinary"""
    try:
        if isinstance(image_data, str):  # file path
            upload_result = cloudinary.uploader.upload(
                image_data,
                public_id=f"{folder}/{filename}",
                overwrite=True
            )
        elif isinstance(image_data, np.ndarray):  # OpenCV image
            _, buffer = cv2.imencode('.jpg', image_data)
            image_bytes = buffer.tobytes()
            upload_result = cloudinary.uploader.upload(
                image_bytes,
                public_id=f"{folder}/{filename}",
                overwrite=True
            )
        else:  # bytes
            upload_result = cloudinary.uploader.upload(
                image_data,
                public_id=f"{folder}/{filename}",
                overwrite=True
            )
        
        return upload_result['secure_url']
    except Exception as e:
        print(f"Cloudinary upload error: {str(e)}")
        return None

def save_json_to_cloudinary(json_data, filename, folder="scraper_data"):
    """Save JSON data to Cloudinary as raw file"""
    try:
        json_string = json.dumps(json_data, indent=2)
        upload_result = cloudinary.uploader.upload(
            json_string,
            public_id=f"{folder}/{filename}",
            resource_type="raw",
            overwrite=True
        )
        return upload_result['secure_url']
    except Exception as e:
        print(f"JSON upload error: {str(e)}")
        return None

# ==========================================
# DATA MANAGEMENT (CLOUD-BASED)
# ==========================================

# In-memory storage for session data (replace with database in production)
CAPTURES_DATA = {}
PROCESSED_LINKS = set()

def save_capture_info(capture_id, info):
    """Save capture info (in production, use database)"""
    CAPTURES_DATA[capture_id] = info

def find_capture_by_id(capture_id):
    """Find capture by ID"""
    return CAPTURES_DATA.get(capture_id)

def get_all_captures():
    """Get all captures"""
    return list(CAPTURES_DATA.values())

# ==========================================
# FLASK ROUTES - ADAPTED FOR CLOUD
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user_type = request.form.get("user_type")
        
        if user_type == "admin":
            if email == "djeryala@gmail.com" and password == "DJERI":
                session["user_type"] = "admin"
                session["logged_in"] = True
                return redirect(url_for("admin_dashboard"))
            else:
                flash("Identifiants administrateur incorrects", "danger")
        else:
            session["user_type"] = "user"
            session["logged_in"] = True
            return redirect(url_for("user_dashboard"))
    
    return render_template("login.html")

@app.route("/user/dashboard")
def user_dashboard():
    if not session.get("logged_in") or session.get("user_type") != "user":
        return redirect(url_for("login"))
    return render_template("user_dashboard.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("logged_in") or session.get("user_type") != "admin":
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")

@app.route("/user/capture", methods=["GET", "POST"])
def user_capture():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    if request.method == "POST":
        url = request.form.get("url")
        if not url:
            flash("Veuillez entrer une URL", "danger")
            return render_template("user_capture.html")
        
        try:
            # Add URL to processed links
            PROCESSED_LINKS.add(url)
            
            # Capture screenshot with Playwright (your existing logic)
            capture_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Take screenshot (adapt your existing playwright code)
            screenshot_bytes = take_screenshot_playwright(url)
            if not screenshot_bytes:
                flash("Erreur lors de la capture d'écran", "danger")
                return render_template("user_capture.html")
            
            # Upload to Cloudinary
            filename = f"capture_{capture_id}.jpg"
            image_url = upload_image_to_cloudinary(screenshot_bytes, filename, "captures")
            
            if not image_url:
                flash("Erreur lors de la sauvegarde", "danger")
                return render_template("user_capture.html")
            
            # Save capture info
            capture_info = {
                "id": capture_id,
                "url": url,
                "filename": filename,
                "image_url": image_url,
                "timestamp": timestamp
            }
            save_capture_info(capture_id, capture_info)
            
            return redirect(url_for("user_view_capture", capture_id=capture_id))
            
        except Exception as e:
            flash(f"Erreur: {str(e)}", "danger")
            return render_template("user_capture.html")
    
    return render_template("user_capture.html")

@app.route("/user/view_capture/<capture_id>")
def user_view_capture(capture_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    capture_info = find_capture_by_id(capture_id)
    if not capture_info:
        flash("Capture non trouvée", "danger")
        return redirect(url_for("user_capture"))
    
    return render_template("user_view_capture.html", capture_info=capture_info)

@app.route("/user/ask_question/<capture_id>", methods=["GET", "POST"])
def user_ask_question(capture_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    capture_info = find_capture_by_id(capture_id)
    if not capture_info:
        flash("Capture non trouvée", "danger")
        return redirect(url_for("user_capture"))
    
    if request.method == "POST":
        question = request.form.get("question")
        if not question:
            flash("Veuillez poser une question", "danger")
            return render_template("user_ask_question.html", capture_info=capture_info)
        
        try:
            # Download image from Cloudinary for OCR
            image_response = requests.get(capture_info["image_url"])
            if image_response.status_code != 200:
                flash("Erreur lors du téléchargement de l'image", "danger")
                return render_template("user_ask_question.html", capture_info=capture_info)
            
            # Process with your NLP module
            image_bytes = image_response.content
            extracted_text = extract_text_with_ocr(image_bytes)
            
            if not extracted_text:
                flash("Aucun texte extrait de l'image", "warning")
                return render_template("user_ask_question.html", capture_info=capture_info)
            
            # Get answer using NLP
            answer = process_text_with_nlp(extracted_text, question)
            
            return render_template("user_ask_question.html", 
                                 capture_info=capture_info, 
                                 question=question, 
                                 answer=answer)
            
        except Exception as e:
            flash(f"Erreur lors du traitement: {str(e)}", "danger")
            return render_template("user_ask_question.html", capture_info=capture_info)
    
    return render_template("user_ask_question.html", capture_info=capture_info)

@app.route("/user/annotate_model/<capture_id>")
def user_annotate_model(capture_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    capture_info = find_capture_by_id(capture_id)
    if not capture_info:
        flash("Capture non trouvée", "danger")
        return redirect(url_for("user_capture"))
    
    try:
        # Download image from Cloudinary
        image_response = requests.get(capture_info["image_url"])
        if image_response.status_code != 200:
            flash("Erreur lors du téléchargement de l'image", "danger")
            return redirect(url_for("user_view_capture", capture_id=capture_id))
        
        # Call Hugging Face API for predictions
        predictions = predict_with_huggingface_api(image_response.content)
        
        if not predictions or 'error' in predictions:
            flash("Erreur lors de la prédiction", "danger")
            return redirect(url_for("user_view_capture", capture_id=capture_id))
        
        # Process predictions and create annotated image
        annotated_image_url, detected_boxes = process_predictions_and_annotate(
            image_response.content, predictions, capture_id
        )
        
        if not annotated_image_url:
            flash("Erreur lors de l'annotation", "danger")
            return redirect(url_for("user_view_capture", capture_id=capture_id))
        
        return render_template("user_annotate_model.html",
                             capture_info=capture_info,
                             annotated_image_url=annotated_image_url,
                             detected_boxes=detected_boxes)
        
    except Exception as e:
        flash(f"Erreur: {str(e)}", "danger")
        return redirect(url_for("user_view_capture", capture_id=capture_id))

def process_predictions_and_annotate(image_bytes, predictions, capture_id):
    """Process HF API predictions and create annotated image"""
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        annotated = img.copy()
        
        # Class configuration (same as your original)
        thing_classes = [
            "advertisement", "chaine", "commentaire", "description", "header", "footer", 
            "left sidebar", "logo", "likes", "media", "pop up", "recommendations", 
            "right sidebar", "suggestions", "title", "vues", "none access", "other"
        ]
        
        unique_class_limit = {
            "footer", "header", "chaine", "commentaire", "description",
            "left sidebar", "likes", "recommendations", "vues", "title", "right sidebar"
        }
        
        to_ignore_classes = {"pop up", "logo", "other", "none access", "suggestions"}
        
        CLASS_COLORS = {
            "advertisement": (255, 0, 0), "chaine": (0, 255, 0), "commentaire": (0, 0, 255),
            "description": (255, 255, 0), "header": (255, 0, 255), "footer": (0, 255, 255),
            "left sidebar": (128, 0, 0), "logo": (0, 128, 0), "likes": (0, 0, 128),
            "media": (128, 128, 0), "pop up": (128, 0, 128), "recommendations": (0, 128, 128),
            "right sidebar": (64, 0, 0), "suggestions": (0, 64, 0), "title": (0, 0, 64),
            "vues": (64, 64, 0), "none access": (64, 0, 64), "other": (0, 64, 64)
        }
        
        detected_boxes = []
        annotations = []
        kept = {}
        
        # Process predictions (adapt based on your HF API response format)
        for i, pred in enumerate(predictions):
            # Adapt this based on your actual HF API response format
            class_name = pred.get("label", "unknown")
            score = float(pred.get("score", 0.0))
            box_data = pred.get("box", {})
            
            # Convert coordinates (adapt based on your API format)
            x1 = int(box_data.get("xmin", 0))
            y1 = int(box_data.get("ymin", 0))
            x2 = int(box_data.get("xmax", 0))
            y2 = int(box_data.get("ymax", 0))
            
            if class_name in to_ignore_classes:
                continue
            
            # Apply unique class logic
            if class_name in unique_class_limit:
                if class_name in kept:
                    if score > kept[class_name]["score"]:
                        kept[class_name] = {"index": i, "score": score, "coords": [x1, y1, x2, y2]}
                    continue
                else:
                    kept[class_name] = {"index": i, "score": score, "coords": [x1, y1, x2, y2]}
            else:
                kept[f"{class_name}_{i}"] = {"index": i, "score": score, "coords": [x1, y1, x2, y2]}
        
        # Draw annotations
        for key, val in kept.items():
            i = val["index"]
            score = val["score"]
            x1, y1, x2, y2 = val["coords"]
            
            class_name = key.split('_')[0] if '_' in key else key
            class_id = thing_classes.index(class_name) if class_name in thing_classes else 0
            
            color = CLASS_COLORS.get(class_name, (0, 255, 0))
            
            # Draw rectangle and text
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{class_name} {score:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            detected_boxes.append({
                "id": f"box{i+1}",
                "class": class_name,
                "coords": [x1, y1, x2, y2],
                "score": score
            })
            
            annotations.append({
                "id": i + 1,
                "image_id": capture_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score
            })
        
        # Upload annotated image to Cloudinary
        annotated_filename = f"annotated_{capture_id}.jpg"
        annotated_image_url = upload_image_to_cloudinary(annotated, annotated_filename, "annotated")
        
        # Save COCO JSON to Cloudinary
        coco_data = {"annotations": annotations}
        json_filename = f"predictions_{capture_id}.json"
        save_json_to_cloudinary(coco_data, json_filename, "predictions")
        
        return annotated_image_url, detected_boxes
        
    except Exception as e:
        print(f"Error processing predictions: {str(e)}")
        return None, []

# ==========================================
# ADMIN ROUTES
# ==========================================

@app.route("/admin/view_links")
def admin_view_links():
    if not session.get("logged_in") or session.get("user_type") != "admin":
        return redirect(url_for("login"))
    
    return render_template("admin_view_links.html", links=list(PROCESSED_LINKS))

@app.route("/admin/human_validated")
def admin_human_validated():
    if not session.get("logged_in") or session.get("user_type") != "admin":
        return redirect(url_for("login"))
    
    # Get human validated images from Cloudinary
    try:
        result = cloudinary.api.resources(
            type="upload",
            prefix="scraper_images/human_validated/",
            max_results=100
        )
        images = result.get('resources', [])
        return render_template("admin_human_validated.html", images=images)
    except Exception as e:
        flash(f"Erreur: {str(e)}", "danger")
        return render_template("admin_human_validated.html", images=[])

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def take_screenshot_playwright(url):
    """Take screenshot using Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            screenshot_bytes = page.screenshot()
            browser.close()
            
            return screenshot_bytes
    except Exception as e:
        print(f"Screenshot error: {str(e)}")
        return None

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ==========================================
# INITIALIZATION
# ==========================================

if __name__ == "__main__":
    init_cloudinary()
    app.run(debug=False)  # Set to False for production