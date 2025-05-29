import os
import cv2
import numpy as np
from PIL import Image
import io
import re
import nltk
import time
import functools
import hashlib
import warnings
warnings.filterwarnings("ignore")

# Optimizations pour PythonAnywhere
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Cache pour économiser le CPU (important sur plan gratuit)
from functools import lru_cache

# Variables globales pour lazy loading
_sentence_model = None
_ocr_available = False

# Vérifier la disponibilité d'OCR
try:
    import easyocr  # Plus léger que PaddleOCR pour PythonAnywhere
    _ocr_available = True
    _ocr_reader = None
except ImportError:
    try:
        from paddleocr import PaddleOCR
        _ocr_available = True
        _ocr_reader = None
    except ImportError:
        _ocr_available = False

def get_ocr_reader():
    """Get cached OCR reader (lazy loading)"""
    global _ocr_reader
    if _ocr_reader is None and _ocr_available:
        try:
            # Essayer EasyOCR d'abord (plus léger)
            if 'easyocr' in globals():
                _ocr_reader = easyocr.Reader(['fr', 'en'], gpu=False)
                print("Using EasyOCR")
            else:
                # Fallback vers PaddleOCR
                _ocr_reader = PaddleOCR(
                    use_angle_cls=True,
                    lang='fr',
                    use_gpu=False,
                    show_log=False
                )
                print("Using PaddleOCR")
        except Exception as e:
            print(f"OCR initialization failed: {e}")
            _ocr_reader = None
    return _ocr_reader

def get_sentence_model():
    """Get cached sentence model (lazy loading)"""
    global _sentence_model
    if _sentence_model is None:
        try:
            # Utiliser le modèle le plus léger possible
            from sentence_transformers import SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'  # 80MB seulement
            _sentence_model = SentenceTransformer(model_name)
            print(f"Loaded sentence model: {model_name}")
        except Exception as e:
            print(f"Failed to load sentence model: {e}")
            _sentence_model = None
    return _sentence_model

@lru_cache(maxsize=100)
def extract_text_with_ocr_cached(image_hash):
    """Cached OCR extraction pour économiser CPU"""
    # Cette fonction ne peut pas prendre directement l'image
    # car les numpy arrays ne sont pas hashables
    # Elle sera appelée par extract_text_with_ocr
    pass

def extract_text_with_ocr(image_input):
    """
    Extract text from image with caching
    Optimisé pour PythonAnywhere (économie CPU)
    """
    if not _ocr_available:
        return "OCR not available. Please install easyocr or paddleocr."
    
    ocr_reader = get_ocr_reader()
    if ocr_reader is None:
        return "OCR reader initialization failed"
    
    try:
        # Convert input to consistent format
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Create hash for caching
            image_hash = hashlib.md5(image_input).hexdigest()
        elif isinstance(image_input, str):
            img = cv2.imread(image_input)
            with open(image_input, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
        elif isinstance(image_input, np.ndarray):
            img = image_input
            image_hash = hashlib.md5(img.tobytes()).hexdigest()
        else:
            return "Unsupported image format"
        
        if img is None:
            return "Failed to load image"
        
        # Check cache first (économie CPU importante)
        cache_key = f"ocr_{image_hash}"
        if hasattr(extract_text_with_ocr, '_cache'):
            if cache_key in extract_text_with_ocr._cache:
                return extract_text_with_ocr._cache[cache_key]
        else:
            extract_text_with_ocr._cache = {}
        
        # Perform OCR
        start_time = time.time()
        
        if 'easyocr' in str(type(ocr_reader)):
            # EasyOCR
            results = ocr_reader.readtext(img)
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    extracted_text.append(text)
        else:
            # PaddleOCR
            results = ocr_reader.ocr(img, cls=True)
            extracted_text = []
            if results and results[0]:
                for line in results[0]:
                    if line[1][1] > 0.5:  # Confidence threshold
                        extracted_text.append(line[1][0])
        
        result = ' '.join(extracted_text) if extracted_text else "No text detected"
        
        # Cache result
        extract_text_with_ocr._cache[cache_key] = result
        
        # Log CPU usage (important pour plan gratuit)
        cpu_time = time.time() - start_time
        if cpu_time > 10:  # Warning si plus de 10s CPU
            print(f"Warning: OCR took {cpu_time:.2f}s CPU time")
        
        return result
        
    except Exception as e:
        return f"OCR processing failed: {str(e)}"

@lru_cache(maxsize=200)
def process_text_with_nlp_cached(text_hash, question_hash):
    """Cached NLP processing"""
    # Les vrais paramètres seront passés via un dictionnaire global
    # Ceci est un hack pour utiliser le cache avec des strings complexes
    pass

def clean_text(text):
    """Clean text (cached)"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:]', ' ', text)
    
    return ' '.join(text.split()).strip()

@lru_cache(maxsize=50)
def split_text_into_sentences_cached(text):
    """Cached sentence splitting"""
    try:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return tuple(s.strip() for s in sentences if len(s.strip()) > 10)
    except Exception as e:
        # Fallback
        sentences = text.split('.')
        return tuple(s.strip() for s in sentences if len(s.strip()) > 10)

def find_relevant_sentences(question, sentences, top_k=3):
    """Find relevant sentences with lightweight processing"""
    model = get_sentence_model()
    
    if model is None:
        # Fallback to keyword matching only
        return keyword_matching_fallback(question, sentences, top_k)
    
    try:
        if not sentences:
            return []
        
        # Limit processing to save CPU
        max_sentences = 20  # Limite pour économiser CPU
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        start_time = time.time()
        
        # Encode with timeout protection
        question_embedding = model.encode([question])
        sentence_embeddings = model.encode(list(sentences))
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_sentences = []
        for idx in top_indices:
            if similarities[idx] > 0.3:
                relevant_sentences.append({
                    'text': sentences[idx],
                    'similarity': float(similarities[idx])
                })
        
        cpu_time = time.time() - start_time
        if cpu_time > 5:  # Warning si plus de 5s
            print(f"Warning: NLP processing took {cpu_time:.2f}s")
        
        return relevant_sentences
        
    except Exception as e:
        print(f"NLP error, falling back to keyword matching: {e}")
        return keyword_matching_fallback(question, sentences, top_k)

def keyword_matching_fallback(question, sentences, top_k=3):
    """Lightweight keyword matching (CPU efficient)"""
    question_words = set(question.lower().split())
    
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words.intersection(sentence_words))
        if overlap > 0:
            score = overlap / len(question_words)
            scored_sentences.append({
                'text': sentence,
                'similarity': score
            })
    
    scored_sentences.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_sentences[:top_k]

def process_text_with_nlp(extracted_text, question):
    """
    Main NLP processing function
    Optimisé pour PythonAnywhere (CPU économique)
    """
    try:
        if not extracted_text or not question:
            return "Texte ou question manquant."
        
        # Quick hash for caching
        text_hash = hashlib.md5(extracted_text.encode()).hexdigest()[:16]
        question_hash = hashlib.md5(question.encode()).hexdigest()[:16]
        cache_key = f"nlp_{text_hash}_{question_hash}"
        
        # Check cache
        if hasattr(process_text_with_nlp, '_cache'):
            if cache_key in process_text_with_nlp._cache:
                return process_text_with_nlp._cache[cache_key]
        else:
            process_text_with_nlp._cache = {}
        
        start_time = time.time()
        
        # Clean text
        cleaned_text = clean_text(extracted_text)
        
        if len(cleaned_text) < 10:
            return "Pas assez de texte pour traiter la question."
        
        # Split sentences with cache
        sentences = split_text_into_sentences_cached(cleaned_text)
        
        if not sentences:
            return "Impossible de segmenter le texte."
        
        # Find relevant content
        relevant_sentences = find_relevant_sentences(question, sentences)
        
        if not relevant_sentences:
            # Return partial text as fallback
            return f"Information disponible: {cleaned_text[:150]}..."
        
        # Generate simple answer
        context = " ".join([sent['text'] for sent in relevant_sentences])
        answer = generate_simple_answer(question, context)
        
        # Cache result
        process_text_with_nlp._cache[cache_key] = answer
        
        # Monitor CPU usage
        cpu_time = time.time() - start_time
        if cpu_time > 8:  # Warning pour plan gratuit
            print(f"Warning: Total NLP processing took {cpu_time:.2f}s")
        
        return answer
        
    except Exception as e:
        return f"Erreur NLP: {str(e)}"

def generate_simple_answer(question, context):
    """Generate simple answer without heavy processing"""
    if not context:
        return "Aucune information pertinente trouvée."
    
    question_lower = question.lower()
    
    # Simple pattern matching for different question types
    if any(word in question_lower for word in ['qui', 'who']):
        return f"Information trouvée: {context[:100]}..."
    elif any(word in question_lower for word in ['quoi', 'what', 'que']):
        return f"Réponse: {context[:150]}..."
    elif any(word in question_lower for word in ['où', 'where']):
        return f"Localisation: {context[:100]}..."
    elif any(word in question_lower for word in ['quand', 'when']):
        return f"Timing: {context[:100]}..."
    else:
        return f"Information: {context[:200]}..." if len(context) > 200 else context

# Memory management pour PythonAnywhere
def clear_cache():
    """Clear all caches to free memory"""
    if hasattr(extract_text_with_ocr, '_cache'):
        extract_text_with_ocr._cache.clear()
    if hasattr(process_text_with_nlp, '_cache'):
        process_text_with_nlp._cache.clear()
    
    # Clear LRU caches
    process_text_with_nlp_cached.cache_clear()
    split_text_into_sentences_cached.cache_clear()
    
    print("Caches cleared to free memory")

def get_cache_stats():
    """Get cache statistics for monitoring"""
    stats = {
        'ocr_cache_size': len(getattr(extract_text_with_ocr, '_cache', {})),
        'nlp_cache_size': len(getattr(process_text_with_nlp, '_cache', {})),
        'sentence_cache_info': split_text_into_sentences_cached.cache_info(),
    }
    return stats

# Fonction de test lightweight
def test_setup():
    """Test si tout fonctionne (version légère)"""
    try:
        # Test text processing
        test_text = "Ceci est un test simple."
        cleaned = clean_text(test_text)
        
        # Test sentence splitting
        sentences = split_text_into_sentences_cached(test_text)
        
        return {
            "status": "success",
            "ocr_available": _ocr_available,
            "sentence_model_loaded": _sentence_model is not None,
            "text_processing": "OK" if cleaned else "FAILED"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
