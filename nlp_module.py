import os
import cv2
import numpy as np
from PIL import Image
import io
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# OCR imports
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Warning: PaddleOCR not available")

# Global variables for caching (memory optimization)
_ocr_instance = None
_sentence_model = None

def get_ocr_instance():
    """Get cached OCR instance"""
    global _ocr_instance
    if _ocr_instance is None and PADDLEOCR_AVAILABLE:
        try:
            _ocr_instance = PaddleOCR(
                use_angle_cls=True, 
                lang='fr',  # French + English
                use_gpu=False,  # Force CPU for cloud deployment
                show_log=False
            )
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            _ocr_instance = None
    return _ocr_instance

def get_sentence_model():
    """Get cached sentence transformer model"""
    global _sentence_model
    if _sentence_model is None:
        try:
            # Use a lighter model for cloud deployment
            model_name = os.getenv('SENTENCE_MODEL', 'all-MiniLM-L6-v2')
            _sentence_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading sentence model: {e}")
            _sentence_model = None
    return _sentence_model

def extract_text_with_ocr(image_input):
    """
    Extract text from image using PaddleOCR
    image_input can be: file path, bytes, or numpy array
    """
    if not PADDLEOCR_AVAILABLE:
        return "OCR not available in this environment"
    
    ocr = get_ocr_instance()
    if ocr is None:
        return "OCR initialization failed"
    
    try:
        # Handle different input types
        if isinstance(image_input, bytes):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, str):
            # File path
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            # Already a numpy array
            img = image_input
        else:
            return "Unsupported image format"
        
        if img is None:
            return "Failed to load image"
        
        # Perform OCR
        result = ocr.ocr(img, cls=True)
        
        if not result or not result[0]:
            return "No text detected"
        
        # Extract text from results
        extracted_text = []
        for line in result[0]:
            if line[1][1] > 0.5:  # Confidence threshold
                extracted_text.append(line[1][0])
        
        return ' '.join(extracted_text)
        
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return f"OCR processing failed: {str(e)}"

def clean_text(text):
    """Clean and preprocess text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:]', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

def split_text_into_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        # Download punkt if not available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    except Exception as e:
        print(f"Sentence tokenization error: {e}")
        # Fallback: simple split by periods
        sentences = text.split('.')
        return [s.strip() for s in sentences if len(s.strip()) > 10]

def find_relevant_sentences(question, sentences, top_k=3):
    """Find most relevant sentences using semantic similarity"""
    model = get_sentence_model()
    if model is None:
        # Fallback: simple keyword matching
        return keyword_matching_fallback(question, sentences, top_k)
    
    try:
        if not sentences:
            return []
        
        # Encode question and sentences
        question_embedding = model.encode([question])
        sentence_embeddings = model.encode(sentences)
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
        
        # Get top-k most similar sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_sentences = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                relevant_sentences.append({
                    'text': sentences[idx],
                    'similarity': float(similarities[idx])
                })
        
        return relevant_sentences
        
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return keyword_matching_fallback(question, sentences, top_k)

def keyword_matching_fallback(question, sentences, top_k=3):
    """Fallback method using keyword matching"""
    question_words = set(question.lower().split())
    
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        # Calculate word overlap
        overlap = len(question_words.intersection(sentence_words))
        if overlap > 0:
            score = overlap / len(question_words)
            scored_sentences.append({
                'text': sentence,
                'similarity': score
            })
    
    # Sort by score and return top-k
    scored_sentences.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_sentences[:top_k]

def generate_answer_from_context(question, relevant_sentences):
    """Generate answer from relevant context"""
    if not relevant_sentences:
        return "Je n'ai pas trouvé d'informations pertinentes dans l'image pour répondre à votre question."
    
    # Combine relevant sentences
    context = " ".join([sent['text'] for sent in relevant_sentences])
    
    # Simple answer generation based on question type
    question_lower = question.lower()
    
    # Question type detection
    if any(word in question_lower for word in ['qui', 'who']):
        # Who questions
        answer = extract_person_info(context)
    elif any(word in question_lower for word in ['quoi', 'what', 'que']):
        # What questions
        answer = extract_main_info(context)
    elif any(word in question_lower for word in ['où', 'where']):
        # Where questions
        answer = extract_location_info(context)
    elif any(word in question_lower for word in ['quand', 'when']):
        # When questions
        answer = extract_time_info(context)
    elif any(word in question_lower for word in ['combien', 'how many', 'how much']):
        # Count/quantity questions
        answer = extract_quantity_info(context)
    elif any(word in question_lower for word in ['comment', 'how']):
        # How questions
        answer = extract_method_info(context)
    elif any(word in question_lower for word in ['pourquoi', 'why']):
        # Why questions
        answer = extract_reason_info(context)
    else:
        # General questions
        answer = context[:200] + "..." if len(context) > 200 else context
    
    return answer if answer else "Informations trouvées: " + context[:150] + "..."

def extract_person_info(context):
    """Extract person-related information"""
    # Simple pattern matching for names and titles
    import re
    
    # Look for common name patterns
    name_patterns = [
        r'[A-Z][a-z]+ [A-Z][a-z]+',  # First Last
        r'M\. [A-Z][a-z]+',          # Mr. Name
        r'Mme [A-Z][a-z]+',          # Mrs. Name
        r'Dr\. [A-Z][a-z]+',         # Dr. Name
    ]
    
    names = []
    for pattern in name_patterns:
        matches = re.findall(pattern, context)
        names.extend(matches)
    
    if names:
        return f"Personnes mentionnées: {', '.join(set(names))}"
    
    return context[:100] + "..."

def extract_main_info(context):
    """Extract main information for 'what' questions"""
    # Look for key phrases and concepts
    sentences = context.split('.')
    if sentences:
        # Return the first substantial sentence
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip()
    
    return context[:150] + "..."

def extract_location_info(context):
    """Extract location information"""
    import re
    
    # Look for location patterns
    location_patterns = [
        r'à [A-Z][a-z]+',        # à Paris
        r'en [A-Z][a-z]+',       # en France
        r'dans [A-Z][a-z]+',     # dans Lyon
        r'[A-Z][a-z]+, [A-Z][a-z]+',  # City, Country
    ]
    
    locations = []
    for pattern in location_patterns:
        matches = re.findall(pattern, context)
        locations.extend(matches)
    
    if locations:
        return f"Lieux mentionnés: {', '.join(set(locations))}"
    
    return context[:100] + "..."

def extract_time_info(context):
    """Extract time/date information"""
    import re
    
    # Look for time patterns
    time_patterns = [
        r'\b\d{1,2}h\d{2}\b',           # 14h30
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',   # 12/03/2024
        r'\b\d{4}-\d{2}-\d{2}\b',       # 2024-03-12
        r'\blundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche\b',  # Days
        r'\bjanvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre\b',  # Months
    ]
    
    times = []
    for pattern in time_patterns:
        matches = re.findall(pattern, context, re.IGNORECASE)
        times.extend(matches)
    
    if times:
        return f"Informations temporelles: {', '.join(set(times))}"
    
    return context[:100] + "..."

def extract_quantity_info(context):
    """Extract quantity/number information"""
    import re
    
    # Look for numbers and quantities
    number_patterns = [
        r'\b\d+\b',                     # Plain numbers
        r'\b\d+[.,]\d+\b',              # Decimals
        r'\b\d+%\b',                    # Percentages
        r'\b\d+€\b',                    # Euros
        r'\b\d+\$\b',                   # Dollars
    ]
    
    numbers = []
    for pattern in number_patterns:
        matches = re.findall(pattern, context)
        numbers.extend(matches)
    
    if numbers:
        return f"Valeurs numériques trouvées: {', '.join(set(numbers))}"
    
    return context[:100] + "..."

def extract_method_info(context):
    """Extract method/process information"""
    # Look for action words and processes
    method_keywords = ['étapes', 'méthode', 'procédure', 'comment', 'pour', 'afin']
    
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in method_keywords):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return ". ".join(relevant_sentences[:2])
    
    return context[:150] + "..."

def extract_reason_info(context):
    """Extract reasoning/explanation information"""
    # Look for explanation keywords
    reason_keywords = ['parce que', 'car', 'puisque', 'grâce à', 'à cause de', 'raison']
    
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in reason_keywords):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return ". ".join(relevant_sentences[:2])
    
    return context[:150] + "..."

def process_text_with_nlp(extracted_text, question):
    """
    Main function to process text and answer questions
    """
    try:
        if not extracted_text or not question:
            return "Texte ou question manquant."
        
        # Clean the extracted text
        cleaned_text = clean_text(extracted_text)
        
        if len(cleaned_text) < 10:
            return "Pas assez de texte extrait pour répondre à la question."
        
        # Split into sentences
        sentences = split_text_into_sentences(cleaned_text)
        
        if not sentences:
            return "Impossible de segmenter le texte en phrases."
        
        # Find relevant sentences
        relevant_sentences = find_relevant_sentences(question, sentences)
        
        if not relevant_sentences:
            return f"Aucune information pertinente trouvée pour: '{question}'. Texte disponible: {cleaned_text[:100]}..."
        
        # Generate answer
        answer = generate_answer_from_context(question, relevant_sentences)
        
        return answer
        
    except Exception as e:
        print(f"NLP Processing Error: {str(e)}")
        return f"Erreur lors du traitement: {str(e)}"

def analyze_text_content(text):
    """
    Analyze text content and extract key information
    """
    try:
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            return {"error": "No text to analyze"}
        
        # Basic text analysis
        analysis = {
            "text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "sentence_count": len(split_text_into_sentences(cleaned_text)),
            "languages_detected": detect_languages(cleaned_text),
            "key_entities": extract_key_entities(cleaned_text),
            "summary": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def detect_languages(text):
    """Simple language detection"""
    french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'est', 'sont', 'avec', 'pour', 'dans', 'sur']
    english_words = ['the', 'and', 'or', 'is', 'are', 'with', 'for', 'in', 'on', 'at', 'to', 'of']
    
    text_lower = text.lower()
    
    french_count = sum(1 for word in french_words if word in text_lower)
    english_count = sum(1 for word in english_words if word in text_lower)
    
    languages = []
    if french_count > 0:
        languages.append('French')
    if english_count > 0:
        languages.append('English')
    
    return languages if languages else ['Unknown']

def extract_key_entities(text):
    """Extract key entities (simplified)"""
    import re
    
    entities = {
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
        'phones': re.findall(r'(\+33|0)[1-9](?:[0-9]{8})', text),
        'dates': re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text),
    }
    
    # Remove empty lists
    entities = {k: v for k, v in entities.items() if v}
    
    return entities

# Test functions for debugging
def test_ocr_functionality():
    """Test OCR functionality"""
    try:
        # Create a simple test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, 'Test OCR', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        result = extract_text_with_ocr(test_image)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def test_nlp_functionality():
    """Test NLP functionality"""
    try:
        test_text = "Bonjour, ceci est un test de la fonctionnalité NLP. Il y a plusieurs phrases ici."
        test_question = "Qu'est-ce que c'est?"
        
        result = process_text_with_nlp(test_text, test_question)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Memory optimization for cloud deployment
def clear_model_cache():
    """Clear model cache to free memory"""
    global _ocr_instance, _sentence_model
    _ocr_instance = None
    _sentence_model = None
    
    # Force garbage collection
    import gc
    gc.collect()

# Environment-specific optimizations
def optimize_for_cloud():
    """Apply cloud-specific optimizations"""
    # Set environment variables for performance
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Disable unnecessary warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    print("Cloud optimizations applied")