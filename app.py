# app.py - Main Flask Application - Complete Version for Railway Deployment
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import re  # ← ADD THIS
import json
import random
import numpy as np  # ← ADD THIS
from datetime import datetime
from collections import defaultdict  # ← ADD THIS

# Document processing imports
import PyPDF2
import docx

# AI/ML imports with error handling
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
    print("AI libraries loaded successfully")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"Warning: AI libraries not installed. {e}")
    print("Install with: pip install sentence-transformers scikit-learn")

# Flask app setup
app = Flask(__name__)

# Configuration for Railway deployment
if os.environ.get('DATABASE_URL'):
    # Railway PostgreSQL - fix for newer SQLAlchemy versions
    database_url = os.environ.get('DATABASE_URL')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print("Using Railway PostgreSQL database")
else:
    # Local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///geosmart.db'
    print("Using local SQLite database")

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'GSL_2025_Geo$mart_Learn!ng_Pl@tf0rm_S3cur3_K3y_#Tr0phy_Winner$_Geography_M@ster_AI_P0w3r3d')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize AI model for Q&A
embedding_model = None
if AI_AVAILABLE:
    try:
        print("Loading AI model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("AI model loaded successfully!")
    except Exception as e:
        print(f"Error loading AI model: {e}")
        AI_AVAILABLE = False
        embedding_model = None

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')
    points = db.Column(db.Integer, default=0)
    streak_days = db.Column(db.Integer, default=0)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Add relationships
    taught_courses = db.relationship('Course', backref='teacher', lazy=True)
    student_progress = db.relationship('StudentProgress', backref='student', lazy=True)
    questions_asked = db.relationship('Question', foreign_keys='Question.student_id', backref='student', lazy=True)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Add relationship
    lessons = db.relationship('Lesson', backref='course', lazy=True)

class Lesson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    video_url = db.Column(db.String(500))
    content_file = db.Column(db.String(200))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Add relationships
    knowledge_base = db.relationship('KnowledgeBase', backref='lesson', lazy=True)
    questions = db.relationship('Question', backref='lesson', lazy=True)
    quiz_questions = db.relationship('Quiz', backref='lesson', lazy=True)
    student_progress = db.relationship('StudentProgress', backref='lesson', lazy=True)

class StudentProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    quiz_score = db.Column(db.Integer, default=0)
    completed_at = db.Column(db.DateTime)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    answer_text = db.Column(db.Text)
    is_answered = db.Column(db.Boolean, default=False)
    answered_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class KnowledgeBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.Text)  # Store as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    option_a = db.Column(db.String(200))
    option_b = db.Column(db.String(200))
    option_c = db.Column(db.String(200))
    option_d = db.Column(db.String(200))
    correct_answer = db.Column(db.String(1))  # A, B, C, or D

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility Functions
def extract_text_from_pdf(file_path):
    """Extract text from PDF file with better error handling"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {e}")
                    continue
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        print(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except:
        return ""

def process_document_for_qa(lesson_id, text):
    """Process document text for Q&A by creating embeddings with better chunking"""
    if not embedding_model or not text.strip():
        print("No embedding model or empty text")
        return False
    
    print(f"Processing document for lesson {lesson_id}, text length: {len(text)}")
    
    # Better text chunking
    # Split by paragraphs first, then by sentences if needed
    paragraphs = text.split('\n\n')
    chunks = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) < 50:  # Skip very short paragraphs
            continue
            
        if len(paragraph) <= 500:
            chunks.append(paragraph)
        else:
            # Split long paragraphs by sentences
            sentences = re.split(r'[.!?]+', paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk + sentence) <= 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings and store in database
    successful_chunks = 0
    for i, chunk in enumerate(chunks):
        if len(chunk) > 20:  # Only process meaningful chunks
            try:
                print(f"Processing chunk {i+1}/{len(chunks)}")
                embedding = embedding_model.encode([chunk])[0]
                embedding_json = json.dumps(embedding.tolist())
                
                kb_entry = KnowledgeBase(
                    lesson_id=lesson_id,
                    content=chunk,
                    embedding=embedding_json
                )
                db.session.add(kb_entry)
                successful_chunks += 1
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
    
    try:
        db.session.commit()
        print(f"Successfully processed {successful_chunks} chunks")
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        db.session.rollback()
        return False

def find_answer_from_knowledge_base(question, lesson_id=None):
    """Find answer using AI similarity search with better error handling"""
    if not embedding_model:
        return "AI Q&A system is not available. Please install the required libraries (sentence-transformers, scikit-learn) and restart the application."
    
    if not question.strip():
        return "Please provide a valid question."
    
    try:
        print(f"Processing question: {question}")
        question_embedding = embedding_model.encode([question])[0]
        
        # Query knowledge base
        if lesson_id:
            kb_entries = KnowledgeBase.query.filter_by(lesson_id=lesson_id).all()
            print(f"Found {len(kb_entries)} knowledge base entries for lesson {lesson_id}")
        else:
            kb_entries = KnowledgeBase.query.all()
            print(f"Found {len(kb_entries)} total knowledge base entries")
        
        if not kb_entries:
            return "No study materials have been uploaded yet. Please ask your teacher to upload lesson content, or try asking a general question."
        
        best_matches = []
        
        for entry in kb_entries:
            if entry.embedding:
                try:
                    content_embedding = np.array(json.loads(entry.embedding))
                    similarity = cosine_similarity([question_embedding], [content_embedding])[0][0]
                    
                    if similarity > 0.2:  # Lower threshold for more results
                        best_matches.append({
                            'content': entry.content,
                            'similarity': similarity,
                            'lesson_id': entry.lesson_id
                        })
                except Exception as e:
                    print(f"Error processing embedding: {e}")
                    continue
        
        # Sort by similarity
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if best_matches:
            # Use the best match
            best_match = best_matches[0]
            similarity_score = best_match['similarity']
            content = best_match['content']
            
            print(f"Best match similarity: {similarity_score}")
            
            if similarity_score > 0.5:
                response = f"Based on the study materials:\n\n{content}\n\nConfidence: High"
            elif similarity_score > 0.3:
                response = f"I found some related information:\n\n{content}\n\nConfidence: Medium - You may want to ask your teacher for more specific details."
            else:
                response = f"Here's what I found that might be related:\n\n{content}\n\nConfidence: Low - This might not directly answer your question. Consider asking your teacher or rephrasing your question."
            
            return response
        else:
            return "I couldn't find relevant information in the uploaded materials. Try rephrasing your question, or ask your teacher for help with this topic."
    
    except Exception as e:
        print(f"Error in find_answer_from_knowledge_base: {e}")
        return f"I encountered an error while processing your question. Please try again, or contact your teacher if the problem persists."

def generate_sample_quiz_questions(lesson_title, content):
    """Generate some basic quiz questions based on content"""
    questions = []
    
    # Extract key terms and concepts
    words = content.lower().split()
    common_geo_terms = ['climate', 'weather', 'temperature', 'rainfall', 'desert', 'mountain', 'river', 'ocean', 'continent', 'country', 'city', 'population', 'geography', 'latitude', 'longitude', 'equator']
    
    found_terms = [term for term in common_geo_terms if term in words]
    
    if found_terms:
        # Create a simple question about the first found term
        term = found_terms[0].capitalize()
        questions.append({
            'question': f'Which of the following is related to {term}?',
            'option_a': f'{term} studies',
            'option_b': 'Mathematics',
            'option_c': 'Literature',
            'option_d': 'Music',
            'correct_answer': 'A'
        })
    
    # Generic geography question
    questions.append({
        'question': f'What is the main topic of the lesson "{lesson_title}"?',
        'option_a': 'Geography concepts',
        'option_b': 'Mathematical equations',
        'option_c': 'Historical events',
        'option_d': 'Scientific experiments',
        'correct_answer': 'A'
    })
    
    return questions[:2]  # Return max 2 questions

def generate_learning_recommendations(student, performance, recent_activity):
    """Generate personalized learning recommendations"""
    recommendations = []
    
    # Based on quiz performance
    weak_subjects = [subject for subject, data in performance.items() if data['average_score'] < 70]
    if weak_subjects:
        recommendations.append(f"Focus on improving in: {', '.join(weak_subjects)}")
    
    # Based on activity
    if len(recent_activity) == 0:
        recommendations.append("Start learning! Complete your first lesson today.")
    elif len(recent_activity) < 3:
        recommendations.append("Try to maintain consistency - aim for 3+ lessons per week.")
    
    # Based on points and engagement
    if student.points < 100:
        recommendations.append("Complete more quizzes to earn points and climb the leaderboard!")
    
    if student.streak_days == 0:
        recommendations.append("Start a learning streak! Log in daily to maintain momentum.")
    
    return recommendations

def generate_brief_summary(content):
    """Generate a brief summary of the content"""
    # Simple extractive summarization
    sentences = content.split('. ')
    if len(sentences) <= 3:
        return content
    
    # Take first sentence, a middle sentence, and last sentence
    summary_sentences = [
        sentences[0],
        sentences[len(sentences)//2] if len(sentences) > 2 else "",
        sentences[-1] if sentences[-1].strip() else sentences[-2]
    ]
    
    summary = '. '.join(filter(None, summary_sentences))
    return f"Brief Summary: {summary}"

def generate_detailed_summary(content):
    """Generate a detailed summary with main topics"""
    # Split content into paragraphs and extract key information
    paragraphs = content.split('\n\n')
    
    summary_parts = []
    for i, paragraph in enumerate(paragraphs[:5]):  # Limit to first 5 paragraphs
        if len(paragraph.strip()) > 50:  # Only meaningful paragraphs
            # Extract first 2 sentences of each paragraph
            sentences = paragraph.split('. ')[:2]
            summary_parts.append('. '.join(sentences) + '.')
    
    detailed_summary = '\n\n'.join(summary_parts)
    return f"Detailed Summary:\n\n{detailed_summary}"

def generate_key_points_summary(content):
    """Extract key points from the content"""
    sentences = content.split('. ')
    key_points = []
    
    # Look for sentences with key indicators
    key_indicators = ['important', 'key', 'main', 'primary', 'essential', 'crucial', 'significant']
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in key_indicators):
            key_points.append(sentence.strip())
        elif len(sentence) > 20 and len(sentence) < 150:  # Good length sentences
            key_points.append(sentence.strip())
    
    # Limit to top 5 key points
    key_points = key_points[:5]
    
    formatted_points = '\n'.join([f"• {point}" for point in key_points])
    return f"Key Points:\n\n{formatted_points}"

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        flash('Registration successful!')
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/teacher_dashboard')
@login_required
def teacher_dashboard():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
    
    courses = Course.query.filter_by(teacher_id=current_user.id).all()
    recent_questions = Question.query.join(Lesson).join(Course).filter(
        Course.teacher_id == current_user.id
    ).order_by(Question.created_at.desc()).limit(10).all()
    
    return render_template('teacher_dashboard.html', courses=courses, recent_questions=recent_questions)

@app.route('/student_dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        return redirect(url_for('teacher_dashboard'))
    
    courses = Course.query.all()
    progress = StudentProgress.query.filter_by(student_id=current_user.id).all()
    
    # Daily challenge
    daily_facts = [
        "The Sahara Desert is larger than the entire United States!",
        "Mount Everest grows about 4 millimeters each year.",
        "The Dead Sea is actually a salt lake, not a sea.",
        "Antarctica is considered a desert despite all the ice.",
        "The Amazon River is longer than the distance from New York to Rome."
    ]
    daily_fact = random.choice(daily_facts)
    
    return render_template('student_dashboard.html', courses=courses, progress=progress, daily_fact=daily_fact)

@app.route('/create_course', methods=['GET', 'POST'])
@login_required
def create_course():
    if current_user.role != 'teacher':
        flash('Access denied')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        
        course = Course(
            title=title,
            description=description,
            teacher_id=current_user.id
        )
        db.session.add(course)
        db.session.commit()
        
        flash('Course created successfully!')
        return redirect(url_for('teacher_dashboard'))
    
    return render_template('create_course.html')

@app.route('/course/<int:course_id>')
@login_required
def view_course(course_id):
    course = Course.query.get_or_404(course_id)
    lessons = Lesson.query.filter_by(course_id=course_id).all()
    return render_template('course.html', course=course, lessons=lessons)

@app.route('/create_lesson/<int:course_id>', methods=['GET', 'POST'])
@login_required
def create_lesson(course_id):
    """Complete create lesson route with robust PDF processing and AI training"""
    if current_user.role != 'teacher':
        flash('Access denied - Teachers only')
        return redirect(url_for('index'))
    
    course = Course.query.get_or_404(course_id)
    if course.teacher_id != current_user.id:
        flash('Access denied - You can only create lessons for your own courses')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title', '').strip()
            description = request.form.get('description', '').strip()
            video_url = request.form.get('video_url', '').strip()
            notes = request.form.get('notes', '').strip()
            
            # Validation
            if not title:
                flash('Lesson title is required')
                return render_template('create_lesson.html', course=course)
            
            print(f"Creating lesson: {title} for course: {course.title}")
            
            # Create lesson object
            lesson = Lesson(
                title=title,
                description=description,
                course_id=course_id,
                video_url=video_url,
                notes=notes
            )
            
            # Handle file upload
            uploaded_file = None
            if 'content_file' in request.files:
                file = request.files['content_file']
                if file and file.filename and file.filename != '':
                    # Validate file type
                    allowed_extensions = {'.pdf', '.docx', '.txt'}
                    file_ext = os.path.splitext(file.filename)[1].lower()
                    
                    if file_ext in allowed_extensions:
                        try:
                            filename = secure_filename(file.filename)
                            # Add timestamp to avoid conflicts
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                            filename = timestamp + filename
                            
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            file.save(file_path)
                            
                            # Verify file was saved
                            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                lesson.content_file = filename
                                uploaded_file = file_path
                                print(f"File uploaded successfully: {filename}")
                            else:
                                flash('Error: File upload failed - file is empty or could not be saved')
                                return render_template('create_lesson.html', course=course)
                                
                        except Exception as e:
                            print(f"Error uploading file: {e}")
                            flash(f'Error uploading file: {str(e)}')
                            return render_template('create_lesson.html', course=course)
                    else:
                        flash('Invalid file type. Please upload PDF, DOCX, or TXT files only.')
                        return render_template('create_lesson.html', course=course)
            
            # Save lesson to database first
            try:
                db.session.add(lesson)
                db.session.commit()
                print(f"Lesson saved with ID: {lesson.id}")
            except Exception as e:
                print(f"Error saving lesson to database: {e}")
                flash('Error creating lesson in database')
                return render_template('create_lesson.html', course=course)
            
            # Now process content for AI training
            training_success = True
            extracted_text = ""
            
            # Process uploaded file for Q&A
            if uploaded_file and os.path.exists(uploaded_file):
                print(f"Processing uploaded file: {lesson.content_file}")
                
                try:
                    if lesson.content_file.lower().endswith('.pdf'):
                        print("Extracting text from PDF...")
                        extracted_text = extract_text_from_pdf(uploaded_file)
                        
                    elif lesson.content_file.lower().endswith('.docx'):
                        print("Extracting text from DOCX...")
                        extracted_text = extract_text_from_docx(uploaded_file)
                        
                    elif lesson.content_file.lower().endswith('.txt'):
                        print("Reading text file...")
                        try:
                            with open(uploaded_file, 'r', encoding='utf-8') as f:
                                extracted_text = f.read()
                        except UnicodeDecodeError:
                            # Try with different encoding
                            with open(uploaded_file, 'r', encoding='latin1') as f:
                                extracted_text = f.read()
                    
                    if extracted_text and len(extracted_text.strip()) > 50:
                        print(f"Extracted text length: {len(extracted_text)} characters")
                        
                        # Process for AI training
                        if AI_AVAILABLE and embedding_model:
                            print("Starting AI training...")
                            training_result = process_document_for_qa(lesson.id, extracted_text)
                            if not training_result:
                                training_success = False
                                print("AI training failed")
                        else:
                            training_success = False
                            print("AI not available for training")
                    else:
                        print("No meaningful text extracted from file")
                        flash('Warning: Could not extract meaningful text from the uploaded file')
                        
                except Exception as e:
                    print(f"Error processing uploaded file: {e}")
                    flash(f'Warning: Error processing uploaded file: {str(e)}')
                    training_success = False
            
            # Also process lesson notes for Q&A
            if notes and len(notes.strip()) > 50:
                print("Processing lesson notes for AI training...")
                try:
                    if AI_AVAILABLE and embedding_model:
                        notes_result = process_document_for_qa(lesson.id, notes)
                        if notes_result:
                            print("Notes processed successfully for AI")
                        else:
                            print("Failed to process notes for AI")
                            training_success = False
                    else:
                        print("AI not available for processing notes")
                        training_success = False
                except Exception as e:
                    print(f"Error processing notes: {e}")
                    training_success = False
            
            # Provide feedback to user
            if training_success and (extracted_text or notes):
                flash('✅ Lesson created successfully! AI training completed - students can now ask questions about this content.')
            elif not AI_AVAILABLE:
                flash('⚠️ Lesson created, but AI Q&A is not available. Install sentence-transformers and scikit-learn for AI functionality.')
            elif not (extracted_text or notes):
                flash('⚠️ Lesson created, but no content was provided for AI training. Add notes or upload a document for Q&A functionality.')
            else:
                flash('⚠️ Lesson created, but AI training encountered issues. Q&A may not work properly for this lesson.')
            
            # Create some sample quiz questions if none exist
            try:
                existing_quiz = Quiz.query.filter_by(lesson_id=lesson.id).first()
                if not existing_quiz and (extracted_text or notes):
                    sample_questions = generate_sample_quiz_questions(lesson.title, extracted_text or notes)
                    for q_data in sample_questions:
                        quiz_q = Quiz(
                            lesson_id=lesson.id,
                            question=q_data['question'],
                            option_a=q_data['option_a'],
                            option_b=q_data['option_b'],
                            option_c=q_data['option_c'],
                            option_d=q_data['option_d'],
                            correct_answer=q_data['correct_answer']
                        )
                        db.session.add(quiz_q)
                    
                    db.session.commit()
                    print(f"Created {len(sample_questions)} sample quiz questions")
                    
            except Exception as e:
                print(f"Error creating sample quiz questions: {e}")
            
            return redirect(url_for('view_course', course_id=course_id))
            
        except Exception as e:
            print(f"Unexpected error in create_lesson: {e}")
            flash(f'An unexpected error occurred: {str(e)}')
            return render_template('create_lesson.html', course=course)
    
    # GET request - show the form
    return render_template('create_lesson.html', course=course)

@app.route('/lesson/<int:lesson_id>')
@login_required
def view_lesson(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    course = Course.query.get_or_404(lesson.course_id)
    
    # Get or create progress record
    progress = StudentProgress.query.filter_by(
        student_id=current_user.id, lesson_id=lesson_id
    ).first()
    
    if not progress and current_user.role == 'student':
        progress = StudentProgress(student_id=current_user.id, lesson_id=lesson_id)
        db.session.add(progress)
        db.session.commit()
    
    # Get quiz questions
    quiz_questions = Quiz.query.filter_by(lesson_id=lesson_id).all()
    
    return render_template('lesson.html', lesson=lesson, course=course, 
                         progress=progress, quiz_questions=quiz_questions)

@app.route('/complete_lesson/<int:lesson_id>', methods=['POST'])
@login_required
def complete_lesson(lesson_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Access denied'}), 403
    
    progress = StudentProgress.query.filter_by(
        student_id=current_user.id, lesson_id=lesson_id
    ).first()
    
    if progress:
        progress.completed = True
        progress.completed_at = datetime.utcnow()
        current_user.points += 10  # Award points for completion
        db.session.commit()
        
        return jsonify({'success': True, 'points': current_user.points})
    
    return jsonify({'error': 'Progress not found'}), 404

@app.route('/ask_question', methods=['GET', 'POST'])
@login_required
def ask_question():
    if request.method == 'GET':
        return render_template('ask_question.html')
    
    # Handle POST request
    if request.is_json:
        data = request.get_json()
        question_text = data.get('question')
        lesson_id = data.get('lesson_id')
    else:
        question_text = request.form.get('question')
        lesson_id = request.form.get('lesson_id')
    
    if not question_text:
        return jsonify({'error': 'Question text is required'}), 400
    
    print(f"Received question: {question_text}, lesson_id: {lesson_id}")
    
    # Try to find answer automatically
    try:
        answer = find_answer_from_knowledge_base(question_text, lesson_id)
        print(f"Generated answer: {answer[:100]}...")
    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = "I'm sorry, there was an error processing your question. Please try again or contact your teacher."
    
    # Save question to database
    try:
        question = Question(
            student_id=current_user.id,
            lesson_id=lesson_id,
            question_text=question_text,
            answer_text=answer,
            is_answered=True
        )
        db.session.add(question)
        db.session.commit()
        print("Question saved to database")
    except Exception as e:
        print(f"Error saving question: {e}")
    
    return jsonify({'answer': answer})

@app.route('/submit_quiz/<int:lesson_id>', methods=['POST'])
@login_required
def submit_quiz(lesson_id):
    if current_user.role != 'student':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    answers = data.get('answers', {})
    
    quiz_questions = Quiz.query.filter_by(lesson_id=lesson_id).all()
    correct_count = 0
    
    for question in quiz_questions:
        submitted_answer = answers.get(str(question.id))
        if submitted_answer == question.correct_answer:
            correct_count += 1
    
    if quiz_questions:
        score = int((correct_count / len(quiz_questions)) * 100)
    else:
        score = 0
    
    # Update progress
    progress = StudentProgress.query.filter_by(
        student_id=current_user.id, lesson_id=lesson_id
    ).first()
    
    if progress:
        progress.quiz_score = score
        current_user.points += score // 10  # Award points based on score
        db.session.commit()
    
    return jsonify({
        'score': score,
        'correct': correct_count,
        'total': len(quiz_questions),
        'points': current_user.points
    })

@app.route('/leaderboard')
@login_required
def leaderboard():
    top_students = User.query.filter_by(role='student').order_by(User.points.desc()).limit(10).all()
    return render_template('leaderboard.html', students=top_students)

@app.route('/questions')
@login_required
def view_questions():
    if current_user.role == 'teacher':
        questions = Question.query.join(Lesson).join(Course).filter(
            Course.teacher_id == current_user.id
        ).order_by(Question.created_at.desc()).all()
    else:
        questions = Question.query.filter_by(student_id=current_user.id).order_by(
            Question.created_at.desc()
        ).all()
    
    return render_template('questions.html', questions=questions)

@app.route('/api/student_summary/<int:student_id>')
@login_required
def get_student_summary(student_id):
    """Generate comprehensive student learning summary"""
    if current_user.role != 'teacher' and current_user.id != student_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    student = User.query.get_or_404(student_id)
    
    # Get all progress data
    progress_data = StudentProgress.query.filter_by(student_id=student_id).all()
    questions_data = Question.query.filter_by(student_id=student_id).all()
    
    # Calculate statistics
    total_lessons = len(progress_data)
    completed_lessons = sum(1 for p in progress_data if p.completed)
    average_quiz_score = sum(p.quiz_score for p in progress_data if p.quiz_score > 0) / max(1, len([p for p in progress_data if p.quiz_score > 0]))
    total_questions_asked = len(questions_data)
    
    # Learning pattern analysis
    recent_activity = []
    for progress in progress_data[-10:]:  # Last 10 activities
        if progress.completed_at:
            recent_activity.append({
                'lesson': progress.lesson.title,
                'completed': progress.completed_at.strftime('%Y-%m-%d'),
                'score': progress.quiz_score
            })
    
    # Subject strengths analysis
    subject_performance = defaultdict(list)
    for progress in progress_data:
        if progress.quiz_score > 0:
            course_title = progress.lesson.course.title
            subject_performance[course_title].append(progress.quiz_score)
    
    strengths = {}
    for subject, scores in subject_performance.items():
        avg_score = sum(scores) / len(scores)
        strengths[subject] = {
            'average_score': round(avg_score, 2),
            'lessons_completed': len(scores),
            'performance_level': 'Excellent' if avg_score >= 80 else 'Good' if avg_score >= 60 else 'Needs Improvement'
        }
    
    summary = {
        'student_info': {
            'username': student.username,
            'points': student.points,
            'streak_days': student.streak_days,
            'joined_date': student.created_at.strftime('%Y-%m-%d')
        },
        'learning_stats': {
            'total_lessons_enrolled': total_lessons,
            'lessons_completed': completed_lessons,
            'completion_rate': round((completed_lessons / max(1, total_lessons)) * 100, 2),
            'average_quiz_score': round(average_quiz_score, 2),
            'total_questions_asked': total_questions_asked
        },
        'recent_activity': recent_activity,
        'subject_performance': strengths,
        'recommendations': generate_learning_recommendations(student, strengths, recent_activity)
    }
    
    return jsonify(summary)

@app.route('/api/generate_summary', methods=['POST'])
@login_required
def generate_content_summary():
    """Generate AI summary of lesson content when students request"""
    data = request.get_json()
    lesson_id = data.get('lesson_id')
    summary_type = data.get('type', 'brief')  # brief, detailed, key_points
    
    lesson = Lesson.query.get_or_404(lesson_id)
    
    # Get all content for the lesson
    content_parts = []
    
    if lesson.notes:
        content_parts.append(lesson.notes)
    
    # Get knowledge base content for this lesson
    kb_entries = KnowledgeBase.query.filter_by(lesson_id=lesson_id).all()
    for entry in kb_entries:
        content_parts.append(entry.content)
    
    if not content_parts:
        return jsonify({'error': 'No content available to summarize'}), 400
    
    # Combine all content
    full_content = " ".join(content_parts)
    
    # Generate different types of summaries
    if summary_type == 'key_points':
        summary = generate_key_points_summary(full_content)
    elif summary_type == 'detailed':
        summary = generate_detailed_summary(full_content)
    else:  # brief
        summary = generate_brief_summary(full_content)
    
    # Store the summary request for analytics
    summary_request = Question(
        student_id=current_user.id,
        lesson_id=lesson_id,
        question_text=f"Summary requested: {summary_type}",
        answer_text=summary,
        is_answered=True
    )
    db.session.add(summary_request)
    db.session.commit()
    
    return jsonify({
        'summary': summary,
        'type': summary_type,
        'lesson_title': lesson.title
    })

@app.route('/api/recent_questions')
@login_required
def get_recent_questions():
    """Get recent questions for the current user"""
    recent_questions = Question.query.filter_by(student_id=current_user.id)\
        .order_by(Question.created_at.desc()).limit(5).all()
    
    questions_data = []
    for q in recent_questions:
        questions_data.append({
            'question': q.question_text,
            'created_at': q.created_at.strftime('%Y-%m-%d %H:%M'),
            'is_answered': q.is_answered
        })
    
    return jsonify({'questions': questions_data})

@app.route('/teacher/analytics')
@login_required
def teacher_analytics():
    """Teacher analytics dashboard"""
    if current_user.role != 'teacher':
        return redirect(url_for('index'))
    
    # Get courses taught by this teacher
    courses = Course.query.filter_by(teacher_id=current_user.id).all()
    course_ids = [course.id for course in courses]
    
    # Get all lessons for these courses
    lessons = Lesson.query.join(Course).filter(Course.teacher_id == current_user.id).all()
    lesson_ids = [lesson.id for lesson in lessons]
    
    # Analytics data
    analytics = {
        'total_courses': len(courses),
        'total_lessons': len(lessons),
        'total_students': len(set(sp.student_id for sp in StudentProgress.query.filter(StudentProgress.lesson_id.in_(lesson_ids)).all())),
        'total_questions': Question.query.filter(Question.lesson_id.in_(lesson_ids)).count(),
        'completion_rate': 0,
        'popular_lessons': [],
        'common_questions': []
    }
    
    # Calculate completion rate
    total_enrollments = StudentProgress.query.filter(StudentProgress.lesson_id.in_(lesson_ids)).count()
    completed_lessons = StudentProgress.query.filter(
        StudentProgress.lesson_id.in_(lesson_ids),
        StudentProgress.completed == True
    ).count()
    
    if total_enrollments > 0:
        analytics['completion_rate'] = round((completed_lessons / total_enrollments) * 100, 2)
    
    return render_template('teacher_analytics.html', analytics=analytics, courses=courses)

@app.route('/debug/knowledge_base/<int:lesson_id>')
@login_required
def debug_knowledge_base(lesson_id):
    """Debug route to check knowledge base content"""
    if current_user.role != 'teacher':
        return "Access denied", 403
    
    kb_entries = KnowledgeBase.query.filter_by(lesson_id=lesson_id).all()
    
    debug_info = {
        'lesson_id': lesson_id,
        'total_entries': len(kb_entries),
        'ai_available': AI_AVAILABLE,
        'embedding_model_loaded': embedding_model is not None,
        'entries': []
    }
    
    for entry in kb_entries[:5]:  # Show first 5 entries
        debug_info['entries'].append({
            'id': entry.id,
            'content_preview': entry.content[:200] + "..." if len(entry.content) > 200 else entry.content,
            'has_embedding': entry.embedding is not None,
            'created_at': entry.created_at.isoformat()
        })
    
    return jsonify(debug_info)

# Error handlers for better Railway deployment
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Health check route for Railway
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'ai_available': AI_AVAILABLE,
        'timestamp': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    print("Starting GeoSmart Learning Platform...")
    port = int(os.environ.get('PORT', 5000))
     debug_mode = os.environ.get('RENDER') is None  # ← NEW LINE
    
    if not debug_mode:
        print(f"Production mode - Server starting on port {port}")
    else:
        print("Development mode - Access at: http://localhost:5000")
    
    # Create database tables
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created/verified successfully")
            
            # Create a default admin user if none exists (optional)
            if not User.query.filter_by(role='teacher').first():
                admin_user = User(
                    username='admin',
                    email='admin@geosmart.com',
                    password_hash=generate_password_hash('admin123'),
                    role='teacher'
                )
                db.session.add(admin_user)
                db.session.commit()
                print("Default admin user created (username: admin, password: admin123)")
                
        except Exception as e:
            print(f"Database setup error: {e}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
