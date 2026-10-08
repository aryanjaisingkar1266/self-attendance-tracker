import os
import json
import base64
import secrets
from datetime import datetime, date, time
from pathlib import Path
from functools import wraps

from dotenv import load_dotenv
import pytz
from flask import Flask, render_template, request, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

load_dotenv()

IST = pytz.timezone('Asia/Kolkata')

def now_ist():
    return datetime.now(IST)

def today_ist():
    return now_ist().date()

def current_time_ist():
    return now_ist().time()

def current_weekday_ist():
    return now_ist().weekday()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///attendance.db').replace('postgres://', 'postgresql://')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 30

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

csrf = CSRFProtect(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv('REDIS_URL', 'memory://'),
)

Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_stream):
    header = file_stream.read(512)
    file_stream.seek(0)
    if header.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if header.startswith(b'RIFF') and b'WEBP' in header[:12]:
        return '.webp'
    return None

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=now_ist)
    timetables = db.relationship('Timetable', backref='user', lazy=True, cascade='all, delete-orphan')
    attendances = db.relationship('Attendance', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='scrypt')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    subject_name = db.Column(db.String(100), nullable=False)
    day_of_week = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    created_at = db.Column(db.DateTime, default=now_ist)
    attendances = db.relationship('Attendance', backref='timetable', lazy=True, cascade='all, delete-orphan')

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    timetable_id = db.Column(db.Integer, db.ForeignKey('timetable.id'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, default=today_ist, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=now_ist)
    status = db.Column(db.String(10), nullable=False)
    photo_path = db.Column(db.String(255), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def parse_time(t):
    for fmt in ('%H:%M', '%I:%M %p', '%H.%M'):
        try:
            return datetime.strptime(t.strip(), fmt).time()
        except ValueError:
            continue
    return None

def call_vision_api(image_path):
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not configured")
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    prompt = "Extract timetable as JSON array: [{'day', 'start_time', 'end_time', 'subject'}]. Days: Monday-Sunday. Times: HH:MM 24h. Output ONLY valid JSON. No conversational text, no markdown block ticks, just the raw JSON array."

    response = client.chat.completions.create(
        model="meta/llama-3.2-90b-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }]
    )
    
    raw_content = response.choices[0].message.content.strip()
    
    clean_json = raw_content
    if clean_json.startswith('```json'):
        clean_json = clean_json[7:]
    elif clean_json.startswith('```'):
        clean_json = clean_json[3:]
    if clean_json.endswith('```'):
        clean_json = clean_json[:-3]
        
    clean_json = clean_json.strip()
    
    try:
        return json.loads(clean_json)
    except Exception as e:
        raise ValueError(f"AI returned invalid JSON: {raw_content[:100]}...")

@app.route('/')
@login_required
def index():
    today = today_ist()
    weekday = current_weekday_ist()
    now = current_time_ist()
    
    classes = Timetable.query.filter_by(user_id=current_user.id, day_of_week=weekday).order_by(Timetable.start_time).all()
    
    attended = {a.timetable_id: a for a in Attendance.query.filter_by(user_id=current_user.id, date=today).all()}
    
    return render_template('index.html', classes=classes, attended=attended, now=now, today=today)

@app.route('/signup', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('auth.html', mode='signup')
        
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('auth.html', mode='signup')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('auth.html', mode='signup')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('auth.html', mode='signup')
        
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully. Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth.html', mode='signup')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        
        flash('Invalid email or password.', 'error')
    
    return render_template('auth.html', mode='login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            ext = validate_image(file.stream)
            if not ext:
                flash('Invalid image file.', 'error')
                return redirect(request.url)
            
            filename = secure_filename(f"{current_user.id}_{now_ist().strftime('%Y%m%d_%H%M%S')}{ext}")
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            try:
                schedule = call_vision_api(path)
                day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
                
                for item in schedule:
                    d = day_map.get(item['day'][:3].title())
                    st = parse_time(item['start_time'])
                    et = parse_time(item['end_time'])
                    if d is not None and st and et:
                        db.session.add(Timetable(
                            user_id=current_user.id,
                            subject_name=item['subject'][:100],
                            day_of_week=d,
                            start_time=st,
                            end_time=et
                        ))
                db.session.commit()
                flash('Timetable imported successfully!', 'success')
            except Exception as e:
                flash(f'Error processing timetable: {str(e)[:200]}', 'error')
            finally:
                try:
                    os.remove(path)
                except:
                    pass
            return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/mark/<int:timetable_id>/<status>', methods=['POST'])
@login_required
def mark(timetable_id, status):
    if status not in ('present', 'absent'):
        abort(400)
    
    timetable = Timetable.query.filter_by(id=timetable_id, user_id=current_user.id).first_or_404()
    
    today = today_ist()
    existing = Attendance.query.filter_by(user_id=current_user.id, timetable_id=timetable_id, date=today).first()
    
    photo_path = None
    if status == 'present':
        if 'photo' not in request.files:
            flash('Photo required for present.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['photo']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Valid photo required.', 'error')
            return redirect(url_for('index'))
        
        ext = validate_image(file.stream)
        if not ext:
            flash('Invalid image file.', 'error')
            return redirect(url_for('index'))
        
        filename = secure_filename(f"{current_user.id}_{today}_{timetable_id}{ext}")
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(photo_path)
    
    if existing:
        existing.status = status
        existing.photo_path = photo_path or existing.photo_path
        existing.timestamp = now_ist()
    else:
        db.session.add(Attendance(
            user_id=current_user.id,
            timetable_id=timetable_id,
            date=today,
            status=status,
            photo_path=photo_path
        ))
    
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/analytics')
@login_required
def analytics():
    records = Attendance.query.filter_by(user_id=current_user.id).join(Timetable).all()
    
    stats = {}
    for a in records:
        subj = a.timetable.subject_name
        if subj not in stats:
            stats[subj] = {'total': 0, 'present': 0}
        stats[subj]['total'] += 1
        if a.status == 'present':
            stats[subj]['present'] += 1
    
    for subj in stats:
        stats[subj]['percent'] = round(stats[subj]['present'] / stats[subj]['total'] * 100) if stats[subj]['total'] else 0
    
    history = Attendance.query.filter_by(user_id=current_user.id).join(Timetable).order_by(Attendance.date.desc(), Timetable.start_time).all()
    
    return render_template('analytics.html', stats=stats, history=history)

@app.route('/timetable')
@login_required
def view_timetable():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    timetable = {d: [] for d in days}
    for t in Timetable.query.filter_by(user_id=current_user.id).order_by(Timetable.day_of_week, Timetable.start_time).all():
        timetable[days[t.day_of_week]].append(t)
    return render_template('timetable.html', timetable=timetable, days=days)

@app.route('/health')
def health():
    return {'status': 'ok', 'time': now_ist().isoformat()}, 200

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, message='Page not found'), 404

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(request.url or url_for('index')), 413

@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    return render_template('error.html', code=500, message='Internal server error'), 500

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_ENV') != 'production')