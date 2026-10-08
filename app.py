import os
import json
import base64
from datetime import datetime, date, time
from pathlib import Path
from dotenv import load_dotenv
import pytz

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

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject_name = db.Column(db.String(100), nullable=False)
    day_of_week = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    attendances = db.relationship('Attendance', backref='timetable', lazy=True, cascade='all, delete-orphan')


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timetable_id = db.Column(db.Integer, db.ForeignKey('timetable.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=today_ist)
    timestamp = db.Column(db.DateTime, nullable=False, default=now_ist)
    status = db.Column(db.String(10), nullable=False)
    photo_path = db.Column(db.String(255), nullable=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_time(t):
    for fmt in ('%H:%M', '%I:%M %p', '%H.%M'):
        try:
            return datetime.strptime(t.strip(), fmt).time()
        except ValueError:
            continue
    return None


from openai import OpenAI

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
def index():
    today = today_ist()
    weekday = current_weekday_ist()
    now = current_time_ist()
    
    classes = Timetable.query.filter_by(day_of_week=weekday).order_by(Timetable.start_time).all()
    
    attended = {a.timetable_id: a for a in Attendance.query.filter_by(date=today).all()}
    
    return render_template('index.html', classes=classes, attended=attended, now=now, today=today)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
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
                            subject_name=item['subject'],
                            day_of_week=d,
                            start_time=st,
                            end_time=et
                        ))
                db.session.commit()
                flash('Timetable imported successfully!')
            except Exception as e:
                flash(f'Error processing timetable: {e}')
            return redirect(url_for('index'))
    return render_template('upload.html')


@app.route('/mark/<int:timetable_id>/<status>', methods=['POST'])
def mark(timetable_id, status):
    if status not in ('present', 'absent'):
        return redirect(url_for('index'))
    
    today = today_ist()
    existing = Attendance.query.filter_by(timetable_id=timetable_id, date=today).first()
    
    photo_path = None
    if status == 'present':
        if 'photo' not in request.files:
            flash('Photo required for present')
            return redirect(url_for('index'))
        file = request.files['photo']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Valid photo required')
            return redirect(url_for('index'))
        filename = secure_filename(f"{today}_{timetable_id}_{file.filename}")
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(photo_path)
    
    if existing:
        existing.status = status
        existing.photo_path = photo_path or existing.photo_path
    else:
        db.session.add(Attendance(timetable_id=timetable_id, date=today, status=status, photo_path=photo_path))
    
    db.session.commit()
    return redirect(url_for('index'))


@app.route('/analytics')
def analytics():
    records = Attendance.query.join(Timetable).all()
    
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
    
    history = Attendance.query.join(Timetable).order_by(Attendance.date.desc(), Timetable.start_time).all()
    
    return render_template('analytics.html', stats=stats, history=history)


@app.route('/timetable')
def view_timetable():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    timetable = {d: [] for d in days}
    for t in Timetable.query.order_by(Timetable.day_of_week, Timetable.start_time).all():
        timetable[days[t.day_of_week]].append(t)
    return render_template('timetable.html', timetable=timetable, days=days)


with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)