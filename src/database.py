import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("database", "app_storage.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

REFERENCE_PROJECTS = [
    {
        'project_id': 'PRJ0001',
        'project_type': 'Residential',
        'location': 'Kanchipuram',
        'area_sqft': 10000.0,
        'number_of_floors': 3,
        'material_type': 'Standard_RCC',
        'material_cost': 4000000.0,
        'labour_cost': 800000.0,
        'labour_count': 25,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 150000.0,
        'planned_duration': 200,
        'predicted_cost': 32432104.0,
        'created_at': '2026-10-01T13:24:10'  # 01/10/2026
    },
    {
        'project_id': 'PRJ0002',
        'project_type': 'Commercial',
        'location': 'Urban',
        'area_sqft': 50000.0,
        'number_of_floors': 10,
        'material_type': 'RCC',
        'material_cost': 50000000.0,
        'labour_cost': 25000000.0,
        'labour_count': 100,
        'equipment_type': 'Heavy',
        'equipment_cost': 18000000.0,
        'planned_duration': 24,
        'predicted_cost': 155383376.0,
        'created_at': '2026-10-01T09:15:00'  # 01/10/2026
    },
    {
        'project_id': 'PRJ0003',
        'project_type': 'Residential',
        'location': 'Vellore',
        'area_sqft': 2400.0,
        'number_of_floors': 1,
        'material_type': 'Premium_RCC',
        'material_cost': 800000.0,
        'labour_cost': 600000.0,
        'labour_count': 15,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 100000.0,
        'planned_duration': 192,
        'predicted_cost': 5145559.0,
        'created_at': '2026-09-20T14:18:57'  # 20/09/2026
    },
    {
        'project_id': 'PRJ0004',
        'project_type': 'Villa',
        'location': 'Chennai',
        'area_sqft': 7000.0,
        'number_of_floors': 2,
        'material_type': 'Brick_Concrete',
        'material_cost': 5000000.0,
        'labour_cost': 1000000.0,
        'labour_count': 30,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 850000.0,
        'planned_duration': 600,
        'predicted_cost': 34999676.0,
        'created_at': '2026-09-20T10:30:00'  # 20/09/2026
    },
    {
        'project_id': 'PRJ0005',
        'project_type': 'Residential',
        'location': 'Vellore',
        'area_sqft': 1500.0,
        'number_of_floors': 2,
        'material_type': 'Standard_RCC',
        'material_cost': 2500000.0,
        'labour_cost': 800000.0,
        'labour_count': 18,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 20000.0,
        'planned_duration': 280,
        'predicted_cost': 4963126.5,
        'created_at': '2026-09-10T19:33:18'  # 10/09/2026
    },
    {
        'project_id': 'PRJ0006',
        'project_type': 'Residential',
        'location': 'Chennai',
        'area_sqft': 2500.0,
        'number_of_floors': 3,
        'material_type': 'Steel_Frame',
        'material_cost': 500000.0,
        'labour_cost': 800000.0,
        'labour_count': 20,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 15000.0,
        'planned_duration': 250,
        'predicted_cost': 9679311.0,
        'created_at': '2026-09-10T15:20:00'  # 10/09/2026
    },
    {
        'project_id': 'PRJ0007',
        'project_type': 'Residential',
        'location': 'Chennai',
        'area_sqft': 2500.0,
        'number_of_floors': 3,
        'material_type': 'Green_Building',
        'material_cost': 500000.0,
        'labour_cost': 800000.0,
        'labour_count': 20,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 15000.0,
        'planned_duration': 250,
        'predicted_cost': 10115778.0,
        'created_at': '2026-09-10T11:00:00'  # 10/09/2026
    },
    {
        'project_id': 'PRJ0008',
        'project_type': 'Residential',
        'location': 'Chennai',
        'area_sqft': 2500.0,
        'number_of_floors': 4,
        'material_type': 'AAC_Block',
        'material_cost': 4000000.0,
        'labour_cost': 600000.0,
        'labour_count': 20,
        'equipment_type': 'Concrete_Mixer',
        'equipment_cost': 150000.0,
        'planned_duration': 360,
        'predicted_cost': 12107922.0,
        'created_at': '2026-09-01T16:45:00'  # 01/09/2026
    },
    {
        'project_id': 'PRJ0009',
        'project_type': 'Commercial',
        'location': 'Coimbatore',
        'area_sqft': 15000.0,
        'number_of_floors': 6,
        'material_type': 'Premium_RCC',
        'material_cost': 12000000.0,
        'labour_cost': 2500000.0,
        'labour_count': 45,
        'equipment_type': 'Crane_and_Mixer',
        'equipment_cost': 400000.0,
        'planned_duration': 540,
        'predicted_cost': 56651000.0,
        'created_at': '2026-09-01T12:00:00'  # 01/09/2026
    },
    {
        'project_id': 'PRJ0010',
        'project_type': 'Commercial',
        'location': 'Urban',
        'area_sqft': 45000.0,
        'number_of_floors': 8,
        'material_type': 'RCC',
        'material_cost': 45000000.0,
        'labour_cost': 22000000.0,
        'labour_count': 85,
        'equipment_type': 'Heavy',
        'equipment_cost': 15000000.0,
        'planned_duration': 24,
        'predicted_cost': 142847568.0,
        'created_at': '2026-09-01T08:30:00'  # 01/09/2026
    }
]

REFERENCE_MONITORING_RECORDS = [
    {
        'project_id': 'PRJ0001',
        'sequence_number': 1,
        'planned_progress_percent': 70.0,
        'actual_progress_percent': 50.0,
        'actual_labour_count': 20,
        'material_availability_percent': 50.0,
        'equipment_availability_percent': 75.0,
        'expected_delay_days': 20.1,
        'delay_risk': 'Medium',
        'risk_probability': 0.78,
        'created_at': '2026-10-01T13:29:10'
    },
    {
        'project_id': 'PRJ0002',
        'sequence_number': 1,
        'planned_progress_percent': 40.0,
        'actual_progress_percent': 38.0,
        'actual_labour_count': 95,
        'material_availability_percent': 90.0,
        'equipment_availability_percent': 85.0,
        'expected_delay_days': 3.8,
        'delay_risk': 'Low',
        'risk_probability': 0.96,
        'created_at': '2026-10-01T09:30:00'
    },
    {
        'project_id': 'PRJ0002',
        'sequence_number': 2,
        'planned_progress_percent': 65.0,
        'actual_progress_percent': 42.0,
        'actual_labour_count': 50,
        'material_availability_percent': 55.0,
        'equipment_availability_percent': 50.0,
        'expected_delay_days': 51.4,
        'delay_risk': 'High',
        'risk_probability': 0.58,
        'created_at': '2026-10-01T10:15:00'
    },
    {
        'project_id': 'PRJ0003',
        'sequence_number': 1,
        'planned_progress_percent': 40.0,
        'actual_progress_percent': 28.0,
        'actual_labour_count': 12,
        'material_availability_percent': 70.0,
        'equipment_availability_percent': 75.0,
        'expected_delay_days': 13.2,
        'delay_risk': 'Medium',
        'risk_probability': 0.65,
        'created_at': '2026-09-20T14:22:35'
    },
    {
        'project_id': 'PRJ0004',
        'sequence_number': 1,
        'planned_progress_percent': 40.0,
        'actual_progress_percent': 28.0,
        'actual_labour_count': 49,
        'material_availability_percent': 80.0,
        'equipment_availability_percent': 75.0,
        'expected_delay_days': 26.8,
        'delay_risk': 'Low',
        'risk_probability': 0.77,
        'created_at': '2026-09-20T12:40:54'
    },
    {
        'project_id': 'PRJ0005',
        'sequence_number': 1,
        'planned_progress_percent': 30.0,
        'actual_progress_percent': 28.0,
        'actual_labour_count': 18,
        'material_availability_percent': 85.0,
        'equipment_availability_percent': 90.0,
        'expected_delay_days': 5.0,
        'delay_risk': 'Low',
        'risk_probability': 0.90,
        'created_at': '2026-09-10T19:40:00'
    },
    {
        'project_id': 'PRJ0006',
        'sequence_number': 1,
        'planned_progress_percent': 35.0,
        'actual_progress_percent': 32.0,
        'actual_labour_count': 20,
        'material_availability_percent': 90.0,
        'equipment_availability_percent': 85.0,
        'expected_delay_days': 4.2,
        'delay_risk': 'Low',
        'risk_probability': 0.92,
        'created_at': '2026-09-10T15:30:00'
    },
    {
        'project_id': 'PRJ0007',
        'sequence_number': 1,
        'planned_progress_percent': 25.0,
        'actual_progress_percent': 20.0,
        'actual_labour_count': 18,
        'material_availability_percent': 75.0,
        'equipment_availability_percent': 80.0,
        'expected_delay_days': 12.5,
        'delay_risk': 'Medium',
        'risk_probability': 0.70,
        'created_at': '2026-09-10T11:15:00'
    },
    {
        'project_id': 'PRJ0008',
        'sequence_number': 1,
        'planned_progress_percent': 40.0,
        'actual_progress_percent': 28.0,
        'actual_labour_count': 18,
        'material_availability_percent': 70.0,
        'equipment_availability_percent': 75.0,
        'expected_delay_days': 26.8,
        'delay_risk': 'Medium',
        'risk_probability': 0.80,
        'created_at': '2026-09-01T17:00:00'
    },
    {
        'project_id': 'PRJ0008',
        'sequence_number': 2,
        'planned_progress_percent': 50.0,
        'actual_progress_percent': 50.0,
        'actual_labour_count': 30,
        'material_availability_percent': 100.0,
        'equipment_availability_percent': 100.0,
        'expected_delay_days': 0.2,
        'delay_risk': 'Low',
        'risk_probability': 0.99,
        'created_at': '2026-09-01T17:30:00'
    },
    {
        'project_id': 'PRJ0009',
        'sequence_number': 1,
        'planned_progress_percent': 25.0,
        'actual_progress_percent': 23.0,
        'actual_labour_count': 42,
        'material_availability_percent': 95.0,
        'equipment_availability_percent': 90.0,
        'expected_delay_days': 6.1,
        'delay_risk': 'Low',
        'risk_probability': 0.91,
        'created_at': '2026-09-01T12:30:00'
    },
    {
        'project_id': 'PRJ0009',
        'sequence_number': 2,
        'planned_progress_percent': 50.0,
        'actual_progress_percent': 35.0,
        'actual_labour_count': 30,
        'material_availability_percent': 60.0,
        'equipment_availability_percent': 65.0,
        'expected_delay_days': 48.2,
        'delay_risk': 'High',
        'risk_probability': 0.62,
        'created_at': '2026-09-01T13:00:00'
    },
    {
        'project_id': 'PRJ0010',
        'sequence_number': 1,
        'planned_progress_percent': 40.0,
        'actual_progress_percent': 38.0,
        'actual_labour_count': 80,
        'material_availability_percent': 90.0,
        'equipment_availability_percent': 85.0,
        'expected_delay_days': 3.8,
        'delay_risk': 'Low',
        'risk_probability': 0.95,
        'created_at': '2026-09-01T09:00:00'
    }
]

def init_db(clean_non_reference=False):
    """
    Initializes SQLite tables for application-level project lifecycle tracking
    and seeds/maintains the 10 permanent reference projects.
    Optionally removes non-reference records if clean_non_reference is True.
    Does NOT modify the ML dataset.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Projects table for Stage 1 pre-construction records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            project_type TEXT NOT NULL,
            location TEXT NOT NULL,
            area_sqft REAL NOT NULL,
            number_of_floors INTEGER NOT NULL,
            material_type TEXT NOT NULL,
            material_cost REAL NOT NULL,
            labour_cost REAL NOT NULL,
            labour_count INTEGER NOT NULL,
            equipment_type TEXT NOT NULL,
            equipment_cost REAL NOT NULL,
            planned_duration INTEGER NOT NULL,
            predicted_cost REAL NOT NULL,
            is_completed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration for existing databases
    try:
        cursor.execute("ALTER TABLE projects ADD COLUMN is_completed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_project_id ON projects(project_id)
    """)
    
    # Monitoring records table for Stage 2 ongoing construction snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_records (
            monitoring_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            sequence_number INTEGER NOT NULL,
            planned_progress_percent REAL NOT NULL,
            actual_progress_percent REAL NOT NULL,
            actual_labour_count INTEGER NOT NULL,
            material_availability_percent REAL NOT NULL,
            equipment_availability_percent REAL NOT NULL,
            expected_delay_days REAL NOT NULL,
            delay_risk TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (project_id)
        )
    """)
    
    conn.commit()

    if clean_non_reference:
        ref_ids = [p['project_id'] for p in REFERENCE_PROJECTS]
        placeholders = ','.join(['?'] * len(ref_ids))
        cursor.execute(f"DELETE FROM monitoring_records WHERE project_id NOT IN ({placeholders})", ref_ids)
        cursor.execute(f"DELETE FROM projects WHERE project_id NOT IN ({placeholders})", ref_ids)
        conn.commit()

    # Insert or update reference projects with permanent dates
    for p in REFERENCE_PROJECTS:
        cursor.execute("""
            INSERT INTO projects (
                project_id, project_type, location, area_sqft, number_of_floors,
                material_type, material_cost, labour_cost, labour_count,
                equipment_type, equipment_cost, planned_duration, predicted_cost, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                project_type=excluded.project_type,
                location=excluded.location,
                area_sqft=excluded.area_sqft,
                number_of_floors=excluded.number_of_floors,
                material_type=excluded.material_type,
                material_cost=excluded.material_cost,
                labour_cost=excluded.labour_cost,
                labour_count=excluded.labour_count,
                equipment_type=excluded.equipment_type,
                equipment_cost=excluded.equipment_cost,
                planned_duration=excluded.planned_duration,
                predicted_cost=excluded.predicted_cost,
                created_at=excluded.created_at
        """, (
            p['project_id'], p['project_type'], p['location'], float(p['area_sqft']), int(p['number_of_floors']),
            p['material_type'], float(p['material_cost']), float(p['labour_cost']), int(p['labour_count']),
            p['equipment_type'], float(p['equipment_cost']), int(p['planned_duration']), float(p['predicted_cost']),
            p['created_at']
        ))
        
    # Seed monitoring records for reference projects if missing
    for m in REFERENCE_MONITORING_RECORDS:
        cursor.execute("""
            SELECT COUNT(*) FROM monitoring_records 
            WHERE project_id = ? AND sequence_number = ?
        """, (m['project_id'], m['sequence_number']))
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO monitoring_records (
                    project_id, sequence_number, planned_progress_percent, actual_progress_percent,
                    actual_labour_count, material_availability_percent, equipment_availability_percent,
                    expected_delay_days, delay_risk, risk_probability, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m['project_id'], m['sequence_number'], float(m['planned_progress_percent']), float(m['actual_progress_percent']),
                int(m['actual_labour_count']), float(m['material_availability_percent']), float(m['equipment_availability_percent']),
                float(m['expected_delay_days']), m['delay_risk'], float(m['risk_probability']),
                m['created_at']
            ))

    conn.commit()
    conn.close()

def cleanup_non_reference_projects():
    """
    Removes non-reference projects and associated monitoring records from SQLite database.
    Retains only PRJ0001 through PRJ0010.
    """
    ref_ids = [p['project_id'] for p in REFERENCE_PROJECTS]
    placeholders = ','.join(['?'] * len(ref_ids))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM monitoring_records WHERE project_id NOT IN ({placeholders})", ref_ids)
    cursor.execute(f"DELETE FROM projects WHERE project_id NOT IN ({placeholders})", ref_ids)
    conn.commit()
    conn.close()

def generate_next_project_id() -> str:
    """
    Generates a unique application-level Project_ID (e.g., PRJ0011).
    Guarantees unique ID generation.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT project_id FROM projects")
    existing_ids = {r['project_id'].strip() for r in cursor.fetchall() if r['project_id']}
    conn.close()
    
    max_num = 0
    for pid in existing_ids:
        if pid.startswith("PRJ"):
            try:
                num = int(pid[3:])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
                
    next_num = max_num + 1
    candidate = f"PRJ{next_num:04d}"
    while candidate in existing_ids:
        next_num += 1
        candidate = f"PRJ{next_num:04d}"
    return candidate

def save_project(p: dict) -> str:
    """
    Saves a Stage 1 project record to SQLite.
    Returns the project_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    project_id = p.get('project_id') or p.get('Project_ID') or generate_next_project_id()
    created_at = p.get('created_at') or datetime.now().isoformat()
    is_completed = int(p.get('is_completed', 0))
    
    cursor.execute("""
        INSERT INTO projects (
            project_id, project_type, location, area_sqft, number_of_floors,
            material_type, material_cost, labour_cost, labour_count,
            equipment_type, equipment_cost, planned_duration, predicted_cost, is_completed, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id) DO UPDATE SET
            project_type=excluded.project_type,
            location=excluded.location,
            area_sqft=excluded.area_sqft,
            number_of_floors=excluded.number_of_floors,
            material_type=excluded.material_type,
            material_cost=excluded.material_cost,
            labour_cost=excluded.labour_cost,
            labour_count=excluded.labour_count,
            equipment_type=excluded.equipment_type,
            equipment_cost=excluded.equipment_cost,
            planned_duration=excluded.planned_duration,
            predicted_cost=excluded.predicted_cost,
            is_completed=excluded.is_completed
    """, (
        project_id, p['project_type'], p['location'], float(p['area_sqft']), int(p['number_of_floors']),
        p['material_type'], float(p['material_cost']), float(p['labour_cost']), int(p['labour_count']),
        p['equipment_type'], float(p['equipment_cost']), int(p['planned_duration']), float(p['predicted_cost']),
        is_completed, created_at
    ))
    
    conn.commit()
    conn.close()
    return project_id

def update_project_completion(project_id: str, is_completed: int = 1) -> bool:
    """
    Updates the is_completed lifecycle status for a project in SQLite.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE projects SET is_completed = ? WHERE project_id = ?", (int(is_completed), project_id))
    conn.commit()
    conn.close()
    return True

def get_project(project_id: str) -> dict:
    """
    Retrieves project details by project_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def list_projects() -> list:
    """
    Retrieves all projects stored in SQLite ordered by created_at DESC.
    Ensures each Project_ID is returned exactly once.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT project_id, project_type, location, area_sqft, number_of_floors, 
               material_type, material_cost, labour_cost, labour_count, 
               equipment_type, equipment_cost, planned_duration, 
               predicted_cost, is_completed, created_at 
        FROM projects 
        ORDER BY created_at DESC, project_id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    seen = set()
    unique_projects = []
    for r in rows:
        pid = r['project_id']
        if pid not in seen:
            seen.add(pid)
            unique_projects.append(dict(r))
    return unique_projects

def save_monitoring_record(m: dict) -> int:
    """
    Saves a Stage 2 monitoring record to SQLite.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get next sequence number for this project
    cursor.execute("SELECT COUNT(*) FROM monitoring_records WHERE project_id = ?", (m['project_id'],))
    seq = cursor.fetchone()[0] + 1
    
    cursor.execute("""
        INSERT INTO monitoring_records (
            project_id, sequence_number, planned_progress_percent, actual_progress_percent,
            actual_labour_count, material_availability_percent, equipment_availability_percent,
            expected_delay_days, delay_risk, risk_probability, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        m['project_id'], seq, float(m['planned_progress_percent']), float(m['actual_progress_percent']),
        int(m['actual_labour_count']), float(m['material_availability_percent']), float(m['equipment_availability_percent']),
        float(m['expected_delay_days']), m['delay_risk'], float(m['risk_probability']),
        datetime.now().isoformat()
    ))
    
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def get_monitoring_history(project_id: str) -> list:
    """
    Retrieves monitoring history records for a project.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM monitoring_records WHERE project_id = ? ORDER BY sequence_number ASC", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

