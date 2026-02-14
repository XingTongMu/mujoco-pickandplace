import sqlite3
import datetime
import os
import json

class GraspDB:
    def __init__(self, db_path="grasp_experiments.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建实验记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            object_type TEXT,
            object_size TEXT,
            object_pos TEXT,
            object_color TEXT,
            success BOOLEAN,
            final_height REAL,
            notes TEXT
        )
        ''')
        
        conn.commit()
        conn.close()

    def log_experiment(self, obj_type, obj_size, obj_pos, obj_color, success, final_height, notes=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().isoformat()
        
        # 转换数组为字符串存储
        size_str = json.dumps(obj_size) if isinstance(obj_size, list) else str(obj_size)
        pos_str = json.dumps(obj_pos) if isinstance(obj_pos, list) else str(obj_pos)
        color_str = json.dumps(obj_color) if isinstance(obj_color, list) else str(obj_color)
        
        cursor.execute('''
        INSERT INTO experiments (timestamp, object_type, object_size, object_pos, object_color, success, final_height, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, obj_type, size_str, pos_str, color_str, success, final_height, notes))
        
        conn.commit()
        exp_id = cursor.lastrowid
        conn.close()
        print(f"[DB] Logged experiment #{exp_id}: Success={success}, Type={obj_type}")
        return exp_id

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT count(*), sum(success) FROM experiments')
        total, successes = cursor.fetchone()
        
        conn.close()
        
        if total is None or total == 0:
            return 0, 0, 0.0
            
        success_rate = (successes / total) * 100 if successes else 0
        return total, successes, success_rate

if __name__ == "__main__":
    # Test
    db = GraspDB()
    db.log_experiment("box", [0.05, 0.05, 0.05], [0.5, 0, 0.025], [1, 0, 0, 1], True, 0.4)
    t, s, r = db.get_stats()
    print(f"Total: {t}, Success: {s}, Rate: {r:.1f}%")
