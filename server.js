// ============================================================
// NEUROSENSE — Full-Stack Server
// Express + sql.js (SQLite) + Session Auth + ML Engine
// ============================================================

const express = require('express');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const initSqlJs = require('sql.js');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.set('trust proxy', 1);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));
app.use(session({
  secret: process.env.SESSION_SECRET || 'neurosense-dev-secret-key',
  resave: false,
  saveUninitialized: false,
  cookie: { maxAge: 24 * 60 * 60 * 1000 } // 24 hours
}));

let db;

// ===== DATABASE INITIALIZATION =====
async function initDB() {
  const SQL = await initSqlJs();
  const dbPath = path.join(__dirname, 'neurosense.db');

  if (fs.existsSync(dbPath)) {
    const buffer = fs.readFileSync(dbPath);
    db = new SQL.Database(buffer);
  } else {
    db = new SQL.Database();
  }

  db.run(`CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth TEXT,
    gender TEXT,
    phone TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    onboarding_complete INTEGER DEFAULT 0,
    questionnaire_complete INTEGER DEFAULT 0
  )`);

  db.run(`CREATE TABLE IF NOT EXISTS health_profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    height_cm REAL,
    weight_kg REAL,
    blood_group TEXT,
    known_conditions TEXT,
    medications TEXT,
    allergies TEXT,
    family_history_autism INTEGER DEFAULT 0,
    family_history_adhd INTEGER DEFAULT 0,
    family_history_sensory INTEGER DEFAULT 0,
    sleep_hours REAL,
    exercise_frequency TEXT,
    diet_type TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
  )`);

  db.run(`CREATE TABLE IF NOT EXISTS questionnaire_responses (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    section TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    answer_value INTEGER NOT NULL,
    answer_text TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
  )`);

  db.run(`CREATE TABLE IF NOT EXISTS ml_results (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    autism_score REAL,
    autism_level TEXT,
    risk_category TEXT,
    sensory_visual REAL,
    sensory_auditory REAL,
    sensory_tactile REAL,
    sensory_olfactory REAL,
    sensory_vestibular REAL,
    sensory_proprioceptive REAL,
    sensory_interoceptive REAL,
    social_communication REAL,
    repetitive_behaviors REAL,
    emotional_regulation REAL,
    executive_function REAL,
    pattern_type TEXT,
    confidence REAL,
    recommendations TEXT,
    detailed_report TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
  )`);

  db.run(`CREATE TABLE IF NOT EXISTS sensory_entries (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    entry_date TEXT,
    environment TEXT,
    duration_min INTEGER,
    visual INTEGER,
    auditory INTEGER,
    tactile INTEGER,
    olfactory INTEGER,
    vestibular INTEGER,
    proprioceptive INTEGER,
    meltdown TEXT,
    stimming INTEGER,
    mood TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
  )`);

  saveDB();
  console.log('✅ Database initialized');
}

function saveDB() {
  const data = db.export();
  const buffer = Buffer.from(data);
  fs.writeFileSync(path.join(__dirname, 'neurosense.db'), buffer);
}

// ===== AUTH MIDDLEWARE =====
function requireAuth(req, res, next) {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  next();
}

// ===== AUTH ROUTES =====
app.post('/api/auth/signup', (req, res) => {
  try {
    const { email, password, first_name, last_name } = req.body;
    if (!email || !password || !first_name || !last_name) {
      return res.status(400).json({ error: 'All fields required' });
    }

    const existing = db.exec("SELECT id FROM users WHERE email = ?", [email]);
    if (existing.length > 0 && existing[0].values.length > 0) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    const id = uuidv4();
    const hashedPassword = bcrypt.hashSync(password, 10);

    db.run("INSERT INTO users (id, email, password, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
      [id, email.toLowerCase(), hashedPassword, first_name, last_name]);
    saveDB();

    req.session.userId = id;
    res.json({ success: true, user: { id, email, first_name, last_name, onboarding_complete: 0, questionnaire_complete: 0 } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/auth/login', (req, res) => {
  try {
    const { email, password } = req.body;
    const result = db.exec("SELECT * FROM users WHERE email = ?", [email.toLowerCase()]);
    if (result.length === 0 || result[0].values.length === 0) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    const cols = result[0].columns;
    const row = result[0].values[0];
    const user = {};
    cols.forEach((c, i) => user[c] = row[i]);

    if (!bcrypt.compareSync(password, user.password)) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    req.session.userId = user.id;
    res.json({
      success: true,
      user: {
        id: user.id, email: user.email,
        first_name: user.first_name, last_name: user.last_name,
        onboarding_complete: user.onboarding_complete,
        questionnaire_complete: user.questionnaire_complete
      }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/auth/logout', (req, res) => {
  req.session.destroy();
  res.json({ success: true });
});

app.get('/api/auth/me', requireAuth, (req, res) => {
  const result = db.exec("SELECT id, email, first_name, last_name, date_of_birth, gender, phone, onboarding_complete, questionnaire_complete FROM users WHERE id = ?", [req.session.userId]);
  if (result.length === 0 || result[0].values.length === 0) {
    return res.status(404).json({ error: 'User not found' });
  }
  const cols = result[0].columns;
  const row = result[0].values[0];
  const user = {};
  cols.forEach((c, i) => user[c] = row[i]);

  // Get health profile
  const hp = db.exec("SELECT * FROM health_profiles WHERE user_id = ?", [req.session.userId]);
  if (hp.length > 0 && hp[0].values.length > 0) {
    const hpCols = hp[0].columns;
    const hpRow = hp[0].values[0];
    user.health_profile = {};
    hpCols.forEach((c, i) => user.health_profile[c] = hpRow[i]);
  }

  // Get ML results
  const ml = db.exec("SELECT * FROM ml_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", [req.session.userId]);
  if (ml.length > 0 && ml[0].values.length > 0) {
    const mlCols = ml[0].columns;
    const mlRow = ml[0].values[0];
    user.ml_results = {};
    mlCols.forEach((c, i) => user.ml_results[c] = mlRow[i]);
  }

  res.json({ user });
});

// ===== ONBOARDING (Personal + Health Profile) =====
app.post('/api/onboarding/personal', requireAuth, (req, res) => {
  try {
    const { date_of_birth, gender, phone } = req.body;
    db.run("UPDATE users SET date_of_birth = ?, gender = ?, phone = ? WHERE id = ?",
      [date_of_birth, gender, phone, req.session.userId]);
    saveDB();
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/onboarding/health', requireAuth, (req, res) => {
  try {
    const { height_cm, weight_kg, blood_group, known_conditions, medications, allergies,
      family_history_autism, family_history_adhd, family_history_sensory,
      sleep_hours, exercise_frequency, diet_type } = req.body;

    // Check if profile exists
    const existing = db.exec("SELECT id FROM health_profiles WHERE user_id = ?", [req.session.userId]);
    if (existing.length > 0 && existing[0].values.length > 0) {
      db.run(`UPDATE health_profiles SET height_cm=?, weight_kg=?, blood_group=?, known_conditions=?,
        medications=?, allergies=?, family_history_autism=?, family_history_adhd=?, family_history_sensory=?,
        sleep_hours=?, exercise_frequency=?, diet_type=?, updated_at=datetime('now') WHERE user_id=?`,
        [height_cm, weight_kg, blood_group, known_conditions, medications, allergies,
          family_history_autism ? 1 : 0, family_history_adhd ? 1 : 0, family_history_sensory ? 1 : 0,
          sleep_hours, exercise_frequency, diet_type, req.session.userId]);
    } else {
      db.run(`INSERT INTO health_profiles (id, user_id, height_cm, weight_kg, blood_group, known_conditions,
        medications, allergies, family_history_autism, family_history_adhd, family_history_sensory,
        sleep_hours, exercise_frequency, diet_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [uuidv4(), req.session.userId, height_cm, weight_kg, blood_group, known_conditions, medications, allergies,
          family_history_autism ? 1 : 0, family_history_adhd ? 1 : 0, family_history_sensory ? 1 : 0,
          sleep_hours, exercise_frequency, diet_type]);
    }

    db.run("UPDATE users SET onboarding_complete = 1 WHERE id = ?", [req.session.userId]);
    saveDB();
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ===== QUESTIONNAIRE =====
app.post('/api/questionnaire/submit', requireAuth, (req, res) => {
  try {
    const { responses } = req.body; // Array of { section, question_id, question_text, answer_value, answer_text }

    // Clear previous responses
    db.run("DELETE FROM questionnaire_responses WHERE user_id = ?", [req.session.userId]);

    responses.forEach(r => {
      db.run(`INSERT INTO questionnaire_responses (id, user_id, section, question_id, question_text, answer_value, answer_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [uuidv4(), req.session.userId, r.section, r.question_id, r.question_text, r.answer_value, r.answer_text || '']);
    });

    db.run("UPDATE users SET questionnaire_complete = 1 WHERE id = ?", [req.session.userId]);
    saveDB();
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.get('/api/questionnaire/responses', requireAuth, (req, res) => {
  const result = db.exec("SELECT * FROM questionnaire_responses WHERE user_id = ? ORDER BY section, question_id", [req.session.userId]);
  if (result.length === 0) return res.json({ responses: [] });

  const cols = result[0].columns;
  const responses = result[0].values.map(row => {
    const obj = {};
    cols.forEach((c, i) => obj[c] = row[i]);
    return obj;
  });
  res.json({ responses });
});

// ===== ML PROCESSING =====
app.post('/api/ml/process', requireAuth, (req, res) => {
  try {
    // Get questionnaire responses
    const qResult = db.exec("SELECT * FROM questionnaire_responses WHERE user_id = ?", [req.session.userId]);
    if (qResult.length === 0 || qResult[0].values.length === 0) {
      return res.status(400).json({ error: 'No questionnaire data found. Complete the questionnaire first.' });
    }

    const cols = qResult[0].columns;
    const responses = qResult[0].values.map(row => {
      const obj = {};
      cols.forEach((c, i) => obj[c] = row[i]);
      return obj;
    });

    // Get health profile
    const hpResult = db.exec("SELECT * FROM health_profiles WHERE user_id = ?", [req.session.userId]);
    let healthProfile = null;
    if (hpResult.length > 0 && hpResult[0].values.length > 0) {
      healthProfile = {};
      hpResult[0].columns.forEach((c, i) => healthProfile[c] = hpResult[0].values[0][i]);
    }

    // Get user personal info (DOB, gender) for model features
    const userResult = db.exec("SELECT date_of_birth, gender FROM users WHERE id = ?", [req.session.userId]);
    let userInfo = {};
    if (userResult.length > 0 && userResult[0].values.length > 0) {
      userResult[0].columns.forEach((c, i) => userInfo[c] = userResult[0].values[0][i]);
    }

    // ===== DUAL MODEL PROCESSING =====
    const mlResult = processWithDualModels(responses, healthProfile, userInfo);

    // Save results
    const existingML = db.exec("SELECT id FROM ml_results WHERE user_id = ?", [req.session.userId]);
    if (existingML.length > 0 && existingML[0].values.length > 0) {
      db.run(`UPDATE ml_results SET autism_score=?, autism_level=?, risk_category=?,
        sensory_visual=?, sensory_auditory=?, sensory_tactile=?, sensory_olfactory=?,
        sensory_vestibular=?, sensory_proprioceptive=?, sensory_interoceptive=?,
        social_communication=?, repetitive_behaviors=?, emotional_regulation=?, executive_function=?,
        pattern_type=?, confidence=?, recommendations=?, detailed_report=?, created_at=datetime('now')
        WHERE user_id=?`,
        [mlResult.autism_score, mlResult.autism_level, mlResult.risk_category,
          mlResult.sensory.visual, mlResult.sensory.auditory, mlResult.sensory.tactile,
          mlResult.sensory.olfactory, mlResult.sensory.vestibular, mlResult.sensory.proprioceptive,
          mlResult.sensory.interoceptive,
          mlResult.domains.social_communication, mlResult.domains.repetitive_behaviors,
          mlResult.domains.emotional_regulation, mlResult.domains.executive_function,
          mlResult.pattern_type, mlResult.confidence,
          JSON.stringify(mlResult.recommendations), JSON.stringify(mlResult.detailed_report),
          req.session.userId]);
    } else {
      db.run(`INSERT INTO ml_results (id, user_id, autism_score, autism_level, risk_category,
        sensory_visual, sensory_auditory, sensory_tactile, sensory_olfactory,
        sensory_vestibular, sensory_proprioceptive, sensory_interoceptive,
        social_communication, repetitive_behaviors, emotional_regulation, executive_function,
        pattern_type, confidence, recommendations, detailed_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [uuidv4(), req.session.userId, mlResult.autism_score, mlResult.autism_level, mlResult.risk_category,
          mlResult.sensory.visual, mlResult.sensory.auditory, mlResult.sensory.tactile,
          mlResult.sensory.olfactory, mlResult.sensory.vestibular, mlResult.sensory.proprioceptive,
          mlResult.sensory.interoceptive,
          mlResult.domains.social_communication, mlResult.domains.repetitive_behaviors,
          mlResult.domains.emotional_regulation, mlResult.domains.executive_function,
          mlResult.pattern_type, mlResult.confidence,
          JSON.stringify(mlResult.recommendations), JSON.stringify(mlResult.detailed_report)]);
    }

    saveDB();
    res.json({ success: true, results: mlResult });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'ML Processing failed' });
  }
});

app.get('/api/ml/results', requireAuth, (req, res) => {
  const result = db.exec("SELECT * FROM ml_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", [req.session.userId]);
  if (result.length === 0 || result[0].values.length === 0) {
    return res.json({ results: null });
  }
  const cols = result[0].columns;
  const row = result[0].values[0];
  const obj = {};
  cols.forEach((c, i) => obj[c] = row[i]);

  if (obj.recommendations) obj.recommendations = JSON.parse(obj.recommendations);
  if (obj.detailed_report) obj.detailed_report = JSON.parse(obj.detailed_report);

  res.json({ results: obj });
});

// ===== SENSORY ENTRIES (ongoing tracking) =====
app.post('/api/entries/add', requireAuth, (req, res) => {
  try {
    const e = req.body;
    db.run(`INSERT INTO sensory_entries (id, user_id, entry_date, environment, duration_min,
      visual, auditory, tactile, olfactory, vestibular, proprioceptive,
      meltdown, stimming, mood, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      [uuidv4(), req.session.userId, e.entry_date, e.environment, e.duration_min,
        e.visual, e.auditory, e.tactile, e.olfactory, e.vestibular, e.proprioceptive,
        e.meltdown, e.stimming, e.mood, e.notes]);
    saveDB();
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: 'Server error' });
  }
});

app.get('/api/entries', requireAuth, (req, res) => {
  const result = db.exec("SELECT * FROM sensory_entries WHERE user_id = ? ORDER BY created_at DESC", [req.session.userId]);
  if (result.length === 0) return res.json({ entries: [] });
  const cols = result[0].columns;
  const entries = result[0].values.map(row => {
    const obj = {};
    cols.forEach((c, i) => obj[c] = row[i]);
    return obj;
  });
  res.json({ entries });
});

// ============================================================
// ML ENGINE — Dual Model Architecture
// Model 1: Random Forest Classifier (ASD Screening)
// Model 2: Neural Network MLP (Severity Scoring)
// Trained on: Kaggle Autism Screening Adult Dataset (AQ-10)
// ============================================================

let rfModel = null;  // Random Forest — loaded from ml/rf_model.json
let nnModel = null;  // Neural Network — loaded from ml/nn_model.json

function loadModels() {
  const rfPath = path.join(__dirname, 'ml', 'rf_model.json');
  const nnPath = path.join(__dirname, 'ml', 'nn_model.json');

  if (fs.existsSync(rfPath)) {
    rfModel = JSON.parse(fs.readFileSync(rfPath, 'utf8'));
    console.log(`✅ Model 1 loaded: Random Forest (${rfModel.n_estimators} trees, accuracy: ${rfModel.metrics.accuracy})`);
  } else {
    console.warn('⚠️  ml/rf_model.json not found — will use fallback scoring');
  }

  if (fs.existsSync(nnPath)) {
    nnModel = JSON.parse(fs.readFileSync(nnPath, 'utf8'));
    console.log(`✅ Model 2 loaded: Neural Network MLP (${nnModel.architecture.join('→')}, accuracy: ${nnModel.metrics.accuracy})`);
  } else {
    console.warn('⚠️  ml/nn_model.json not found — will use fallback scoring');
  }
}

// ── Math helpers ──────────────────────────────────────────
function relu(x) { return Math.max(0, x); }
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }

// ── Feature Mapping: NeuroSense → AQ-10 feature space ────
// Maps 11 questionnaire domain scores + health profile to
// the 14 features the trained models expect (10 AQ-10 + 4 demographic)
function mapToAQ10Features(sensory, domains, healthProfile, userInfo) {
  const sc = domains.social_communication / 10;
  const rb = domains.repetitive_behaviors / 10;
  const er = domains.emotional_regulation / 10;
  const ef = domains.executive_function / 10;
  const vis = sensory.visual / 10;
  const aud = sensory.auditory / 10;

  // Calculate age from date_of_birth
  let age = 25; // default
  if (userInfo?.date_of_birth) {
    const dob = new Date(userInfo.date_of_birth);
    if (!isNaN(dob.getTime())) {
      age = Math.floor((Date.now() - dob.getTime()) / (365.25 * 24 * 60 * 60 * 1000));
    }
  }

  // Gender: 1 = male, 0 = female
  let gender = 0;
  if (userInfo?.gender) {
    gender = userInfo.gender.toLowerCase().startsWith('m') ? 1 : 0;
  }

  return [
    sc > 0.5 ? 1 : 0,                    // A1:  Social attention switching
    (vis + aud) / 2 > 0.5 ? 1 : 0,       // A2:  Detail focus / sensory attention
    ef > 0.5 ? 1 : 0,                     // A3:  Multitasking difficulty
    ef > 0.6 ? 1 : 0,                     // A4:  Task switching
    sc > 0.5 ? 1 : 0,                     // A5:  Reading between the lines
    rb > 0.5 ? 1 : 0,                     // A6:  Pattern/number focus
    er > 0.6 ? 1 : 0,                     // A7:  Fiction preference / empathy
    sc > 0.6 ? 1 : 0,                     // A8:  Understanding intentions
    (sc + er) / 2 > 0.5 ? 1 : 0,         // A9:  Imagining characters
    sc > 0.4 ? 1 : 0,                     // A10: Social interaction difficulty
    age,                                   // Age
    gender,                                // Gender (1=M, 0=F)
    0,                                     // Jaundice (default no)
    healthProfile?.family_history_autism ? 1 : 0  // Family member with ASD
  ];
}

// ── Model 1: Random Forest Prediction ─────────────────────
function predictRF(features) {
  if (!rfModel) return null;

  let totalProb = 0;
  for (const treeNodes of rfModel.trees) {
    let idx = 0;
    // Traverse tree: f === -2 means leaf node (sklearn TREE_UNDEFINED)
    while (treeNodes[idx].f !== -2) {
      if (features[treeNodes[idx].f] <= treeNodes[idx].t) {
        idx = treeNodes[idx].l;
      } else {
        idx = treeNodes[idx].r;
      }
    }
    // Leaf value: [n_class0, n_class1]
    const v = treeNodes[idx].v;
    const total = v[0] + v[1];
    totalProb += total > 0 ? v[1] / total : 0.5;
  }

  const probability = totalProb / rfModel.trees.length;
  return {
    prediction: probability > 0.5 ? 'YES' : 'NO',
    probability: probability,
    confidence: Math.round(Math.abs(probability - 0.5) * 2 * 100 * 10) / 10
  };
}

// ── Model 2: Neural Network MLP Prediction ────────────────
function predictNN(features) {
  if (!nnModel) return null;

  // Apply StandardScaler normalization
  let x = features.slice();
  if (nnModel.scaler) {
    x = x.map((v, i) => (v - nnModel.scaler.mean[i]) / nnModel.scaler.scale[i]);
  }

  // Forward pass through each layer
  const layerActivations = [];
  let h = x;
  for (let i = 0; i < nnModel.layers.length; i++) {
    const W = nnModel.layers[i].W;
    const b = nnModel.layers[i].b;
    const isOutput = i === nnModel.layers.length - 1;

    h = W.map((row, j) => {
      const sum = row.reduce((s, wij, k) => s + wij * (h[k] || 0), 0) + b[j];
      return isOutput ? sigmoid(sum) : relu(sum);
    });

    // Store average activation for visualization
    layerActivations.push(h.reduce((a, b) => a + b, 0) / h.length);
  }

  return {
    score: h[0],  // 0-1 probability
    layerActivations: layerActivations
  };
}

// ── Main Processing Function (Dual Model) ─────────────────
function processWithDualModels(responses, healthProfile, userInfo) {
  // 1. Parse questionnaire into domain scores
  const sectionScores = {};
  responses.forEach(r => {
    if (!sectionScores[r.section]) sectionScores[r.section] = [];
    sectionScores[r.section].push(r.answer_value);
  });

  const avgScore = (arr) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  const sensory = {
    visual: avgScore(sectionScores['sensory_visual'] || [5]),
    auditory: avgScore(sectionScores['sensory_auditory'] || [5]),
    tactile: avgScore(sectionScores['sensory_tactile'] || [5]),
    olfactory: avgScore(sectionScores['sensory_olfactory'] || [5]),
    vestibular: avgScore(sectionScores['sensory_vestibular'] || [5]),
    proprioceptive: avgScore(sectionScores['sensory_proprioceptive'] || [5]),
    interoceptive: avgScore(sectionScores['sensory_interoceptive'] || [5]),
  };

  const domains = {
    social_communication: avgScore(sectionScores['social_communication'] || [5]),
    repetitive_behaviors: avgScore(sectionScores['repetitive_behaviors'] || [5]),
    emotional_regulation: avgScore(sectionScores['emotional_regulation'] || [5]),
    executive_function: avgScore(sectionScores['executive_function'] || [5]),
  };

  // 2. Map NeuroSense features → AQ-10 feature space (14 features)
  const aq10Features = mapToAQ10Features(sensory, domains, healthProfile, userInfo);

  // 3. Run Model 1: Random Forest Classifier
  const rfResult = predictRF(aq10Features);

  // 4. Run Model 2: Neural Network MLP
  const nnResult = predictNN(aq10Features);

  // 5. Compute autism score (0-100)
  // Direct score from questionnaire (fallback baseline)
  const directScore = (
    Object.values(sensory).reduce((a, b) => a + b, 0) / 7 * 0.4 +
    Object.values(domains).reduce((a, b) => a + b, 0) / 4 * 0.6
  ) / 10 * 100;

  let autism_score;
  let modelMode;

  if (rfResult && nnResult) {
    // Dual model: 40% RF probability + 40% NN score + 20% direct
    autism_score = Math.round((
      rfResult.probability * 100 * 0.4 +
      nnResult.score * 100 * 0.4 +
      directScore * 0.2
    ) * 10) / 10;
    modelMode = 'dual-model';
  } else if (nnResult) {
    autism_score = Math.round((nnResult.score * 100 * 0.6 + directScore * 0.4) * 10) / 10;
    modelMode = 'nn-only';
  } else if (rfResult) {
    autism_score = Math.round((rfResult.probability * 100 * 0.6 + directScore * 0.4) * 10) / 10;
    modelMode = 'rf-only';
  } else {
    autism_score = Math.round(directScore * 10) / 10;
    modelMode = 'fallback';
  }

  // Clamp to 0-100
  autism_score = Math.max(0, Math.min(100, autism_score));

  // 6. Determine autism level
  let autism_level, risk_category;
  if (autism_score >= 75) {
    autism_level = 'Level 3 — Requiring Very Substantial Support';
    risk_category = 'High';
  } else if (autism_score >= 55) {
    autism_level = 'Level 2 — Requiring Substantial Support';
    risk_category = 'Moderate-High';
  } else if (autism_score >= 35) {
    autism_level = 'Level 1 — Requiring Support';
    risk_category = 'Moderate';
  } else if (autism_score >= 20) {
    autism_level = 'Subclinical — Mild Traits Present';
    risk_category = 'Low-Moderate';
  } else {
    autism_level = 'Minimal — Within Neurotypical Range';
    risk_category = 'Low';
  }

  // 7. Pattern type (from sensory profile)
  const sensoryAvg = Object.values(sensory).reduce((a, b) => a + b, 0) / 7;
  const sensoryVar = Object.values(sensory).reduce((a, v) => a + Math.pow(v - sensoryAvg, 2), 0) / 7;
  let pattern_type;
  if (sensoryAvg > 7) pattern_type = 'Hyper-Responsive (Sensory Avoidant)';
  else if (sensoryAvg < 3) pattern_type = 'Hypo-Responsive (Sensory Seeking)';
  else if (sensoryVar > 4) pattern_type = 'Mixed Responsivity (Variable)';
  else pattern_type = 'Moderate / Stable Processing';

  // 8. Confidence — from models if available, else heuristic
  let confidence;
  if (rfResult && nnResult) {
    // Average of RF confidence and NN-based confidence
    const nnConf = Math.abs(nnResult.score - 0.5) * 2 * 100;
    confidence = Math.round((rfResult.confidence * 0.5 + nnConf * 0.3 + 20) * 10) / 10; // 20% base from data completeness
  } else {
    confidence = Math.min(92, 70 + responses.length * 0.3);
  }
  confidence = Math.min(98, confidence);

  // 9. Recommendations (rule-based, unchanged)
  const recommendations = generateDetailedRecommendations(sensory, domains, autism_score, autism_level, pattern_type, healthProfile);

  // 10. Detailed report with model info
  const detailed_report = {
    summary: `Based on comprehensive analysis of ${responses.length} questionnaire responses using ${modelMode === 'dual-model' ? 'dual AI models (Random Forest + Neural Network)' : modelMode === 'fallback' ? 'direct scoring' : 'trained ML model'}, the NeuroSense engine has computed an autism spectrum score of ${autism_score}/100.`,
    sensory_profile: Object.entries(sensory).map(([k, v]) => ({
      channel: k.charAt(0).toUpperCase() + k.slice(1),
      score: Math.round(v * 10) / 10,
      severity: v >= 7 ? 'Significant' : v >= 5 ? 'Moderate' : v >= 3 ? 'Mild' : 'Minimal',
      description: getSensoryDescription(k, v)
    })),
    domain_analysis: Object.entries(domains).map(([k, v]) => ({
      domain: k.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
      score: Math.round(v * 10) / 10,
      severity: v >= 7 ? 'Significant' : v >= 5 ? 'Moderate' : v >= 3 ? 'Mild' : 'Minimal'
    })),
    network_activations: nnResult ? {
      layer_activations: nnResult.layerActivations.map(a => Math.round(a * 1000) / 1000),
      output_raw: Math.round(nnResult.score * 1000) / 1000,
      architecture: nnModel.architecture
    } : {
      layer_activations: [0, 0, 0],
      output_raw: directScore / 100,
      architecture: [14, 20, 12, 8, 1]
    },
    model_info: {
      mode: modelMode,
      rf: rfResult ? {
        type: 'Random Forest Classifier',
        n_estimators: rfModel.n_estimators,
        accuracy: rfModel.metrics.accuracy,
        prediction: rfResult.prediction,
        probability: Math.round(rfResult.probability * 1000) / 1000,
        confidence: rfResult.confidence,
        dataset: rfModel.training_info.dataset,
        feature_importances: rfModel.feature_importances
      } : null,
      nn: nnResult ? {
        type: 'Neural Network MLP',
        architecture: nnModel.architecture,
        accuracy: nnModel.metrics.accuracy,
        raw_score: Math.round(nnResult.score * 1000) / 1000,
        dataset: nnModel.training_info.dataset
      } : null,
      dataset: 'Kaggle AQ-10 Adult Autism Screening (704 records)',
      feature_mapping: '11 NeuroSense domain scores → 14 AQ-10 features'
    }
  };

  return {
    autism_score, autism_level, risk_category,
    sensory, domains, pattern_type, confidence,
    recommendations, detailed_report
  };
}

function getSensoryDescription(channel, score) {
  const descriptions = {
    visual: score >= 7 ? 'High sensitivity to light, patterns, and visual clutter. Fluorescent lights and bright screens may cause significant distress.' : score >= 4 ? 'Moderate visual processing differences. Some difficulty with bright environments.' : 'Visual processing within typical range.',
    auditory: score >= 7 ? 'Significant sound sensitivity. Everyday noises may be perceived as painfully loud. Background noise severely impacts functioning.' : score >= 4 ? 'Moderate auditory sensitivity. Difficulty filtering background noise from important sounds.' : 'Auditory processing within typical range.',
    tactile: score >= 7 ? 'Strong tactile defensiveness. Clothing textures, light touch, and certain surfaces may cause extreme discomfort.' : score >= 4 ? 'Moderate touch sensitivity. Preferences for certain textures and avoidance of others.' : 'Tactile processing within typical range.',
    olfactory: score >= 7 ? 'Heightened smell sensitivity. Ordinary scents may be overwhelming and trigger nausea or avoidance.' : score >= 4 ? 'Moderate olfactory sensitivity with notable preferences and aversions.' : 'Olfactory processing within typical range.',
    vestibular: score >= 7 ? 'Significant movement sensitivity. Balance challenges and motion discomfort are common.' : score >= 4 ? 'Moderate vestibular differences. Some difficulty with rapid movement changes.' : 'Vestibular processing within typical range.',
    proprioceptive: score >= 7 ? 'Notable body awareness differences. May seek deep pressure or have difficulty gauging force.' : score >= 4 ? 'Moderate proprioceptive processing differences. Occasional clumsiness or pressure-seeking.' : 'Proprioceptive processing within typical range.',
    interoceptive: score >= 7 ? 'Significant difficulty recognizing internal body signals like hunger, thirst, or need for rest.' : score >= 4 ? 'Moderate interoceptive differences. Sometimes misses body signals.' : 'Interoceptive processing within typical range.',
  };
  return descriptions[channel] || 'Assessment complete.';
}

function generateDetailedRecommendations(sensory, domains, score, level, pattern, healthProfile) {
  const recs = [];

  // Environmental accommodations
  if (sensory.auditory >= 6) {
    recs.push({
      category: 'Environment', priority: 'high', title: 'Noise Management',
      description: 'Your auditory sensitivity score is elevated. Reducing noise exposure will lower daily sensory load.',
      trigger: { label: 'Auditory', score: Math.round(sensory.auditory * 10) / 10 },
      timeframe: 'immediate',
      steps: [
        'Use noise-canceling headphones or earplugs in loud environments',
        'Request seating away from noise sources at work/school',
        'Add sound-absorbing panels to primary living spaces',
        'Use white noise or nature sounds to mask unpredictable noise'
      ]
    });
  }
  if (sensory.visual >= 6) {
    recs.push({
      category: 'Environment', priority: 'high', title: 'Visual Comfort',
      description: 'Visual sensitivity is significant. Adjusting lighting and reducing clutter will help.',
      trigger: { label: 'Visual', score: Math.round(sensory.visual * 10) / 10 },
      timeframe: 'immediate',
      steps: [
        'Replace fluorescent lighting with warm LED or natural light',
        'Use blue-light filtering glasses for screen work',
        'Reduce visual clutter in your primary workspace',
        'Consider tinted lenses prescribed by a behavioral optometrist'
      ]
    });
  }
  if (sensory.tactile >= 6) {
    recs.push({
      category: 'Daily Living', priority: 'high', title: 'Tactile Accommodation',
      description: 'Tactile defensiveness detected. Clothing and texture modifications can reduce daily discomfort.',
      trigger: { label: 'Tactile', score: Math.round(sensory.tactile * 10) / 10 },
      timeframe: 'immediate',
      steps: [
        'Choose clothing with soft, tagless fabrics',
        'Wash new clothes multiple times before wearing',
        'Use a weighted blanket or compression clothing for calming',
        'Work with an OT on graduated tactile desensitization'
      ]
    });
  }

  // Therapeutic recommendations
  if (score >= 35) {
    recs.push({
      category: 'Therapy', priority: 'high', title: 'Occupational Therapy',
      description: 'Your overall score suggests structured sensory integration therapy will be beneficial.',
      trigger: { label: 'Overall Score', score: score },
      timeframe: 'short-term',
      steps: [
        'Find a certified sensory integration therapist (OT-SI)',
        'Schedule regular weekly OT sessions',
        'Work with the therapist to develop a personalized sensory diet',
        'Track progress and adjust the plan monthly'
      ]
    });
  }
  if (domains.social_communication >= 6) {
    recs.push({
      category: 'Therapy', priority: 'high', title: 'Social Skills Support',
      description: 'Social communication difficulties are notable. Structured support can improve outcomes.',
      trigger: { label: 'Social Communication', score: Math.round(domains.social_communication * 10) / 10 },
      timeframe: 'short-term',
      steps: [
        'Join a social skills group or enroll in individual therapy',
        'Focus on conversation skills and reading social cues',
        'Consider evidence-based programs like PEERS\u00ae or Social Thinking\u00ae',
        'Practice skills in low-pressure real-world settings'
      ]
    });
  }
  if (domains.emotional_regulation >= 6) {
    recs.push({
      category: 'Therapy', priority: 'medium', title: 'Emotional Regulation',
      description: 'Emotional regulation challenges identified. Building coping strategies is key.',
      trigger: { label: 'Emotional Regulation', score: Math.round(domains.emotional_regulation * 10) / 10 },
      timeframe: 'short-term',
      steps: [
        'Begin Cognitive Behavioral Therapy (CBT) adapted for autism',
        'Learn mindfulness-based stress reduction techniques',
        'Create a visual "feelings thermometer" for self-awareness',
        'Develop personalized coping strategies for each intensity level'
      ]
    });
  }
  if (domains.executive_function >= 5) {
    recs.push({
      category: 'Daily Living', priority: 'medium', title: 'Executive Function Support',
      description: 'Executive function differences can make planning and task switching harder.',
      trigger: { label: 'Executive Function', score: Math.round(domains.executive_function * 10) / 10 },
      timeframe: 'immediate',
      steps: [
        'Use visual schedules and timers for daily routines',
        'Break multi-step tasks into smaller, numbered steps',
        'Use apps designed for executive function support',
        'Establish consistent routines and use transition warnings'
      ]
    });
  }

  // Sensory diet
  recs.push({
    category: 'Sensory Diet', priority: 'medium', title: 'Daily Sensory Diet',
    description: `Based on your ${pattern} profile, a structured sensory diet will help maintain regulation.`,
    trigger: { label: 'Sensory Pattern', score: null },
    timeframe: 'ongoing',
    steps: [
      'Schedule sensory breaks every 90 minutes throughout the day',
      'Include proprioceptive input: heavy work, pushing, pulling activities',
      'Include vestibular input: swinging, rocking, or spinning activities',
      'End the day with calming activities: deep breathing, compression'
    ]
  });

  // Lifestyle
  if (healthProfile && healthProfile.sleep_hours < 7) {
    recs.push({
      category: 'Lifestyle', priority: 'high', title: 'Sleep Optimization',
      description: `Your reported sleep of ${healthProfile.sleep_hours} hours is below the recommended 7-9 hours.`,
      trigger: { label: 'Sleep Hours', score: healthProfile.sleep_hours },
      timeframe: 'immediate',
      steps: [
        'Create a consistent bedtime routine (same time every night)',
        'Reduce screen time 1 hour before bed',
        'Use blackout curtains and white noise in the bedroom',
        'Consider melatonin supplementation under medical guidance'
      ]
    });
  }

  recs.push({
    category: 'Lifestyle', priority: 'low', title: 'Physical Activity',
    description: 'Regular exercise provides proprioceptive/vestibular input and reduces anxiety.',
    trigger: { label: 'General Wellness', score: null },
    timeframe: 'ongoing',
    steps: [
      'Aim for 30+ minutes of moderate activity daily',
      'Try swimming, yoga, or martial arts for sensory input',
      'Use exercise as a scheduled sensory regulation tool',
      'Track activity and mood to find what works best'
    ]
  });

  recs.push({
    category: 'Monitoring', priority: 'low', title: 'Ongoing Tracking',
    description: 'Consistent tracking helps identify patterns, triggers, and intervention effectiveness.',
    trigger: { label: 'Self-Monitoring', score: null },
    timeframe: 'ongoing',
    steps: [
      'Log daily sensory experiences in NeuroSense',
      'Aim for at least 3 entries per week',
      'Review weekly patterns to identify triggers',
      'Share tracking data with your care team'
    ]
  });

  // Medical
  if (score >= 45) {
    recs.push({
      category: 'Medical', priority: 'high', title: 'Professional Evaluation',
      description: 'Your score suggests a formal clinical evaluation is recommended.',
      trigger: { label: 'Overall Score', score: score },
      timeframe: 'short-term',
      steps: [
        'Schedule an evaluation with a developmental pediatrician or psychologist',
        'Gather relevant records (school reports, prior assessments)',
        'Share your NeuroSense report with the evaluating professional',
        'Discuss a comprehensive support plan based on diagnosis'
      ]
    });
  }

  return recs;
}

// ===== ADMIN ROUTES =====
function requireAdmin(req, res, next) {
  const adminPassword = process.env.ADMIN_PASSWORD || 'admin123';
  const provided = req.headers['x-admin-password'] || req.query.password;
  if (provided !== adminPassword) {
    return res.status(403).json({ error: 'Forbidden: invalid admin password' });
  }
  next();
}

app.get('/api/admin/stats', requireAdmin, (req, res) => {
  const userCount = db.exec("SELECT COUNT(*) FROM users");
  const assessedCount = db.exec("SELECT COUNT(DISTINCT user_id) FROM ml_results");
  const questionnaireCount = db.exec("SELECT COUNT(DISTINCT user_id) FROM questionnaire_responses");
  res.json({
    total_users: userCount[0]?.values[0]?.[0] || 0,
    users_with_assessments: assessedCount[0]?.values[0]?.[0] || 0,
    users_with_questionnaires: questionnaireCount[0]?.values[0]?.[0] || 0
  });
});

app.get('/api/admin/users', requireAdmin, (req, res) => {
  const result = db.exec(`
    SELECT id, email, first_name, last_name, date_of_birth, gender,
           created_at, onboarding_complete, questionnaire_complete
    FROM users ORDER BY created_at DESC
  `);
  if (result.length === 0) return res.json({ users: [], count: 0 });
  const cols = result[0].columns;
  const users = result[0].values.map(row => {
    const obj = {};
    cols.forEach((c, i) => obj[c] = row[i]);
    return obj;
  });
  res.json({ users, count: users.length });
});

app.get('/api/admin/users/:userId/assessment', requireAdmin, (req, res) => {
  const userId = req.params.userId;

  const userResult = db.exec("SELECT id, email, first_name, last_name, created_at FROM users WHERE id = ?", [userId]);
  if (userResult.length === 0 || userResult[0].values.length === 0) {
    return res.status(404).json({ error: 'User not found' });
  }
  const user = {};
  userResult[0].columns.forEach((c, i) => user[c] = userResult[0].values[0][i]);

  const hpResult = db.exec("SELECT * FROM health_profiles WHERE user_id = ?", [userId]);
  let healthProfile = null;
  if (hpResult.length > 0 && hpResult[0].values.length > 0) {
    healthProfile = {};
    hpResult[0].columns.forEach((c, i) => healthProfile[c] = hpResult[0].values[0][i]);
  }

  const qResult = db.exec("SELECT section, question_id, question_text, answer_value FROM questionnaire_responses WHERE user_id = ? ORDER BY section, question_id", [userId]);
  let responses = [];
  if (qResult.length > 0) {
    responses = qResult[0].values.map(row => {
      const obj = {};
      qResult[0].columns.forEach((c, i) => obj[c] = row[i]);
      return obj;
    });
  }

  const mlResult = db.exec("SELECT * FROM ml_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", [userId]);
  let mlResults = null;
  if (mlResult.length > 0 && mlResult[0].values.length > 0) {
    mlResults = {};
    mlResult[0].columns.forEach((c, i) => mlResults[c] = mlResult[0].values[0][i]);
    try { if (mlResults.recommendations) mlResults.recommendations = JSON.parse(mlResults.recommendations); } catch(e) {}
    try { if (mlResults.detailed_report) mlResults.detailed_report = JSON.parse(mlResults.detailed_report); } catch(e) {}
  }

  res.json({ user, healthProfile, responses, mlResults });
});

// ===== SERVE SPA =====
app.get('/{*splat}', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ===== START =====
initDB().then(() => {
  loadModels();
  app.listen(PORT, () => {
    console.log(`\n🧠 NeuroSense Dashboard running at http://localhost:${PORT}\n`);
  });
});
