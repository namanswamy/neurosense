// ============================================================
// NEUROSENSE — Full-Stack Server
// Express + Turso/libSQL (Cloud SQLite) + Session Auth + ML Engine
// ============================================================

const express = require('express');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const { createClient } = require('@libsql/client');
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
  // Use Turso cloud DB if URL is set, otherwise local SQLite file
  if (process.env.TURSO_DATABASE_URL) {
    db = createClient({
      url: process.env.TURSO_DATABASE_URL,
      authToken: process.env.TURSO_AUTH_TOKEN,
    });
    console.log('☁️  Connected to Turso cloud database');
  } else {
    db = createClient({
      url: 'file:' + path.join(__dirname, 'neurosense.db'),
    });
    console.log('📁 Using local SQLite database');
  }

  await db.execute(`CREATE TABLE IF NOT EXISTS users (
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

  await db.execute(`CREATE TABLE IF NOT EXISTS health_profiles (
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

  await db.execute(`CREATE TABLE IF NOT EXISTS questionnaire_responses (
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

  await db.execute(`CREATE TABLE IF NOT EXISTS ml_results (
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

  await db.execute(`CREATE TABLE IF NOT EXISTS sensory_entries (
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

  console.log('✅ Database initialized');
}

// Helper: convert libsql result rows to array of objects
function rowsToObjects(result) {
  return result.rows.map(row => {
    const obj = {};
    result.columns.forEach((col, i) => obj[col] = row[i]);
    return obj;
  });
}

// ===== AUTH MIDDLEWARE =====
function requireAuth(req, res, next) {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  next();
}

// ===== AUTH ROUTES =====
app.post('/api/auth/signup', async (req, res) => {
  try {
    const { email, password, first_name, last_name } = req.body;
    if (!email || !password || !first_name || !last_name) {
      return res.status(400).json({ error: 'All fields required' });
    }

    const existing = await db.execute({ sql: "SELECT id FROM users WHERE email = ?", args: [email] });
    if (existing.rows.length > 0) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    const id = uuidv4();
    const hashedPassword = bcrypt.hashSync(password, 10);

    await db.execute({ sql: "INSERT INTO users (id, email, password, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
      args: [id, email.toLowerCase(), hashedPassword, first_name, last_name] });

    req.session.userId = id;
    res.json({ success: true, user: { id, email, first_name, last_name, onboarding_complete: 0, questionnaire_complete: 0 } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const result = await db.execute({ sql: "SELECT * FROM users WHERE email = ?", args: [email.toLowerCase()] });
    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    const user = {};
    result.columns.forEach((c, i) => user[c] = result.rows[0][i]);

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

app.get('/api/auth/me', requireAuth, async (req, res) => {
  const result = await db.execute({ sql: "SELECT id, email, first_name, last_name, date_of_birth, gender, phone, onboarding_complete, questionnaire_complete FROM users WHERE id = ?", args: [req.session.userId] });
  if (result.rows.length === 0) {
    return res.status(404).json({ error: 'User not found' });
  }
  const user = {};
  result.columns.forEach((c, i) => user[c] = result.rows[0][i]);

  // Get health profile
  const hp = await db.execute({ sql: "SELECT * FROM health_profiles WHERE user_id = ?", args: [req.session.userId] });
  if (hp.rows.length > 0) {
    user.health_profile = {};
    hp.columns.forEach((c, i) => user.health_profile[c] = hp.rows[0][i]);
  }

  // Get ML results
  const ml = await db.execute({ sql: "SELECT * FROM ml_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", args: [req.session.userId] });
  if (ml.rows.length > 0) {
    user.ml_results = {};
    ml.columns.forEach((c, i) => user.ml_results[c] = ml.rows[0][i]);
  }

  res.json({ user });
});

// ===== ONBOARDING (Personal + Health Profile) =====
app.post('/api/onboarding/personal', requireAuth, async (req, res) => {
  try {
    const { date_of_birth, gender, phone } = req.body;
    await db.execute({ sql: "UPDATE users SET date_of_birth = ?, gender = ?, phone = ? WHERE id = ?",
      args: [date_of_birth, gender, phone, req.session.userId] });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: 'Server error' });
  }
});

app.post('/api/onboarding/health', requireAuth, async (req, res) => {
  try {
    const { height_cm, weight_kg, blood_group, known_conditions, medications, allergies,
      family_history_autism, family_history_adhd, family_history_sensory,
      sleep_hours, exercise_frequency, diet_type } = req.body;

    const existing = await db.execute({ sql: "SELECT id FROM health_profiles WHERE user_id = ?", args: [req.session.userId] });
    if (existing.rows.length > 0) {
      await db.execute({ sql: `UPDATE health_profiles SET height_cm=?, weight_kg=?, blood_group=?, known_conditions=?,
        medications=?, allergies=?, family_history_autism=?, family_history_adhd=?, family_history_sensory=?,
        sleep_hours=?, exercise_frequency=?, diet_type=?, updated_at=datetime('now') WHERE user_id=?`,
        args: [height_cm, weight_kg, blood_group, known_conditions, medications, allergies,
          family_history_autism ? 1 : 0, family_history_adhd ? 1 : 0, family_history_sensory ? 1 : 0,
          sleep_hours, exercise_frequency, diet_type, req.session.userId] });
    } else {
      await db.execute({ sql: `INSERT INTO health_profiles (id, user_id, height_cm, weight_kg, blood_group, known_conditions,
        medications, allergies, family_history_autism, family_history_adhd, family_history_sensory,
        sleep_hours, exercise_frequency, diet_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [uuidv4(), req.session.userId, height_cm, weight_kg, blood_group, known_conditions, medications, allergies,
          family_history_autism ? 1 : 0, family_history_adhd ? 1 : 0, family_history_sensory ? 1 : 0,
          sleep_hours, exercise_frequency, diet_type] });
    }

    await db.execute({ sql: "UPDATE users SET onboarding_complete = 1 WHERE id = ?", args: [req.session.userId] });
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ===== QUESTIONNAIRE =====
app.post('/api/questionnaire/submit', requireAuth, async (req, res) => {
  try {
    const { responses } = req.body;

    await db.execute({ sql: "DELETE FROM questionnaire_responses WHERE user_id = ?", args: [req.session.userId] });

    for (const r of responses) {
      await db.execute({ sql: `INSERT INTO questionnaire_responses (id, user_id, section, question_id, question_text, answer_value, answer_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)`,
        args: [uuidv4(), req.session.userId, r.section, r.question_id, r.question_text, r.answer_value, r.answer_text || ''] });
    }

    await db.execute({ sql: "UPDATE users SET questionnaire_complete = 1 WHERE id = ?", args: [req.session.userId] });
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.get('/api/questionnaire/responses', requireAuth, async (req, res) => {
  const result = await db.execute({ sql: "SELECT * FROM questionnaire_responses WHERE user_id = ? ORDER BY section, question_id", args: [req.session.userId] });
  if (result.rows.length === 0) return res.json({ responses: [] });
  res.json({ responses: rowsToObjects(result) });
});

// ===== ML PROCESSING =====
app.post('/api/ml/process', requireAuth, async (req, res) => {
  try {
    const qResult = await db.execute({ sql: "SELECT * FROM questionnaire_responses WHERE user_id = ?", args: [req.session.userId] });
    if (qResult.rows.length === 0) {
      return res.status(400).json({ error: 'No questionnaire data found. Complete the questionnaire first.' });
    }

    const responses = rowsToObjects(qResult);

    const hpResult = await db.execute({ sql: "SELECT * FROM health_profiles WHERE user_id = ?", args: [req.session.userId] });
    let healthProfile = null;
    if (hpResult.rows.length > 0) {
      healthProfile = {};
      hpResult.columns.forEach((c, i) => healthProfile[c] = hpResult.rows[0][i]);
    }

    const userResult = await db.execute({ sql: "SELECT date_of_birth, gender FROM users WHERE id = ?", args: [req.session.userId] });
    let userInfo = {};
    if (userResult.rows.length > 0) {
      userResult.columns.forEach((c, i) => userInfo[c] = userResult.rows[0][i]);
    }

    // ===== DUAL MODEL PROCESSING =====
    const mlResult = processWithDualModels(responses, healthProfile, userInfo);

    // Save results
    const existingML = await db.execute({ sql: "SELECT id FROM ml_results WHERE user_id = ?", args: [req.session.userId] });
    if (existingML.rows.length > 0) {
      await db.execute({ sql: `UPDATE ml_results SET autism_score=?, autism_level=?, risk_category=?,
        sensory_visual=?, sensory_auditory=?, sensory_tactile=?, sensory_olfactory=?,
        sensory_vestibular=?, sensory_proprioceptive=?, sensory_interoceptive=?,
        social_communication=?, repetitive_behaviors=?, emotional_regulation=?, executive_function=?,
        pattern_type=?, confidence=?, recommendations=?, detailed_report=?, created_at=datetime('now')
        WHERE user_id=?`,
        args: [mlResult.autism_score, mlResult.autism_level, mlResult.risk_category,
          mlResult.sensory.visual, mlResult.sensory.auditory, mlResult.sensory.tactile,
          mlResult.sensory.olfactory, mlResult.sensory.vestibular, mlResult.sensory.proprioceptive,
          mlResult.sensory.interoceptive,
          mlResult.domains.social_communication, mlResult.domains.repetitive_behaviors,
          mlResult.domains.emotional_regulation, mlResult.domains.executive_function,
          mlResult.pattern_type, mlResult.confidence,
          JSON.stringify(mlResult.recommendations), JSON.stringify(mlResult.detailed_report),
          req.session.userId] });
    } else {
      await db.execute({ sql: `INSERT INTO ml_results (id, user_id, autism_score, autism_level, risk_category,
        sensory_visual, sensory_auditory, sensory_tactile, sensory_olfactory,
        sensory_vestibular, sensory_proprioceptive, sensory_interoceptive,
        social_communication, repetitive_behaviors, emotional_regulation, executive_function,
        pattern_type, confidence, recommendations, detailed_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [uuidv4(), req.session.userId, mlResult.autism_score, mlResult.autism_level, mlResult.risk_category,
          mlResult.sensory.visual, mlResult.sensory.auditory, mlResult.sensory.tactile,
          mlResult.sensory.olfactory, mlResult.sensory.vestibular, mlResult.sensory.proprioceptive,
          mlResult.sensory.interoceptive,
          mlResult.domains.social_communication, mlResult.domains.repetitive_behaviors,
          mlResult.domains.emotional_regulation, mlResult.domains.executive_function,
          mlResult.pattern_type, mlResult.confidence,
          JSON.stringify(mlResult.recommendations), JSON.stringify(mlResult.detailed_report)] });
    }

    res.json({ success: true, results: mlResult });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'ML Processing failed' });
  }
});

app.get('/api/ml/results', requireAuth, async (req, res) => {
  const result = await db.execute({ sql: "SELECT * FROM ml_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", args: [req.session.userId] });
  if (result.rows.length === 0) {
    return res.json({ results: null });
  }
  const obj = {};
  result.columns.forEach((c, i) => obj[c] = result.rows[0][i]);

  if (obj.recommendations) obj.recommendations = JSON.parse(obj.recommendations);
  if (obj.detailed_report) obj.detailed_report = JSON.parse(obj.detailed_report);

  res.json({ results: obj });
});

// ===== SENSORY ENTRIES (ongoing tracking) =====
app.post('/api/entries/add', requireAuth, async (req, res) => {
  try {
    const e = req.body;
    await db.execute({ sql: `INSERT INTO sensory_entries (id, user_id, entry_date, environment, duration_min,
      visual, auditory, tactile, olfactory, vestibular, proprioceptive,
      meltdown, stimming, mood, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      args: [uuidv4(), req.session.userId, e.entry_date, e.environment, e.duration_min,
        e.visual, e.auditory, e.tactile, e.olfactory, e.vestibular, e.proprioceptive,
        e.meltdown, e.stimming, e.mood, e.notes] });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: 'Server error' });
  }
});

app.get('/api/entries', requireAuth, async (req, res) => {
  const result = await db.execute({ sql: "SELECT * FROM sensory_entries WHERE user_id = ? ORDER BY created_at DESC", args: [req.session.userId] });
  if (result.rows.length === 0) return res.json({ entries: [] });
  res.json({ entries: rowsToObjects(result) });
});

// ============================================================
// ML ENGINE — 4-Model Ensemble Architecture
// Model 1: Random Forest Classifier
// Model 2: XGBoost Classifier
// Model 3: SVM (RBF Kernel)
// Model 4: LightGBM Classifier
// Trained on: Kaggle Autism Screening Adult Dataset (AQ-10)
// ============================================================

let rfModel = null;   // Random Forest — loaded from ml/rf_model.json
let xgbModel = null;  // XGBoost — loaded from ml/xgb_model.json
let svmModel = null;  // SVM — loaded from ml/svm_model.json
let lgbModel = null;  // LightGBM — loaded from ml/lgb_model.json

function loadModels() {
  const models = [
    { name: 'Random Forest', file: 'rf_model.json', setter: (m) => rfModel = m, info: (m) => `${m.n_estimators} trees` },
    { name: 'XGBoost', file: 'xgb_model.json', setter: (m) => xgbModel = m, info: (m) => `${m.n_estimators} trees` },
    { name: 'SVM (RBF)', file: 'svm_model.json', setter: (m) => svmModel = m, info: (m) => `${m.support_vectors.length} support vectors` },
    { name: 'LightGBM', file: 'lgb_model.json', setter: (m) => lgbModel = m, info: (m) => `${m.n_estimators} trees` },
  ];

  let loaded = 0;
  for (const { name, file, setter, info } of models) {
    const filePath = path.join(__dirname, 'ml', file);
    if (fs.existsSync(filePath)) {
      const model = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      setter(model);
      console.log(`✅ Model loaded: ${name} (${info(model)}, accuracy: ${model.metrics.accuracy})`);
      loaded++;
    } else {
      console.warn(`⚠️  ml/${file} not found — ${name} unavailable`);
    }
  }

  if (loaded === 0) {
    console.warn('⚠️  No ML models found — will use fallback scoring only');
  } else {
    console.log(`✅ ${loaded}/4 models loaded for ensemble scoring`);
  }
}

// ── Math helpers ──────────────────────────────────────────
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }

// ── Feature Mapping: NeuroSense → AQ-10 feature space ────
function mapToAQ10Features(sensory, domains, healthProfile, userInfo) {
  const sc = domains.social_communication / 10;
  const rb = domains.repetitive_behaviors / 10;
  const er = domains.emotional_regulation / 10;
  const ef = domains.executive_function / 10;
  const vis = sensory.visual / 10;
  const aud = sensory.auditory / 10;

  let age = 25;
  if (userInfo?.date_of_birth) {
    const dob = new Date(userInfo.date_of_birth);
    if (!isNaN(dob.getTime())) {
      age = Math.floor((Date.now() - dob.getTime()) / (365.25 * 24 * 60 * 60 * 1000));
    }
  }

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
    while (treeNodes[idx].f !== -2) {
      if (features[treeNodes[idx].f] <= treeNodes[idx].t) {
        idx = treeNodes[idx].l;
      } else {
        idx = treeNodes[idx].r;
      }
    }
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

// ── Model 2: XGBoost Prediction ───────────────────────────
function traverseXGBTree(node, features) {
  // Leaf node
  if (node.leaf !== undefined) return node.leaf;
  // Split node: "split" is feature name like "f0", "f1", etc.
  const featureIdx = parseInt(node.split.replace('f', ''));
  const val = features[featureIdx] || 0;
  // XGBoost uses "yes" for <= threshold, "no" for > threshold
  if (val < node.split_condition) {
    // Go to "yes" child
    const yesIdx = node.yes;
    for (const child of node.children) {
      if (child.nodeid === yesIdx) return traverseXGBTree(child, features);
    }
  } else {
    // Go to "no" child
    const noIdx = node.no;
    for (const child of node.children) {
      if (child.nodeid === noIdx) return traverseXGBTree(child, features);
    }
  }
  return 0;
}

function predictXGB(features) {
  if (!xgbModel) return null;

  let sum = 0;
  for (const tree of xgbModel.trees) {
    sum += traverseXGBTree(tree, features);
  }

  // Apply sigmoid to get probability (binary classification with logistic objective)
  const probability = sigmoid(sum);
  return {
    prediction: probability > 0.5 ? 'YES' : 'NO',
    probability: probability,
    confidence: Math.round(Math.abs(probability - 0.5) * 2 * 100 * 10) / 10
  };
}

// ── Model 3: SVM (RBF Kernel) Prediction ──────────────────
function predictSVM(features) {
  if (!svmModel) return null;

  // Standardize input using stored scaler
  const x = features.map((v, i) => (v - svmModel.scaler.mean[i]) / svmModel.scaler.scale[i]);

  // RBF kernel: K(x, sv) = exp(-gamma * ||x - sv||^2)
  const gamma = svmModel.gamma;
  const supportVectors = svmModel.support_vectors;
  const dualCoef = svmModel.dual_coef[0]; // shape: [1, n_support_vectors]

  let decision = svmModel.intercept[0];
  for (let i = 0; i < supportVectors.length; i++) {
    let sqDist = 0;
    for (let j = 0; j < x.length; j++) {
      const diff = x[j] - supportVectors[i][j];
      sqDist += diff * diff;
    }
    const kernelVal = Math.exp(-gamma * sqDist);
    decision += dualCoef[i] * kernelVal;
  }

  // Convert decision value to probability using Platt scaling approximation
  const probability = sigmoid(decision);
  return {
    prediction: decision > 0 ? 'YES' : 'NO',
    probability: probability,
    confidence: Math.round(Math.min(Math.abs(decision), 3) / 3 * 100 * 10) / 10
  };
}

// ── Model 4: LightGBM Prediction ─────────────────────────
function traverseLGBTree(node, features) {
  // Leaf node
  if (node.leaf_value !== undefined) return node.leaf_value;

  // Split node
  const featureIdx = node.split_feature;
  const val = features[featureIdx] || 0;
  const threshold = node.threshold;

  // LightGBM default: left child for <= threshold
  const decision_type = node.decision_type || '<=';
  let goLeft;
  if (decision_type === '<=') {
    goLeft = val <= threshold;
  } else {
    goLeft = val < threshold;
  }

  if (goLeft) {
    return traverseLGBTree(node.left_child, features);
  } else {
    return traverseLGBTree(node.right_child, features);
  }
}

function predictLGB(features) {
  if (!lgbModel) return null;

  let sum = 0;
  for (const treeInfo of lgbModel.trees) {
    sum += traverseLGBTree(treeInfo.tree_structure, features);
  }

  // Apply sigmoid to get probability (binary classification)
  const probability = sigmoid(sum);
  return {
    prediction: probability > 0.5 ? 'YES' : 'NO',
    probability: probability,
    confidence: Math.round(Math.abs(probability - 0.5) * 2 * 100 * 10) / 10
  };
}

// ── Main Processing Function (4-Model Ensemble) ──────────
function processWithDualModels(responses, healthProfile, userInfo) {
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

  const aq10Features = mapToAQ10Features(sensory, domains, healthProfile, userInfo);

  // Run all 4 models
  const rfResult = predictRF(aq10Features);
  const xgbResult = predictXGB(aq10Features);
  const svmResult = predictSVM(aq10Features);
  const lgbResult = predictLGB(aq10Features);

  const directScore = (
    Object.values(sensory).reduce((a, b) => a + b, 0) / 7 * 0.4 +
    Object.values(domains).reduce((a, b) => a + b, 0) / 4 * 0.6
  ) / 10 * 100;

  // Soft-voting ensemble: collect all available model probabilities
  const modelResults = [
    { result: rfResult, weight: 0.25, name: 'rf' },
    { result: xgbResult, weight: 0.25, name: 'xgb' },
    { result: svmResult, weight: 0.20, name: 'svm' },
    { result: lgbResult, weight: 0.20, name: 'lgb' },
  ].filter(m => m.result !== null);

  let autism_score;
  let modelMode;

  if (modelResults.length >= 2) {
    // Redistribute weights proportionally among available models
    const totalMLWeight = modelResults.reduce((s, m) => s + m.weight, 0);
    const mlWeight = 0.9; // 90% ML, 10% direct
    let mlScore = 0;
    for (const m of modelResults) {
      const normalizedWeight = (m.weight / totalMLWeight) * mlWeight;
      mlScore += m.result.probability * 100 * normalizedWeight;
    }
    autism_score = Math.round((mlScore + directScore * 0.1) * 10) / 10;
    modelMode = modelResults.length === 4 ? 'quad-model' : `ensemble-${modelResults.length}`;
  } else if (modelResults.length === 1) {
    autism_score = Math.round((modelResults[0].result.probability * 100 * 0.6 + directScore * 0.4) * 10) / 10;
    modelMode = `${modelResults[0].name}-only`;
  } else {
    autism_score = Math.round(directScore * 10) / 10;
    modelMode = 'fallback';
  }

  autism_score = Math.max(0, Math.min(100, autism_score));

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

  const sensoryAvg = Object.values(sensory).reduce((a, b) => a + b, 0) / 7;
  const sensoryVar = Object.values(sensory).reduce((a, v) => a + Math.pow(v - sensoryAvg, 2), 0) / 7;
  let pattern_type;
  if (sensoryAvg > 7) pattern_type = 'Hyper-Responsive (Sensory Avoidant)';
  else if (sensoryAvg < 3) pattern_type = 'Hypo-Responsive (Sensory Seeking)';
  else if (sensoryVar > 4) pattern_type = 'Mixed Responsivity (Variable)';
  else pattern_type = 'Moderate / Stable Processing';

  // Confidence: average confidence across all available models
  let confidence;
  if (modelResults.length >= 2) {
    const avgConf = modelResults.reduce((s, m) => s + m.result.confidence, 0) / modelResults.length;
    confidence = Math.round((avgConf * 0.7 + 20 + modelResults.length * 2) * 10) / 10;
  } else {
    confidence = Math.min(92, 70 + responses.length * 0.3);
  }
  confidence = Math.min(98, confidence);

  const recommendations = generateDetailedRecommendations(sensory, domains, autism_score, autism_level, pattern_type, healthProfile);

  const modelNames = modelResults.map(m => {
    const names = { rf: 'Random Forest', xgb: 'XGBoost', svm: 'SVM', lgb: 'LightGBM' };
    return names[m.name];
  }).join(' + ');

  const detailed_report = {
    summary: `Based on comprehensive analysis of ${responses.length} questionnaire responses using ${modelResults.length >= 2 ? modelResults.length + ' AI models (' + modelNames + ')' : modelMode === 'fallback' ? 'direct scoring' : 'trained ML model'}, the NeuroSense engine has computed an autism spectrum score of ${autism_score}/100.`,
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
    model_predictions: {
      rf: rfResult ? { probability: Math.round(rfResult.probability * 1000) / 1000, prediction: rfResult.prediction, confidence: rfResult.confidence } : null,
      xgb: xgbResult ? { probability: Math.round(xgbResult.probability * 1000) / 1000, prediction: xgbResult.prediction, confidence: xgbResult.confidence } : null,
      svm: svmResult ? { probability: Math.round(svmResult.probability * 1000) / 1000, prediction: svmResult.prediction, confidence: svmResult.confidence } : null,
      lgb: lgbResult ? { probability: Math.round(lgbResult.probability * 1000) / 1000, prediction: lgbResult.prediction, confidence: lgbResult.confidence } : null,
    },
    model_info: {
      mode: modelMode,
      models_loaded: modelResults.length,
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
      xgb: xgbResult ? {
        type: 'XGBoost Classifier',
        n_estimators: xgbModel.n_estimators,
        accuracy: xgbModel.metrics.accuracy,
        prediction: xgbResult.prediction,
        probability: Math.round(xgbResult.probability * 1000) / 1000,
        confidence: xgbResult.confidence,
        dataset: xgbModel.training_info.dataset,
        feature_importances: xgbModel.feature_importances
      } : null,
      svm: svmResult ? {
        type: 'SVM (RBF Kernel)',
        n_support_vectors: svmModel.support_vectors.length,
        accuracy: svmModel.metrics.accuracy,
        prediction: svmResult.prediction,
        probability: Math.round(svmResult.probability * 1000) / 1000,
        confidence: svmResult.confidence,
        dataset: svmModel.training_info.dataset
      } : null,
      lgb: lgbResult ? {
        type: 'LightGBM Classifier',
        n_estimators: lgbModel.n_estimators,
        accuracy: lgbModel.metrics.accuracy,
        prediction: lgbResult.prediction,
        probability: Math.round(lgbResult.probability * 1000) / 1000,
        confidence: lgbResult.confidence,
        dataset: lgbModel.training_info.dataset,
        feature_importances: lgbModel.feature_importances
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

app.get('/api/admin/stats', requireAdmin, async (req, res) => {
  const userCount = await db.execute("SELECT COUNT(*) as cnt FROM users");
  const assessedCount = await db.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM ml_results");
  const questionnaireCount = await db.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM questionnaire_responses");
  res.json({
    total_users: userCount.rows[0][0] || 0,
    users_with_assessments: assessedCount.rows[0][0] || 0,
    users_with_questionnaires: questionnaireCount.rows[0][0] || 0
  });
});

app.get('/api/admin/users', requireAdmin, async (req, res) => {
  const result = await db.execute(`
    SELECT id, email, first_name, last_name, date_of_birth, gender,
           created_at, onboarding_complete, questionnaire_complete
    FROM users ORDER BY created_at DESC
  `);
  if (result.rows.length === 0) return res.json({ users: [], count: 0 });
  res.json({ users: rowsToObjects(result), count: result.rows.length });
});

app.get('/api/admin/users/:userId/assessment', requireAdmin, async (req, res) => {
  const userId = req.params.userId;

  const userResult = await db.execute({ sql: "SELECT id, email, first_name, last_name, created_at FROM users WHERE id = ?", args: [userId] });
  if (userResult.rows.length === 0) {
    return res.status(404).json({ error: 'User not found' });
  }
  const user = {};
  userResult.columns.forEach((c, i) => user[c] = userResult.rows[0][i]);

  const hpResult = await db.execute({ sql: "SELECT * FROM health_profiles WHERE user_id = ?", args: [userId] });
  let healthProfile = null;
  if (hpResult.rows.length > 0) {
    healthProfile = {};
    hpResult.columns.forEach((c, i) => healthProfile[c] = hpResult.rows[0][i]);
  }

  const qResult = await db.execute({ sql: "SELECT section, question_id, question_text, answer_value FROM questionnaire_responses WHERE user_id = ? ORDER BY section, question_id", args: [userId] });
  let responses = [];
  if (qResult.rows.length > 0) {
    responses = rowsToObjects(qResult);
  }

  const mlResult = await db.execute({ sql: "SELECT * FROM ml_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", args: [userId] });
  let mlResults = null;
  if (mlResult.rows.length > 0) {
    mlResults = {};
    mlResult.columns.forEach((c, i) => mlResults[c] = mlResult.rows[0][i]);
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
