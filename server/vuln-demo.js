/**
 * INTENTIONALLY INSECURE DEMO FILE (for security scanner testing only)
 * This file contains common anti-patterns:
 * - Command injection
 * - SQL injection (simulated)
 * - Reflected XSS
 * - Hardcoded secret
 */

const express = require("express");
const { exec } = require("child_process");

const app = express();
app.use(express.json());

// Hardcoded secret (should be flagged)
const STRIPE_SECRET_KEY = "sk_test_51_FAKE_FAKE_FAKE_FAKE_FAKE";

// Reflected XSS
app.get("/hello", (req, res) => {
  res.send(`<h1>Hello ${req.query.name}</h1>`); // unsanitized user input
});

// Command injection
app.get("/ping", (req, res) => {
  const host = req.query.host; // attacker-controlled
  exec(`ping -c 1 ${host}`, (err, stdout, stderr) => {
    if (err) return res.status(500).send(stderr);
    res.send(stdout);
  });
});

// SQL injection (simulated; pattern only)
app.get("/user", async (req, res) => {
  const id = req.query.id;
  const query = `SELECT * FROM users WHERE id = '${id}'`; // vulnerable pattern
  res.json({ query });
});

app.listen(3000, () => console.log("running"));
