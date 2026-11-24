import { mkdirSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';

const classes = 4;
const temperature = 3;
const alpha = 0.7;
const batchSize = 32;

function random(seed) {
  return () => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

function softmax(logits, t = 1) {
  const max = Math.max(...logits);
  const exps = logits.map(value => Math.exp((value - max) / t));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(value => value / sum);
}

function generate(size, rng) {
  return Array.from({ length: size }, () => {
    const x = [2 * rng() - 1, 2 * rng() - 1];
    const [a, b] = x;
    const prob = softmax([
      3 * a + 2 * Math.sin(4 * b), -3 * a + 2 * Math.cos(4 * b),
      3 * b + 2 * Math.sin(4 * a), -3 * b - 2 * Math.cos(4 * a),
    ]);
    let draw = rng();
    const y = prob.findIndex(p => (draw -= p) < 0);
    return { x, y };
  });
}

function matrix(rows, columns, initializer) {
  return Array.from({ length: rows }, () => Array.from({ length: columns }, initializer));
}

class Model {
  constructor(hidden, seed) {
    const rng = random(seed);
    this.first = matrix(hidden, 3, () => rng() - 0.5);
    this.last = matrix(classes, hidden + 1, () => rng() - 0.5);
  }

  forward(x) {
    const h = this.first.map(([bias, a, b]) => Math.tanh(bias + a * x[0] + b * x[1]));
    const logits = this.last.map(([bias, ...weights]) =>
      weights.reduce((sum, weight, i) => sum + weight * h[i], bias));
    return { h, logits };
  }

  step(batch, learningRate, distill) {
    const firstGrad = matrix(this.first.length, 3, () => 0);
    const lastGrad = matrix(classes, this.first.length + 1, () => 0);
    for (const { x, y, teacher } of batch) {
      const { h, logits } = this.forward(x);
      const hard = softmax(logits);
      const soft = distill ? softmax(logits, temperature) : null;
      const hiddenGrad = Array(h.length).fill(0);
      for (let k = 0; k < classes; k++) {
        const ceGrad = hard[k] - Number(k === y);
        const grad = distill
          ? (1 - alpha) * ceGrad + alpha * temperature * (soft[k] - teacher[k])
          : ceGrad;
        lastGrad[k][0] += grad;
        for (let j = 0; j < h.length; j++) {
          lastGrad[k][j + 1] += grad * h[j];
          hiddenGrad[j] += grad * this.last[k][j + 1];
        }
      }
      for (let j = 0; j < h.length; j++) {
        const grad = hiddenGrad[j] * (1 - h[j] ** 2);
        firstGrad[j][0] += grad;
        firstGrad[j][1] += grad * x[0];
        firstGrad[j][2] += grad * x[1];
      }
    }
    for (const [weights, grads] of [[this.first, firstGrad], [this.last, lastGrad]]) {
      weights.forEach((row, i) => row.forEach((_, j) => {
        row[j] -= learningRate * grads[i][j] / batch.length;
      }));
    }
  }
}

function evaluate(model, data) {
  let correct = 0;
  let loss = 0;
  for (const { x, y } of data) {
    const p = softmax(model.forward(x).logits);
    correct += Number(p.indexOf(Math.max(...p)) === y);
    loss -= Math.log(p[y]);
  }
  return { accuracy: correct / data.length, loss: loss / data.length };
}

function train(model, data, validation, epochs, rate, seed, distill = false) {
  const rng = random(seed);
  const order = data.slice();
  const history = [];
  let seconds = 0;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const start = performance.now();
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let i = 0; i < order.length; i += batchSize) {
      model.step(order.slice(i, i + batchSize), rate, distill);
    }
    seconds += (performance.now() - start) / 1000;
    history.push({ epoch, ...evaluate(model, validation) });
  }
  return { seconds, history };
}

const rng = random(2026);
const training = generate(960, rng);
const validation = generate(2048, rng);
const test = generate(4096, rng);
const teacher = new Model(48, 89);
const teacherRun = train(teacher, training, validation, 240, 0.12, 90);
const cacheStart = performance.now();
const softTraining = training.map(row => ({
  ...row,
  teacher: softmax(teacher.forward(row.x).logits, temperature),
}));
const cacheSeconds = (performance.now() - cacheStart) / 1000;

const runs = [];
const checkpoints = { teacher };
for (const seed of [11, 22, 33, 44, 55]) {
  checkpoints[seed] = {};
  for (const [variant, distill] of [['baseline', false], ['distilled', true]]) {
    const model = new Model(6, seed);
    const run = train(model, distill ? softTraining : training, validation, 100, 0.08, seed + 100, distill);
    runs.push({ seed, variant, ...run, test: evaluate(model, test) });
    checkpoints[seed][variant] = model;
  }
}

const teacherTest = evaluate(teacher, test);
const out = process.argv[2] || 'results';
mkdirSync(out, { recursive: true });
writeFileSync(out + '/metrics.json', JSON.stringify({
  node: process.version, teacher: { ...teacherRun, test: teacherTest }, cacheSeconds, runs,
}, null, 2) + '\n');
writeFileSync(out + '/checkpoints.json', JSON.stringify(checkpoints) + '\n');
console.table(runs.map(({ seed, variant, test, seconds }) => ({ seed, variant, ...test, seconds })));
console.log('Teacher:', teacherTest, 'training seconds:', teacherRun.seconds, 'cache seconds:', cacheSeconds);

