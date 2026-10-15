import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { CameraTestCounter, formatBlinkFeedback } from "../src/camera-feedback.ts";

test("Camera Test markup uses unique element IDs", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(([, id]) => id);
  assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique for IPC feedback targets");
});

const diagnostic = {
  version: 1,
  decision: "no_closure_candidate",
  left_status: "open",
  right_status: "open",
  smoothed_left_ear: 0.3,
  smoothed_right_ear: 0.29,
  threshold_left: 0.195,
  threshold_right: 0.19,
  left_closure_ms: 0,
  right_closure_ms: 0,
  is_resting: false,
};

test("backend totals count adjacent events once each, without pulse edge detection", () => {
  const counter = new CameraTestCounter();
  counter.reset(10);
  counter.update(11);
  assert.equal(counter.value, 1);
  counter.update(12);
  assert.equal(counter.value, 2);
  counter.update(12);
  assert.equal(counter.value, 2);
});

test("backend total recovers skipped events and Reset changes only the view offset", () => {
  const counter = new CameraTestCounter();
  counter.reset(10);
  counter.update(14);
  assert.equal(counter.value, 4);
  counter.reset();
  assert.equal(counter.value, 0);
  counter.update(16);
  assert.equal(counter.value, 2);
});

test("tab re-entry starts a new test and stale preview totals cannot create phantom counts", () => {
  const counter = new CameraTestCounter();
  counter.update(25);
  assert.equal(counter.value, 0);
  counter.reset(30);
  counter.update(31);
  assert.equal(counter.value, 1);
  counter.update(29);
  assert.equal(counter.value, 1);
  counter.reset(0);
  assert.equal(counter.value, 0);
  counter.update(1);
  assert.equal(counter.value, 1);
});

test("out-of-order status snapshots cannot reset a newer Camera Test entry", () => {
  const counter = new CameraTestCounter();
  const oldRequest = counter.beginStatusSnapshot();
  const currentRequest = counter.beginStatusSnapshot();
  assert.equal(counter.acceptStatusSnapshot(currentRequest, 11), true);
  assert.equal(counter.acceptStatusSnapshot(oldRequest, 10), false);
  counter.update(12);
  assert.equal(counter.value, 1);
});

test("feedback distinguishes accepted, pending, resting, and unpaired candidates", () => {
  assert.equal(formatBlinkFeedback({ ...diagnostic, decision: "counted_binocular" }).result, "Counted blink (+1)");
  assert.match(formatBlinkFeedback({ ...diagnostic, decision: "counted_monocular" }).result, /monocular/);
  const pending = formatBlinkFeedback({ ...diagnostic, decision: "closure_pending" });
  assert.match(pending.state, /waiting for recovery/);
  assert.equal(pending.result, null);
  assert.match(formatBlinkFeedback({ ...diagnostic, decision: "closure_pending", is_resting: true }).state, /resting/);
  const unpaired = formatBlinkFeedback({ ...diagnostic, decision: "awaiting_other_eye", left_status: "candidate_completed" });
  assert.match(unpaired.result, /Not counted yet/);
  // Ordinary open frames do not erase the last meaningful result in the UI.
  assert.equal(formatBlinkFeedback(diagnostic).result, null);
});

test("rejections retain the eye, reason, and sampled duration", () => {
  for (const [status, reason] of [
    ["rejected_too_short", "too short (30 ms)"],
    ["rejected_insufficient_samples", "insufficient closed samples"],
    ["rejected_downward_gaze", "downward gaze"],
  ]) {
    const feedback = formatBlinkFeedback({ ...diagnostic, left_status: status, left_closure_ms: status === "rejected_too_short" ? 30 : 950 });
    assert.equal(feedback.result, `Not counted — L: ${reason}`);
  }
  assert.equal(
    formatBlinkFeedback({ ...diagnostic, left_status: "rejected_too_long", left_closure_ms: 950 }).result,
    "Long closure — not counted as a blink (L: long closure (950 ms))",
  );
  const feedback = formatBlinkFeedback({ ...diagnostic, left_status: "rejected_too_long", right_status: "rejected_insufficient_samples", left_closure_ms: 950 });
  assert.match(feedback.result, /Not counted — L: long closure \(950 ms\); R: insufficient/);
});

test("feedback reports measured per-eye values and unavailable data honestly", () => {
  assert.equal(formatBlinkFeedback(diagnostic).metrics, "Smoothed L/R: 0.300 / 0.290 · Cutoffs: 0.195 / 0.190");
  assert.match(formatBlinkFeedback({ ...diagnostic, decision: "face_unavailable" }).state, /Face unavailable/);
  assert.match(formatBlinkFeedback({ ...diagnostic, decision: "looking_away" }).state, /Looking away/);
  for (const data of [null, undefined, { ...diagnostic, version: 2 }]) {
    assert.equal(formatBlinkFeedback(data).state, "Detector feedback unavailable");
  }
  assert.match(formatBlinkFeedback({ ...diagnostic, threshold_left: null }).metrics, /Cutoffs: —/);
});
