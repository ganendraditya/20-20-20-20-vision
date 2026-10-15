type EyeBlinkStatus = "open" | "closure_pending" | "candidate_completed"
  | "rejected_too_short" | "rejected_too_long" | "rejected_insufficient_samples"
  | "rejected_downward_gaze";

export interface BlinkDiagnosticsDto {
  version: 1;
  decision: "face_unavailable" | "looking_away" | "no_closure_candidate"
    | "closure_pending" | "awaiting_other_eye" | "counted_binocular" | "counted_monocular";
  left_status: EyeBlinkStatus;
  right_status: EyeBlinkStatus;
  smoothed_left_ear: number | null;
  smoothed_right_ear: number | null;
  threshold_left: number | null;
  threshold_right: number | null;
  left_closure_ms: number;
  right_closure_ms: number;
  is_resting: boolean;
}

// Camera Test keeps only a reset offset; the backend owns all blink increments.
export class CameraTestCounter {
  private latest: number | null = null;
  private baseline: number | null = null;
  private snapshotRequest = 0;
  value = 0;

  beginStatusSnapshot() {
    return ++this.snapshotRequest;
  }

  invalidateStatusSnapshot() {
    this.snapshotRequest += 1;
  }

  acceptStatusSnapshot(request: number, total: number) {
    if (request !== this.snapshotRequest) return false;
    this.setBaseline(total);
    return true;
  }

  reset(total: number | null = this.latest) {
    this.invalidateStatusSnapshot();
    this.setBaseline(total);
  }

  private setBaseline(total: number | null) {
    this.latest = total;
    this.baseline = total;
    this.value = 0;
  }

  update(total: number) {
    // A preview queued before the tab's status snapshot must not look like a reset.
    if (this.latest !== null && total < this.latest) return;
    if (this.baseline === null) this.baseline = total;
    this.latest = total;
    this.value = total - this.baseline;
  }
}

export function formatBlinkFeedback(d: BlinkDiagnosticsDto | null | undefined) {
  if (!d || d.version !== 1) {
    return { state: "Detector feedback unavailable", result: null, metrics: "Smoothed EAR / cutoffs: —" };
  }

  const ear = (n: number | null) => n === null ? "—" : n.toFixed(3);
  const metrics = `Smoothed L/R: ${ear(d.smoothed_left_ear)} / ${ear(d.smoothed_right_ear)} · Cutoffs: ${ear(d.threshold_left)} / ${ear(d.threshold_right)}`;
  const states = {
    face_unavailable: "Face unavailable — detection paused",
    looking_away: "Looking away — detection paused",
    no_closure_candidate: "No closure candidate",
    closure_pending: d.is_resting ? "Eyes held closed — resting" : "Closure pending — waiting for recovery",
    awaiting_other_eye: "Waiting for other eye (1 s pairing window)",
    counted_binocular: "Counted blink (+1)",
    counted_monocular: "Counted blink (+1, monocular mode)",
  };
  const state = states[d.decision];
  if (d.decision === "counted_binocular" || d.decision === "counted_monocular") {
    return { state, result: state, metrics };
  }

  function rejection(eye: string, status: EyeBlinkStatus, ms: number): string | null {
    switch (status) {
      case "rejected_too_short": return `${eye}: too short (${Math.round(ms)} ms)`;
      case "rejected_too_long": return `${eye}: long closure (${Math.round(ms)} ms)`;
      case "rejected_insufficient_samples": return `${eye}: insufficient closed samples`;
      case "rejected_downward_gaze": return `${eye}: downward gaze`;
      default: return null;
    }
  }
  const rejected = [
    rejection("L", d.left_status, d.left_closure_ms),
    rejection("R", d.right_status, d.right_closure_ms),
  ].filter((reason): reason is string => reason !== null);
  const longClosureOnly = rejected.length > 0
    && [d.left_status, d.right_status].filter((status) => status.startsWith("rejected_")).every((status) => status === "rejected_too_long");
  const result = rejected.length > 0
    ? longClosureOnly
      ? `Long closure — not counted as a blink (${rejected.join("; ")})`
      : `Not counted — ${rejected.join("; ")}`
    : d.decision === "awaiting_other_eye"
      && (d.left_status === "candidate_completed" || d.right_status === "candidate_completed")
      ? "Not counted yet — waiting for the other eye"
      : null;
  return { state, result, metrics };
}
