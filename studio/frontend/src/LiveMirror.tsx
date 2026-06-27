import {
  AlertTriangle,
  FileAudio,
  FileImage,
  Loader2,
  Mic,
  Play,
  Radio,
  RefreshCw,
  ShieldCheck,
  Square,
  Upload,
  Video
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Api, UploadInfo } from "./App";

// Status strings the enrollment job may report as terminal (assumed shape — see report).
const ENROLL_TERMINAL = new Set<string>(["succeeded", "done", "failed", "cancelled", "error"]);

type VoiceModel = { id: string; name?: string; status?: string; created_at?: string };
type EnrollStatus = {
  id: string;
  status: string;
  progress?: string | null;
  error?: string | null;
  model_id?: string | null;
};
type OfferAnswer = { sdp: string; type: string; session_id?: string };

// aiortc does not trickle ICE — send the offer only once gathering is done so the
// SDP carries all candidates. ponytail: 2s cap covers a host that never reports "complete".
function waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", done);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", done);
    window.setTimeout(resolve, 2000);
  });
}

export default function LiveMirror({
  api,
  upload
}: {
  api: Api;
  upload: (kind: string, file: File) => Promise<UploadInfo>;
}) {
  const [refImage, setRefImage] = useState<UploadInfo | null>(null);
  const [consent, setConsent] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [error, setError] = useState<string | null>(null);

  // WebRTC session
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const remoteVideoRef = useRef<HTMLVideoElement | null>(null);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [connState, setConnState] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [mirroring, setMirroring] = useState(false);
  const [starting, setStarting] = useState(false);

  // Voice enrollment
  const [models, setModels] = useState<VoiceModel[]>([]);
  const [enrollAvailable, setEnrollAvailable] = useState<boolean | null>(null);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollStatus, setEnrollStatus] = useState("");
  const [voiceName, setVoiceName] = useState("");
  const enrollTimer = useRef<number | undefined>(undefined);

  const secureContext =
    typeof window !== "undefined" &&
    (window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1");
  const canCapture = secureContext && typeof navigator !== "undefined" && !!navigator.mediaDevices;

  useEffect(() => {
    if (localVideoRef.current) localVideoRef.current.srcObject = localStream;
  }, [localStream]);
  useEffect(() => {
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = remoteStream;
  }, [remoteStream]);

  useEffect(() => {
    void loadModels();
    return () => {
      teardown();
      window.clearTimeout(enrollTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadModels() {
    try {
      const data = await api<VoiceModel[]>("/api/voice/models");
      setModels(Array.isArray(data) ? data : []);
      setEnrollAvailable(true);
    } catch {
      // 404 / unreachable → endpoint not built yet, degrade gracefully.
      setEnrollAvailable(false);
    }
  }

  function teardown() {
    pcRef.current?.getSenders().forEach((sender) => sender.track?.stop());
    pcRef.current?.close();
    pcRef.current = null;
    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    localStreamRef.current = null;
    setLocalStream(null);
    setRemoteStream(null);
  }

  async function startMirror() {
    setError(null);
    if (!canCapture)
      return setError("Camera/mic need a secure context. Open the studio over https:// or via localhost.");
    if (!refImage) return setError("Upload the target person's reference image.");
    if (!consent) return setError("Please confirm consent to use this likeness and voice.");
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      localStreamRef.current = stream;
      setLocalStream(stream);

      const pc = new RTCPeerConnection();
      pcRef.current = pc;
      pc.addEventListener("connectionstatechange", () => {
        setConnState(pc.connectionState);
        if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
          setError(`Connection ${pc.connectionState}.`);
        }
      });
      pc.addEventListener("track", (event) => {
        const [incoming] = event.streams;
        if (incoming) setRemoteStream(incoming);
      });
      stream.getTracks().forEach((track) => pc.addTrack(track, stream));

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGathering(pc);
      const local = pc.localDescription;
      if (!local) throw new Error("Failed to create local SDP offer.");

      const answer = await api<OfferAnswer>("/api/live/offer", {
        method: "POST",
        body: JSON.stringify({
          sdp: local.sdp,
          type: local.type,
          target_image: refImage.id,
          target_voice: selectedVoice || null,
          engine: "live"
        })
      });
      await pc.setRemoteDescription({ type: answer.type as RTCSdpType, sdp: answer.sdp });
      setSessionId(answer.session_id ?? null);
      setMirroring(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      teardown();
    } finally {
      setStarting(false);
    }
  }

  async function stopMirror() {
    const sid = sessionId;
    teardown();
    setMirroring(false);
    setConnState("closed");
    setSessionId(null);
    if (sid) {
      try {
        await api(`/api/live/${sid}/stop`, { method: "POST" });
      } catch {
        // best-effort: session is already torn down client-side.
      }
    }
  }

  async function uploadRef(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    try {
      setRefImage(await upload("reference_image", file));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function enrollVoice(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setError(null);
    setEnrolling(true);
    setEnrollStatus("uploading…");
    try {
      const audio = await upload("voice", file);
      const job = await api<EnrollStatus>("/api/voice/enroll", {
        method: "POST",
        body: JSON.stringify({ audio: audio.id, name: voiceName.trim() || file.name.replace(/\.[^.]+$/, "") })
      });
      setEnrollAvailable(true);
      pollEnroll(job.id);
    } catch (err) {
      setEnrolling(false);
      setEnrollStatus("");
      setError(`Voice enrollment unavailable: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function pollEnroll(id: string) {
    const tick = async () => {
      try {
        const st = await api<EnrollStatus>(`/api/voice/enroll/${id}`);
        setEnrollStatus(st.progress || st.status);
        if (ENROLL_TERMINAL.has(st.status)) {
          setEnrolling(false);
          if (st.status === "succeeded" || st.status === "done") {
            await loadModels();
            if (st.model_id) setSelectedVoice(st.model_id);
          } else if (st.error) {
            setError(`Voice enrollment failed: ${st.error}`);
          }
          return;
        }
        enrollTimer.current = window.setTimeout(() => void tick(), 2000);
      } catch (err) {
        setEnrolling(false);
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    void tick();
  }

  const connected = connState === "connected";
  const stateClass = connected ? "ok" : connState === "failed" || connState === "disconnected" ? "bad" : "soft";

  return (
    <main className="workspace">
      <div className="panel input-panel">
        <div className="alert compact" style={{ borderColor: "#cdd7e4", background: "#eef2f8", color: "#33445f" }}>
          <Radio size={16} />
          <span>
            Live mirror drives the target portrait in real time over WebRTC (~250–400 ms). Beta — lower fidelity than
            Render, one session at a time.
          </span>
        </div>

        {!canCapture && (
          <div className="alert compact">
            <AlertTriangle size={16} />
            <span>
              Camera/mic need a secure context. Open the studio over <code>https://</code> (run{" "}
              <code>scripts/make_studio_cert.sh</code>) or via <code>localhost</code>.
            </span>
          </div>
        )}

        <section className="section">
          <div className="section-title">
            <FileImage size={17} />
            <h2>Target person</h2>
          </div>
          <div className="upload-grid" style={{ gridTemplateColumns: "1fr" }}>
            <div className="upload-slot">
              <div className="slot-header">
                <FileImage size={18} />
                <strong>Reference image</strong>
              </div>
              <label className="file-button">
                <Upload size={16} />
                <span>{refImage ? refImage.filename : "Upload face"}</span>
                <input type="file" accept="image/*" onChange={(event) => void uploadRef(event.target.files)} />
              </label>
            </div>
          </div>
          <label className="check-row">
            <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
            <ShieldCheck size={15} /> I have consent to use this person's likeness and voice
          </label>
        </section>

        <section className="section">
          <div className="section-title">
            <Mic size={17} />
            <h2>Target voice (RVC)</h2>
          </div>
          {enrollAvailable === false ? (
            <div className="alert compact">
              <AlertTriangle size={16} />
              <span>Voice enrollment unavailable. The mirror will pass your mic through unchanged.</span>
            </div>
          ) : (
            <>
              <label>
                Enrolled voice
                <select value={selectedVoice} onChange={(event) => setSelectedVoice(event.target.value)}>
                  <option value="">None (mic passthrough)</option>
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name || model.id}
                      {model.status && model.status !== "ready" ? ` (${model.status})` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <div className="upload-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <label>
                  New voice name
                  <input value={voiceName} onChange={(event) => setVoiceName(event.target.value)} placeholder="e.g. Narrator" />
                </label>
                <div className="upload-slot">
                  <div className="slot-header">
                    <FileAudio size={18} />
                    <strong>Enroll new voice</strong>
                  </div>
                  <label className="file-button">
                    {enrolling ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
                    <span>{enrolling ? enrollStatus || "Enrolling…" : "Upload audio & train"}</span>
                    <input
                      type="file"
                      accept="audio/*,.wav"
                      disabled={enrolling}
                      onChange={(event) => void enrollVoice(event.target.files)}
                    />
                  </label>
                </div>
              </div>
              <div className="toggles">
                <button type="button" className="file-button" onClick={() => void loadModels()}>
                  <RefreshCw size={15} /> Refresh voices
                </button>
              </div>
            </>
          )}
        </section>

        {error && (
          <div className="alert compact">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        <div className="toggles">
          {!mirroring ? (
            <button className="primary-button" disabled={starting || !canCapture} onClick={() => void startMirror()}>
              {starting ? <Loader2 className="spin" size={18} /> : <Play size={18} />} Start mirror
            </button>
          ) : (
            <button className="danger-button" onClick={() => void stopMirror()}>
              <Square size={18} /> Stop mirror
            </button>
          )}
          {(mirroring || connState) && <span className={`status-pill ${stateClass}`}>{connState || "idle"}</span>}
        </div>
      </div>

      <aside className="panel preview-panel">
        <section className="preview">
          <div className="section-title">
            <Video size={17} />
            <h2>Live mirror</h2>
          </div>
          <div style={{ position: "relative" }}>
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              style={{
                width: "100%",
                aspectRatio: "16 / 10",
                borderRadius: 8,
                border: "1px solid #dbe1dd",
                background: "#111815",
                objectFit: "cover",
                display: "block"
              }}
            />
            {(mirroring || remoteStream) && (
              <span
                style={{
                  position: "absolute",
                  top: 10,
                  left: 10,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "4px 10px",
                  borderRadius: 999,
                  background: "rgba(15,20,18,0.74)",
                  color: "#ff8a8a",
                  fontSize: 12,
                  fontWeight: 800,
                  letterSpacing: 0.5
                }}
              >
                <span style={{ color: "#ff4d4d", fontSize: 10 }}>⬤</span> SYNTHETIC
              </span>
            )}
            {mirroring && !remoteStream && (
              <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#b7c4be" }}>
                <div style={{ textAlign: "center" }}>
                  <Loader2 className="spin" size={28} />
                  <div style={{ marginTop: 8 }}>Connecting… ({connState || "new"})</div>
                </div>
              </div>
            )}
            {!mirroring && !remoteStream && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "grid",
                  placeItems: "center",
                  color: "#b7c4be",
                  textAlign: "center"
                }}
              >
                <div>
                  <Radio size={40} />
                  <div style={{ marginTop: 8 }}>Start mirror to drive the portrait live</div>
                </div>
              </div>
            )}
            {localStream && (
              <video
                ref={localVideoRef}
                autoPlay
                playsInline
                muted
                style={{
                  position: "absolute",
                  right: 10,
                  bottom: 10,
                  width: "30%",
                  borderRadius: 8,
                  border: "1px solid rgba(255,255,255,0.35)",
                  background: "#000",
                  display: "block"
                }}
              />
            )}
          </div>
        </section>
      </aside>
    </main>
  );
}
