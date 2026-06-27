import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  CircleStop,
  Download,
  FileAudio,
  FileImage,
  FileVideo,
  FolderOpen,
  Loader2,
  Mic,
  Play,
  Radio,
  RefreshCw,
  Server,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Terminal,
  Upload,
  Video
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import LiveMirror from "./LiveMirror";

export type UploadInfo = {
  id: string;
  filename: string;
  content_type?: string;
  kind: string;
  path: string;
  size: number;
  url: string;
  created_at: string;
};

type ArtifactInfo = {
  name: string;
  size: number;
  kind: string;
  url: string;
  created_at: string;
};

type JobInfo = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  created_at: string;
  updated_at: string;
  request: Record<string, unknown>;
  command: string[];
  artifacts: ArtifactInfo[];
  logs_tail: string[];
  error?: string | null;
  returncode?: number | null;
  progress?: string | null;
};

type RuntimeInfo = {
  project_root: string;
  data_root: string;
  python: string;
  platform: Record<string, string>;
  cuda: Record<string, unknown>;
  torch: Record<string, unknown>;
  flash_attention: Record<string, unknown>;
  models: Array<{ name: string; ok: boolean; path: string }>;
  packages: { no_x86_64_wheels: boolean; x86_64_wheels: string[] };
  auth_enabled: boolean;
  active_job?: string | null;
};

type Tab = "live" | "generate" | "queue" | "settings";

const API_ROOT = "";
const terminalStatuses = new Set(["succeeded", "failed", "cancelled"]);

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[index]}`;
}

function classNames(...items: Array<string | false | null | undefined>) {
  return items.filter(Boolean).join(" ");
}

function useStoredState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    const stored = localStorage.getItem(key);
    if (!stored) return initial;
    try {
      return JSON.parse(stored) as T;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue] as const;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [liveEngine, setLiveEngine] = useStoredState<"render" | "live">("hallo4.liveEngine", "render");
  const [authToken, setAuthToken] = useStoredState("hallo4.authToken", "");
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [selectedJobId, setSelectedJobId] = useStoredState<string | null>("hallo4.selectedJob", null);
  const [recentJobs, setRecentJobs] = useStoredState<string[]>("hallo4.recentJobs", []);
  const [busy, setBusy] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState("a person is talking");
  const [sourceVideo, setSourceVideo] = useState<UploadInfo | null>(null);
  const [referenceImages, setReferenceImages] = useState<UploadInfo[]>([]);
  const [audio, setAudio] = useState<UploadInfo | null>(null);
  const [manualSourceVideo, setManualSourceVideo] = useState("assets/01.mp4");
  const [manualReferenceImages, setManualReferenceImages] = useState("assets/01.png");
  const [manualAudio, setManualAudio] = useState("assets/01.wav");
  const [size, setSize] = useState("480*832");
  const [frameNum, setFrameNum] = useState(81);
  const [startFrame, setStartFrame] = useState(0);
  const [motionFrames, setMotionFrames] = useState(1);
  const [seed, setSeed] = useState(2025);
  const [solver, setSolver] = useState<"unipc" | "dpm++">("unipc");
  const [steps, setSteps] = useState(25);
  const [maxRound, setMaxRound] = useState<number | "">("");
  const [shift, setShift] = useState(8);
  const [guideScale, setGuideScale] = useState(6);
  const [offload, setOffload] = useState(false);
  const [t5Cpu, setT5Cpu] = useState(false);
  const [ckptDir, setCkptDir] = useState("pretrained_models/Wan2.1_Encoders");
  const [modelPath, setModelPath] = useState("pretrained_models/hallo4/model_weight.ckpt");
  const [audioSeparator, setAudioSeparator] = useState("pretrained_models/audio_separator/Kim_Vocal_2.onnx");
  const [wav2vecPath, setWav2vecPath] = useState("pretrained_models/wav2vec2-base-960h");
  const [activeLogs, setActiveLogs] = useState<string[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  const selectedJob = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null, [jobs, selectedJobId]);
  const finalArtifact = useMemo(() => {
    return selectedJob?.artifacts.find((artifact) => artifact.name.includes("_out_video") && artifact.kind === "video") ?? selectedJob?.artifacts.find((artifact) => artifact.kind === "video");
  }, [selectedJob]);

  const headers = useMemo(() => {
    const output: Record<string, string> = {};
    if (authToken.trim()) output.Authorization = `Bearer ${authToken.trim()}`;
    return output;
  }, [authToken]);

  async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers: {
        ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...headers,
        ...(init.headers ?? {})
      }
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    return response.json() as Promise<T>;
  }

  async function refresh() {
    try {
      const [runtimePayload, jobPayload] = await Promise.all([api<RuntimeInfo>("/api/runtime"), api<JobInfo[]>("/api/jobs")]);
      setRuntime(runtimePayload);
      setJobs(jobPayload);
      if (!selectedJobId && jobPayload[0]) setSelectedJobId(jobPayload[0].id);
      setApiError(null);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    }
  }

  // One-click auth: a ?token=... in the URL is stored, then stripped from the bar
  // (so it isn't shoulder-surfed / left in history). Lets you bookmark an
  // authenticated link instead of pasting the token under Preflight > Access.
  useEffect(() => {
    const urlToken = new URLSearchParams(window.location.search).get("token");
    if (urlToken) {
      setAuthToken(urlToken);
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3500);
    return () => window.clearInterval(timer);
  }, [authToken]);

  useEffect(() => {
    if (!selectedJob) return;
    setActiveLogs(selectedJob.logs_tail ?? []);
    if (eventSourceRef.current) eventSourceRef.current.close();
    if (terminalStatuses.has(selectedJob.status)) return;
    const tokenQuery = authToken.trim() ? `?token=${encodeURIComponent(authToken.trim())}` : "";
    const source = new EventSource(`/api/jobs/${selectedJob.id}/events${tokenQuery}`);
    eventSourceRef.current = source;
    source.addEventListener("log", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as { message: string };
      setActiveLogs((current) => [...current, payload.message].slice(-300));
    });
    source.addEventListener("status", () => void refresh());
    source.addEventListener("done", () => {
      source.close();
      void refresh();
    });
    source.onerror = () => source.close();
    return () => source.close();
  }, [selectedJob?.id, authToken]);

  async function upload(kind: string, file: File) {
    const form = new FormData();
    form.append("kind", kind);
    form.append("file", file);
    return api<UploadInfo>("/api/uploads", { method: "POST", body: form });
  }

  async function onUploadVideo(fileList: FileList | null) {
    const file = fileList?.[0];
    if (!file) return;
    setSourceVideo(await upload("video", file));
  }

  async function onUploadImages(fileList: FileList | null) {
    const files = Array.from(fileList ?? []);
    if (!files.length) return;
    const uploaded = await Promise.all(files.map((file) => upload("reference_image", file)));
    setReferenceImages((current) => [...current, ...uploaded]);
  }

  async function onUploadAudio(fileList: FileList | null) {
    const file = fileList?.[0];
    if (!file) return;
    setAudio(await upload("audio", file));
  }

  async function submitJob(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const payload = {
        prompt,
        source_video: sourceVideo?.id || manualSourceVideo,
        reference_images: referenceImages.length ? referenceImages.map((item) => item.id) : manualReferenceImages.split(",").map((item) => item.trim()).filter(Boolean),
        audio: audio?.id || manualAudio,
        size,
        frame_num: frameNum,
        start_inf_frame: startFrame,
        n_motion_frame: motionFrames,
        seed,
        sample_solver: solver,
        sample_steps: steps || null,
        max_round: maxRound === "" ? null : maxRound,
        sample_shift: shift || null,
        sample_guide_scale: guideScale,
        offload_model: offload,
        t5_cpu: t5Cpu,
        ckpt_dir: ckptDir,
        model_path: modelPath,
        audio_separator_model_path: audioSeparator,
        wav2vec_model_path: wav2vecPath
      };
      const job = await api<JobInfo>("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
      setSelectedJobId(job.id);
      setRecentJobs([job.id, ...recentJobs.filter((id) => id !== job.id)].slice(0, 10));
      setTab("queue");
      await refresh();
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function cancelJob(jobId: string) {
    await api<JobInfo>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    await refresh();
  }

  const runtimeOk = Boolean(runtime?.torch?.cuda_available && runtime?.packages?.no_x86_64_wheels);
  const missingModels = runtime?.models.filter((item) => !item.ok) ?? [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Jetson AGX Thor</div>
          <h1>Hallo4 Studio</h1>
        </div>
        <div className="topbar-actions">
          <StatusPill ok={runtimeOk} label={runtimeOk ? "GPU ready" : "Preflight needed"} />
          <button className="icon-button" onClick={() => void refresh()} title="Refresh runtime and jobs">
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      {apiError && (
        <div className="alert">
          <AlertTriangle size={18} />
          <span>{apiError}</span>
        </div>
      )}

      <nav className="tabs">
        <button className={classNames(tab === "live" && "active")} onClick={() => setTab("live")}>
          <Camera size={17} /> Live Studio
        </button>
        <button className={classNames(tab === "generate" && "active")} onClick={() => setTab("generate")}>
          <Video size={17} /> Generate
        </button>
        <button className={classNames(tab === "queue" && "active")} onClick={() => setTab("queue")}>
          <Terminal size={17} /> Queue
        </button>
        <button className={classNames(tab === "settings" && "active")} onClick={() => setTab("settings")}>
          <Settings size={17} /> Preflight
        </button>
      </nav>

      {tab === "live" && (
        <>
          <nav className="tabs">
            <button className={classNames(liveEngine === "render" && "active")} onClick={() => setLiveEngine("render")}>
              <Video size={16} /> Render (hallo4)
            </button>
            <button className={classNames(liveEngine === "live" && "active")} onClick={() => setLiveEngine("live")}>
              <Radio size={16} /> Mirror (live · beta)
            </button>
          </nav>
          {liveEngine === "render" ? (
            <LiveStudio
              api={api}
              upload={upload}
              onSubmitted={(id) => {
                setSelectedJobId(id);
                setRecentJobs([id, ...recentJobs.filter((existing) => existing !== id)].slice(0, 10));
              }}
            />
          ) : (
            <LiveMirror api={api} upload={upload} />
          )}
        </>
      )}

      {tab === "generate" && (
        <main className="workspace">
          <form className="panel input-panel" onSubmit={submitJob}>
            <section className="section">
              <div className="section-title">
                <Upload size={17} />
                <h2>Inputs</h2>
              </div>
              <label>
                Prompt
                <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={3} />
              </label>
              <div className="upload-grid">
                <UploadSlot icon={<FileVideo size={18} />} label="Source video" upload={sourceVideo} fallback={manualSourceVideo} onFallback={setManualSourceVideo} onFile={onUploadVideo} accept="video/*" />
                <UploadSlot icon={<FileImage size={18} />} label="Reference images" upload={referenceImages[0]} count={referenceImages.length} fallback={manualReferenceImages} onFallback={setManualReferenceImages} onFile={onUploadImages} accept="image/*" multiple />
                <UploadSlot icon={<FileAudio size={18} />} label="Driving audio" upload={audio} fallback={manualAudio} onFallback={setManualAudio} onFile={onUploadAudio} accept="audio/*,.wav" />
              </div>
            </section>

            <section className="section">
              <div className="section-title">
                <SlidersHorizontal size={17} />
                <h2>Generation</h2>
              </div>
              <div className="control-grid">
                <label>
                  Size
                  <select value={size} onChange={(event) => setSize(event.target.value)}>
                    <option value="480*832">480*832</option>
                    <option value="832*480">832*480</option>
                  </select>
                </label>
                <NumberInput label="Frames" value={frameNum} onChange={(value) => value !== "" && setFrameNum(value)} min={5} step={4} />
                <NumberInput label="Start frame" value={startFrame} onChange={(value) => value !== "" && setStartFrame(value)} min={0} />
                <NumberInput label="Motion frames" value={motionFrames} onChange={(value) => value !== "" && setMotionFrames(value)} min={1} />
                <NumberInput label="Seed" value={seed} onChange={(value) => value !== "" && setSeed(value)} />
                <label>
                  Solver
                  <select value={solver} onChange={(event) => setSolver(event.target.value as "unipc" | "dpm++")}>
                    <option value="unipc">UniPC</option>
                    <option value="dpm++">DPM++</option>
                  </select>
                </label>
                <NumberInput label="Steps" value={steps} onChange={(value) => value !== "" && setSteps(value)} min={1} />
                <NumberInput label="Max rounds" value={maxRound} onChange={setMaxRound} min={1} allowBlank />
                <NumberInput label="Shift" value={shift} onChange={(value) => value !== "" && setShift(value)} min={0} step={0.5} />
                <NumberInput label="Guide scale" value={guideScale} onChange={(value) => value !== "" && setGuideScale(value)} min={0} step={0.5} />
              </div>
              <div className="toggles">
                <label className="check-row">
                  <input type="checkbox" checked={offload} onChange={(event) => setOffload(event.target.checked)} />
                  Offload model
                </label>
                <label className="check-row">
                  <input type="checkbox" checked={t5Cpu} onChange={(event) => setT5Cpu(event.target.checked)} />
                  T5 on CPU
                </label>
              </div>
            </section>

            <section className="section">
              <div className="section-title">
                <FolderOpen size={17} />
                <h2>Models</h2>
              </div>
              <div className="path-grid">
                <TextInput label="Wan encoders" value={ckptDir} onChange={setCkptDir} />
                <TextInput label="Hallo4 checkpoint" value={modelPath} onChange={setModelPath} />
                <TextInput label="Audio separator" value={audioSeparator} onChange={setAudioSeparator} />
                <TextInput label="Wav2Vec" value={wav2vecPath} onChange={setWav2vecPath} />
              </div>
            </section>

            <button className="primary-button" disabled={busy}>
              {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              Submit job
            </button>
          </form>

          <aside className="panel preview-panel">
            <Preview job={selectedJob} artifact={finalArtifact} />
            <RuntimeStrip runtime={runtime} missingModels={missingModels.length} />
          </aside>
        </main>
      )}

      {tab === "queue" && (
        <main className="queue-layout">
          <section className="panel jobs-list">
            <div className="section-title">
              <Activity size={17} />
              <h2>Jobs</h2>
            </div>
            {jobs.length === 0 && <div className="empty">No jobs yet</div>}
            {jobs.map((job) => (
              <button key={job.id} className={classNames("job-row", selectedJob?.id === job.id && "selected")} onClick={() => setSelectedJobId(job.id)}>
                <span className={classNames("dot", job.status)} />
                <span>
                  <strong>{job.id}</strong>
                  <small>{String(job.request.prompt ?? "batch prompt")}</small>
                </span>
                <em>{job.progress ?? job.status}</em>
              </button>
            ))}
          </section>

          <section className="panel job-detail">
            {selectedJob ? (
              <>
                <div className="detail-header">
                  <div>
                    <div className="eyebrow">{selectedJob.status}</div>
                    <h2>{selectedJob.id}</h2>
                  </div>
                  {!terminalStatuses.has(selectedJob.status) && (
                    <button className="danger-button" onClick={() => void cancelJob(selectedJob.id)}>
                      <CircleStop size={17} /> Cancel
                    </button>
                  )}
                </div>
                {selectedJob.error && <div className="alert compact">{selectedJob.error}</div>}
                <Preview job={selectedJob} artifact={finalArtifact} />
                <ArtifactBrowser artifacts={selectedJob.artifacts} />
                <LogView lines={activeLogs.length ? activeLogs : selectedJob.logs_tail} />
              </>
            ) : (
              <div className="empty">Select a job</div>
            )}
          </section>
        </main>
      )}

      {tab === "settings" && (
        <main className="settings-layout">
          <section className="panel">
            <div className="section-title">
              <Settings size={17} />
              <h2>Runtime</h2>
            </div>
            <div className="runtime-grid">
              <Metric label="Python" value={runtime?.python ?? "unknown"} />
              <Metric label="Machine" value={runtime?.platform?.machine ?? "unknown"} />
              <Metric label="Torch" value={String(runtime?.torch?.version ?? "missing")} ok={Boolean(runtime?.torch?.available)} />
              <Metric label="CUDA" value={String(runtime?.torch?.cuda_version ?? "unknown")} ok={Boolean(runtime?.torch?.cuda_available)} />
              <Metric label="Device" value={String(runtime?.torch?.device_name ?? "unknown")} ok={Boolean(runtime?.torch?.cuda_available)} />
              <Metric label="FlashAttention" value={runtime?.flash_attention?.available ? "available" : "SDPA fallback"} ok={Boolean(runtime?.flash_attention?.available)} soft />
              <Metric label="x86_64 wheels" value={runtime?.packages?.no_x86_64_wheels ? "none" : "found"} ok={Boolean(runtime?.packages?.no_x86_64_wheels)} />
              <Metric label="Auth" value={runtime?.auth_enabled ? "enabled" : "localhost only"} ok={true} soft />
            </div>
          </section>

          <section className="panel">
            <div className="section-title">
              <CheckCircle2 size={17} />
              <h2>Models</h2>
            </div>
            <div className="model-list">
              {runtime?.models.map((model) => (
                <div key={model.name} className="model-row">
                  <StatusPill ok={model.ok} label={model.ok ? "found" : "missing"} />
                  <span>{model.name}</span>
                  <code>{model.path}</code>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="section-title">
              <Terminal size={17} />
              <h2>Access</h2>
            </div>
            <label>
              Bearer token
              <input value={authToken} onChange={(event) => setAuthToken(event.target.value)} placeholder="HALLO4_STUDIO_TOKEN" />
            </label>
            <pre className="command">conda run -n hallo4-thor python scripts/thor_preflight.py</pre>
          </section>
        </main>
      )}
    </div>
  );
}

function StatusPill({ ok, label, soft = false }: { ok: boolean; label: string; soft?: boolean }) {
  return <span className={classNames("status-pill", ok ? "ok" : soft ? "soft" : "bad")}>{label}</span>;
}

function UploadSlot({
  icon,
  label,
  upload,
  count,
  fallback,
  onFallback,
  onFile,
  accept,
  multiple
}: {
  icon: React.ReactNode;
  label: string;
  upload: UploadInfo | null;
  count?: number;
  fallback: string;
  onFallback: (value: string) => void;
  onFile: (files: FileList | null) => void;
  accept: string;
  multiple?: boolean;
}) {
  return (
    <div className="upload-slot">
      <div className="slot-header">
        {icon}
        <strong>{label}</strong>
      </div>
      <label className="file-button">
        <Upload size={16} />
        <span>{upload ? (count && count > 1 ? `${count} files` : upload.filename) : "Upload"}</span>
        <input type="file" accept={accept} multiple={multiple} onChange={(event) => void onFile(event.target.files)} />
      </label>
      <input value={fallback} onChange={(event) => onFallback(event.target.value)} />
    </div>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  step,
  allowBlank
}: {
  label: string;
  value: number | "";
  onChange: (value: number | "") => void;
  min?: number;
  step?: number;
  allowBlank?: boolean;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value}
        min={min}
        step={step}
        onChange={(event) => {
          if (allowBlank && event.target.value === "") onChange("");
          else onChange(Number(event.target.value));
        }}
      />
    </label>
  );
}

function TextInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label>
      {label}
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function Preview({ job, artifact }: { job: JobInfo | null; artifact?: ArtifactInfo }) {
  return (
    <section className="preview">
      <div className="section-title">
        <Video size={17} />
        <h2>Preview</h2>
      </div>
      {artifact ? (
        <video src={artifact.url} controls playsInline />
      ) : (
        <div className="preview-empty">
          <Video size={42} />
          <span>{job ? "Waiting for video artifact" : "Submit or select a job"}</span>
        </div>
      )}
    </section>
  );
}

function RuntimeStrip({ runtime, missingModels }: { runtime: RuntimeInfo | null; missingModels: number }) {
  return (
    <section className="runtime-strip">
      <Metric label="Torch" value={String(runtime?.torch?.version ?? "missing")} ok={Boolean(runtime?.torch?.available)} />
      <Metric label="CUDA" value={runtime?.torch?.cuda_available ? "available" : "missing"} ok={Boolean(runtime?.torch?.cuda_available)} />
      <Metric label="Capability" value={String(runtime?.torch?.device_capability ?? "unknown")} ok={String(runtime?.torch?.device_capability) === "11,0"} soft />
      <Metric label="Models" value={missingModels ? `${missingModels} missing` : "ready"} ok={missingModels === 0} />
    </section>
  );
}

function Metric({ label, value, ok, soft }: { label: string; value: string; ok?: boolean; soft?: boolean }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={classNames(ok === undefined ? "" : ok ? "good" : soft ? "warn" : "bad")}>{value}</strong>
    </div>
  );
}

function ArtifactBrowser({ artifacts }: { artifacts: ArtifactInfo[] }) {
  return (
    <section className="artifacts">
      <div className="section-title">
        <Download size={17} />
        <h2>Artifacts</h2>
      </div>
      {artifacts.length === 0 && <div className="empty">No artifacts yet</div>}
      <div className="artifact-grid">
        {artifacts.map((artifact) => (
          <a key={artifact.name} href={artifact.url} className="artifact-item" download>
            <span>{artifact.kind === "video" ? <FileVideo size={17} /> : artifact.kind === "image" ? <FileImage size={17} /> : <Download size={17} />}</span>
            <strong>{artifact.name}</strong>
            <small>{formatBytes(artifact.size)}</small>
          </a>
        ))}
      </div>
    </section>
  );
}

function LogView({ lines }: { lines: string[] }) {
  return (
    <section className="logs">
      <div className="section-title">
        <Terminal size={17} />
        <h2>Live logs</h2>
      </div>
      <pre>{lines.length ? lines.join("\n") : "Logs will appear when the job starts."}</pre>
    </section>
  );
}

export type Api = <T,>(path: string, init?: RequestInit) => Promise<T>;

function LiveStudio({
  api,
  upload,
  onSubmitted
}: {
  api: Api;
  upload: (kind: string, file: File) => Promise<UploadInfo>;
  onSubmitted: (jobId: string) => void;
}) {
  const [source, setSource] = useState<"client" | "server">("client");
  const [prompt, setPrompt] = useState("a person is talking");

  // Client (browser) capture
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [videoDeviceId, setVideoDeviceId] = useState("");
  const [audioDeviceId, setAudioDeviceId] = useState("");
  const [previewing, setPreviewing] = useState(false);
  const [recording, setRecording] = useState(false);
  const previewRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const blobsRef = useRef<Blob[]>([]);

  // Server (Thor box) capture
  const [serverVideo, setServerVideo] = useState<Array<{ path: string; name: string }>>([]);
  const [serverAudio, setServerAudio] = useState<Array<{ id: string; name: string }>>([]);
  const [serverVideoDev, setServerVideoDev] = useState("");
  const [serverAudioDev, setServerAudioDev] = useState("default");
  const [serverCapturing, setServerCapturing] = useState(false);

  // Shared
  const [take, setTake] = useState<UploadInfo | null>(null);
  const [refImage, setRefImage] = useState<UploadInfo | null>(null);
  const [voiceSample, setVoiceSample] = useState<UploadInfo | null>(null);
  const [voiceConvert, setVoiceConvert] = useState(true);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const secureContext =
    typeof window !== "undefined" &&
    (window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1");
  const canCaptureBrowser = secureContext && typeof navigator !== "undefined" && !!navigator.mediaDevices;

  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), []);

  async function enableCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoDeviceId ? { deviceId: { exact: videoDeviceId } } : true,
        audio: audioDeviceId ? { deviceId: { exact: audioDeviceId } } : true
      });
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      if (previewRef.current) {
        previewRef.current.srcObject = stream;
        await previewRef.current.play().catch(() => undefined);
      }
      setPreviewing(true);
      const devices = await navigator.mediaDevices.enumerateDevices();
      setVideoDevices(devices.filter((device) => device.kind === "videoinput"));
      setAudioDevices(devices.filter((device) => device.kind === "audioinput"));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function startRecording() {
    const stream = streamRef.current;
    if (!stream) {
      setError("Enable the camera first.");
      return;
    }
    blobsRef.current = [];
    const preferred = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
    const mimeType = preferred.find((type) => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = (event) => {
      if (event.data.size) blobsRef.current.push(event.data);
    };
    recorder.onstop = async () => {
      const blob = new Blob(blobsRef.current, { type: "video/webm" });
      const file = new File([blob], `take_${Date.now()}.webm`, { type: "video/webm" });
      try {
        setTake(await upload("video", file));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    recorderRef.current = recorder;
    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function loadServerDevices() {
    setError(null);
    try {
      const data = await api<{ video: Array<{ path: string; name: string }>; audio: Array<{ id: string; name: string }> }>(
        "/api/devices"
      );
      setServerVideo(data.video);
      setServerAudio(data.audio);
      if (data.video[0]) setServerVideoDev(data.video[0].path);
      if (data.audio[0]) setServerAudioDev(data.audio[0].id);
      if (data.video.length === 0) setError("No webcam found on the Thor box (/dev/video*).");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function serverStart() {
    setError(null);
    try {
      await api("/api/capture/start", {
        method: "POST",
        body: JSON.stringify({ video_device: serverVideoDev, audio_device: serverAudioDev })
      });
      setServerCapturing(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function serverStop() {
    try {
      const info = await api<UploadInfo>("/api/capture/stop", { method: "POST" });
      setTake(info);
      setServerCapturing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function uploadInto(setter: (info: UploadInfo) => void, kind: string, files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    try {
      setter(await upload(kind, file));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function submit() {
    setError(null);
    if (!take) return setError("Record or capture a driving take first.");
    if (!refImage) return setError("Upload the target person's reference image.");
    if (voiceConvert && !voiceSample) return setError("Upload a target voice sample, or turn off voice cloning.");
    if (voiceConvert && !consent) return setError("Please confirm consent to use this likeness and voice.");
    setBusy(true);
    try {
      const payload = {
        prompt,
        source_video: take.id,
        reference_images: [refImage.id],
        audio: take.id,
        target_voice: voiceSample?.id ?? null,
        voice_convert: voiceConvert,
        stream_chunks: true,
        consent,
        size: "480*832"
      };
      const job = await api<JobInfo>("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
      setJobId(job.id);
      onSubmitted(job.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="workspace">
      <div className="panel input-panel">
        <section className="section">
          <div className="section-title">
            <Camera size={17} />
            <h2>Capture source</h2>
          </div>
          <div className="tabs" style={{ marginBottom: 12 }}>
            <button className={classNames(source === "client" && "active")} onClick={() => setSource("client")}>
              <Camera size={15} /> This computer
            </button>
            <button className={classNames(source === "server" && "active")} onClick={() => {
              setSource("server");
              if (serverVideo.length === 0) void loadServerDevices();
            }}>
              <Server size={15} /> Thor box
            </button>
          </div>

          {source === "client" && (
            <>
              {!canCaptureBrowser && (
                <div className="alert compact">
                  <AlertTriangle size={16} />
                  <span>Camera/mic need a secure context. Open the studio over <code>https://</code> (run
                  <code> scripts/make_studio_cert.sh</code>) or via <code>localhost</code>.</span>
                </div>
              )}
              <div className="control-grid">
                <label>
                  Webcam
                  <select value={videoDeviceId} onChange={(event) => setVideoDeviceId(event.target.value)}>
                    {videoDevices.length === 0 && <option value="">Default camera</option>}
                    {videoDevices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>{device.label || "Camera"}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Microphone
                  <select value={audioDeviceId} onChange={(event) => setAudioDeviceId(event.target.value)}>
                    {audioDevices.length === 0 && <option value="">Default mic</option>}
                    {audioDevices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>{device.label || "Microphone"}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="toggles">
                <button className="file-button" type="button" onClick={() => void enableCamera()} disabled={!canCaptureBrowser}>
                  <Camera size={16} /> {previewing ? "Restart camera" : "Enable camera"}
                </button>
                {!recording ? (
                  <button className="file-button" type="button" onClick={startRecording} disabled={!previewing}>
                    <Radio size={16} /> Record take
                  </button>
                ) : (
                  <button className="danger-button" type="button" onClick={stopRecording}>
                    <Square size={16} /> Stop & upload
                  </button>
                )}
              </div>
            </>
          )}

          {source === "server" && (
            <>
              <div className="control-grid">
                <label>
                  Webcam (Thor)
                  <select value={serverVideoDev} onChange={(event) => setServerVideoDev(event.target.value)}>
                    {serverVideo.length === 0 && <option value="">No /dev/video*</option>}
                    {serverVideo.map((device) => (
                      <option key={device.path} value={device.path}>{device.name}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Microphone (Thor)
                  <select value={serverAudioDev} onChange={(event) => setServerAudioDev(event.target.value)}>
                    {serverAudio.map((device) => (
                      <option key={device.id} value={device.id}>{device.name}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="toggles">
                <button className="file-button" type="button" onClick={() => void loadServerDevices()}>
                  <RefreshCw size={16} /> Refresh devices
                </button>
                {!serverCapturing ? (
                  <button className="file-button" type="button" onClick={() => void serverStart()} disabled={!serverVideoDev}>
                    <Radio size={16} /> Start capture
                  </button>
                ) : (
                  <button className="danger-button" type="button" onClick={() => void serverStop()}>
                    <Square size={16} /> Stop & use
                  </button>
                )}
              </div>
            </>
          )}
          {take && (
            <div className="metric"><span>Driving take</span><strong className="good">{take.filename}</strong></div>
          )}
        </section>

        <section className="section">
          <div className="section-title">
            <Mic size={17} />
            <h2>Target person</h2>
          </div>
          <label>
            Prompt
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={2} />
          </label>
          <div className="upload-grid">
            <div className="upload-slot">
              <div className="slot-header"><FileImage size={18} /><strong>Reference image</strong></div>
              <label className="file-button">
                <Upload size={16} /><span>{refImage ? refImage.filename : "Upload face"}</span>
                <input type="file" accept="image/*" onChange={(event) => void uploadInto(setRefImage, "reference_image", event.target.files)} />
              </label>
            </div>
            <div className="upload-slot">
              <div className="slot-header"><FileAudio size={18} /><strong>Target voice sample</strong></div>
              <label className="file-button">
                <Upload size={16} /><span>{voiceSample ? voiceSample.filename : "Upload voice clip"}</span>
                <input type="file" accept="audio/*,video/*" onChange={(event) => void uploadInto(setVoiceSample, "voice", event.target.files)} />
              </label>
            </div>
          </div>
          <div className="toggles">
            <label className="check-row">
              <input type="checkbox" checked={voiceConvert} onChange={(event) => setVoiceConvert(event.target.checked)} />
              Clone target voice (kNN-VC)
            </label>
            <label className="check-row">
              <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
              <ShieldCheck size={15} /> I have consent to use this person's likeness and voice
            </label>
          </div>
        </section>

        {error && <div className="alert compact"><AlertTriangle size={16} /><span>{error}</span></div>}
        <button className="primary-button" disabled={busy} onClick={() => void submit()}>
          {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Animate
        </button>
      </div>

      <aside className="panel preview-panel">
        <section className="preview">
          <div className="section-title"><Video size={17} /><h2>{jobId ? "Result (streaming)" : "Camera preview"}</h2></div>
          {jobId ? (
            <ChunkPlayer jobId={jobId} api={api} />
          ) : (
            <video ref={previewRef} muted playsInline autoPlay style={{ width: "100%", borderRadius: 12, background: "#000" }} />
          )}
        </section>
      </aside>
    </main>
  );
}

function ChunkPlayer({ jobId, api }: { jobId: string; api: Api }) {
  const [job, setJob] = useState<JobInfo | null>(null);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    setIdx(0);
    let active = true;
    let timer = 0;
    const tick = async () => {
      try {
        const next = await api<JobInfo>(`/api/jobs/${jobId}`);
        if (!active) return;
        setJob(next);
        if (!terminalStatuses.has(next.status)) timer = window.setTimeout(() => void tick(), 1500);
      } catch {
        if (active) timer = window.setTimeout(() => void tick(), 2000);
      }
    };
    void tick();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [jobId]);

  const done = job ? terminalStatuses.has(job.status) : false;
  const finalArtifact = useMemo(
    () => (job?.artifacts ?? []).find((artifact) => artifact.name.includes("_out_video") && artifact.kind === "video"),
    [job]
  );
  const chunks = useMemo(
    () =>
      (job?.artifacts ?? [])
        .filter((artifact) => artifact.name.includes("_chunk_") && artifact.kind === "video")
        .sort((a, b) => a.name.localeCompare(b.name)),
    [job]
  );

  if (done && finalArtifact) {
    return <video key="final" src={finalArtifact.url} controls autoPlay playsInline style={{ width: "100%", borderRadius: 12, background: "#000" }} />;
  }
  const current = chunks[Math.min(idx, chunks.length - 1)];
  if (!current) {
    return (
      <div className="preview-empty">
        <Loader2 className="spin" size={28} />
        <span>{job ? `Generating (${job.progress ?? job.status})…` : "Submitting…"}</span>
      </div>
    );
  }
  return (
    <video
      key={current.name}
      src={current.url}
      autoPlay
      controls
      playsInline
      onEnded={() => setIdx((value) => (value + 1 < chunks.length ? value + 1 : value))}
      style={{ width: "100%", borderRadius: 12, background: "#000" }}
    />
  );
}
