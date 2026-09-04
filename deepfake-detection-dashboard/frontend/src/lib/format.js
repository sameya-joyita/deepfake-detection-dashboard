export function formatScore(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

export function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

export function formatMilliseconds(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(1)} ms`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function downloadJson(record) {
  const blob = new Blob([JSON.stringify(record, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `deepfake-evidence-${record.analysis_id || "record"}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

