const API_BASE = '/api';

export async function submitVideo(url) {
  const response = await fetch(`${API_BASE}/videos`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to submit video');
  }
  return response.json();
}

export async function getJobStatus(jobId) {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error('Failed to get job status');
  }
  return response.json();
}

export async function listJobs() {
  const response = await fetch(`${API_BASE}/jobs`);
  if (!response.ok) {
    throw new Error('Failed to list jobs');
  }
  return response.json();
}

export async function queryKnowledgeBase(question, videoIds = null, topK = 5) {
  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, video_ids: videoIds, top_k: topK }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to query');
  }
  return response.json();
}

export async function clearHistory() {
  const response = await fetch(`${API_BASE}/query/history`, {
    method: 'DELETE',
  });
  return response.json();
}

export async function getStats() {
  const response = await fetch(`${API_BASE}/stats`);
  if (!response.ok) {
    throw new Error('Failed to get stats');
  }
  return response.json();
}
