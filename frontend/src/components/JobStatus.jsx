import { useState, useEffect } from 'react';
import { getJobStatus } from '../services/api';

export default function JobStatus({ job, onComplete }) {
  const [status, setStatus] = useState(job);
  const [polling, setPolling] = useState(true);

  useEffect(() => {
    if (!polling) return;
    if (status.status === 'completed' || status.status === 'failed') {
      setPolling(false);
      if (status.status === 'completed') {
        onComplete?.();
      }
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await getJobStatus(status.job_id);
        setStatus(updated);
        if (updated.status === 'completed' || updated.status === 'failed') {
          setPolling(false);
          if (updated.status === 'completed') {
            onComplete?.();
          }
        }
      } catch (err) {
        console.error('Failed to poll job status:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [status.job_id, status.status, polling, onComplete]);

  const getStatusColor = () => {
    switch (status.status) {
      case 'completed': return '#4caf50';
      case 'failed': return '#f44336';
      case 'processing': return '#2196f3';
      default: return '#9e9e9e';
    }
  };

  return (
    <div className="job-status">
      <div className="job-header">
        <span className="job-id">Job: {status.job_id}</span>
        <span className="status-badge" style={{ backgroundColor: getStatusColor() }}>
          {status.status}
        </span>
      </div>

      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${status.progress}%` }}
        />
      </div>
      <p className="progress-text">{status.message}</p>

      {status.videos && status.videos.length > 0 && (
        <div className="video-list">
          {status.videos.map((video) => (
            <div key={video.video_id} className="video-item">
              <span className="video-title">{video.title}</span>
              <span className={`index-status ${video.indexed ? 'indexed' : ''}`}>
                {video.indexed ? `${video.chunks_count} chunks` : 'pending'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
