import { useState, useEffect } from 'react';
import { getStats, clearCache } from '../services/api';

export default function Stats({ refreshTrigger, onCacheCleared }) {
  const [stats, setStats] = useState(null);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getStats();
        setStats(data);
      } catch (err) {
        console.error('Failed to fetch stats:', err);
      }
    };

    fetchStats();
  }, [refreshTrigger]);

  const handleClearCache = async () => {
    if (!window.confirm('Clear all cached data? You will need to re-submit videos.')) {
      return;
    }

    setClearing(true);
    try {
      await clearCache();
      onCacheCleared?.();
    } catch (err) {
      console.error('Failed to clear cache:', err);
    } finally {
      setClearing(false);
    }
  };

  if (!stats) return null;

  return (
    <div className="stats">
      <div className="stat-item">
        <span className="stat-value">{stats.total_videos}</span>
        <span className="stat-label">Videos</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">{stats.total_chunks}</span>
        <span className="stat-label">Chunks</span>
      </div>
      <button
        className="clear-cache-btn"
        onClick={handleClearCache}
        disabled={clearing}
        title="Clear all cached videos and embeddings"
      >
        {clearing ? 'Clearing...' : 'Clear Cache'}
      </button>
    </div>
  );
}
