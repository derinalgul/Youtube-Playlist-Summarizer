import { useState, useEffect } from 'react';
import { getStats } from '../services/api';

export default function Stats({ refreshTrigger }) {
  const [stats, setStats] = useState(null);

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
    </div>
  );
}
