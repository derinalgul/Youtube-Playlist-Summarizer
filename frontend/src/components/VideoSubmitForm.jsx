import { useState } from 'react';
import { submitVideo } from '../services/api';

export default function VideoSubmitForm({ onJobCreated }) {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    setError('');

    try {
      const job = await submitVideo(url);
      setUrl('');
      onJobCreated(job);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="submit-form">
      <h2>Add Videos</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste YouTube video or playlist URL..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !url.trim()}>
          {loading ? 'Processing...' : 'Submit'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
